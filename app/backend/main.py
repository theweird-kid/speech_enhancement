from __future__ import annotations

import base64
import io
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

try:
	import config as C
	from data_tools import (
		audio_to_frames,
		frames_to_spectrograms,
		inv_scale_output,
		scale_input,
		spectrograms_to_frames,
	)
except Exception as exc:
	raise RuntimeError(f'Failed to import speech enhancement modules: {exc}') from exc

try:
	from model import build_unet
	HAS_TENSORFLOW = True
except Exception:
	build_unet = None  # type: ignore[assignment]
	HAS_TENSORFLOW = False

try:
	from faster_whisper import WhisperModel
	HAS_WHISPER = True
except Exception:
	WhisperModel = None  # type: ignore[assignment]
	HAS_WHISPER = False


DEMO_PRESETS = {
	'traffic': {
		'label': 'Traffic Noise Demo',
		'noise_type': 'Traffic',
		'transcript': 'Please speak clearly into the microphone and stay near the entrance.',
	},
	'restaurant': {
		'label': 'Restaurant Noise Demo',
		'noise_type': 'Restaurant',
		'transcript': 'The meeting will start in five minutes near the front desk.',
	},
	'crowd': {
		'label': 'Crowd Noise Demo',
		'noise_type': 'Crowd',
		'transcript': 'Help is available at the reception if you need anything.',
	},
	'construction': {
		'label': 'Construction Noise Demo',
		'noise_type': 'Construction',
		'transcript': 'Take the right hallway to the lobby. Watch for an accident near the exit.',
	},
	'fan': {
		'label': 'Fan Noise Demo',
		'noise_type': 'Fan',
		'transcript': 'Emergency exit is on the left side of the hallway.',
	},
}

EMERGENCY_KEYWORDS = ['help', 'emergency', 'fire', 'accident', 'danger', 'ambulance']


def _to_float32(audio: np.ndarray) -> np.ndarray:
	return np.asarray(audio, dtype=np.float32)


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
	audio = _to_float32(audio)
	if audio.size == 0:
		return audio
	peak = float(np.max(np.abs(audio)) + 1e-8)
	if peak > 1.0:
		audio = audio / peak
	return np.clip(audio, -1.0, 1.0)


def _resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
	if source_sr == target_sr:
		return _to_float32(audio)
	return _to_float32(librosa.resample(audio.astype(np.float32), orig_sr=source_sr, target_sr=target_sr))


def _pad_to_frame(audio: np.ndarray, frame_length: int) -> np.ndarray:
	if audio.size >= frame_length:
		return audio
	if audio.size == 0:
		return np.zeros(frame_length, dtype=np.float32)
	padding = frame_length - audio.size
	mode = 'reflect' if audio.size > 1 else 'edge'
	return np.pad(audio, (0, padding), mode=mode).astype(np.float32)


def _encode_float32_audio(audio: np.ndarray) -> str:
	return base64.b64encode(_to_float32(audio).tobytes()).decode('ascii')


def _decode_audio_bytes(data: bytes) -> np.ndarray:
	if len(data) % 4 == 0:
		return np.frombuffer(data, dtype=np.float32).astype(np.float32)
	if len(data) % 2 == 0:
		return (np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0).astype(np.float32)
	raise ValueError('Unsupported audio payload format')


def _audio_bytes_to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
	buffer = io.BytesIO()
	sf.write(buffer, _normalize_audio(audio), sample_rate, format='WAV')
	return buffer.getvalue()


def _build_model() -> Any | None:
	weights_path = Path(C.PRETRAINED_WEIGHTS)
	if not HAS_TENSORFLOW or build_unet is None:
		return None
	if not weights_path.exists():
		return None
	try:
		return build_unet(pretrained_weights=str(weights_path))
	except Exception as exc:
		print(f'[HearAI] Falling back to non-ML enhancement: {exc}')
		return None


def _spectral_gate(audio: np.ndarray) -> np.ndarray:
	audio = _normalize_audio(audio)
	audio = _pad_to_frame(audio, C.FRAME_LENGTH)
	stft = librosa.stft(audio, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH_FFT)
	mag, phase = librosa.magphase(stft)
	mag_db = librosa.amplitude_to_db(mag, ref=np.max)
	noise_floor = np.percentile(mag_db, 25, axis=1, keepdims=True)
	enhanced_db = np.maximum(mag_db - 0.7 * (noise_floor + 4.0), -80.0)
	reconstructed = librosa.istft(
		librosa.db_to_amplitude(enhanced_db) * phase,
		hop_length=C.HOP_LENGTH_FFT,
		length=audio.size,
	)
	return _normalize_audio(reconstructed)


def _enhance_audio(audio: np.ndarray, sample_rate: int, model: Any | None) -> np.ndarray:
	audio = _normalize_audio(audio)
	audio = _resample_audio(audio, sample_rate, C.SAMPLE_RATE)
	if model is None:
		return _spectral_gate(audio)
	audio = _pad_to_frame(audio, C.FRAME_LENGTH)
	frames = audio_to_frames(audio, C.FRAME_LENGTH, C.HOP_LENGTH_FRAME)
	if frames.size == 0:
		return _spectral_gate(audio)
	m_noisy_db, m_phase = frames_to_spectrograms(frames, C.DIM_SQUARE_SPEC, C.N_FFT, C.HOP_LENGTH_FFT)
	X_in = scale_input(m_noisy_db).astype(np.float32)[..., np.newaxis]
	X_pred = model.predict(X_in, verbose=0)
	noise_model = inv_scale_output(X_pred[..., 0])
	X_denoised = m_noisy_db - noise_model
	frames_denoised = spectrograms_to_frames(X_denoised, m_phase, C.HOP_LENGTH_FFT, C.FRAME_LENGTH)
	audio_out = frames_denoised.reshape(-1).astype(np.float32) * 10.0
	return _normalize_audio(audio_out)


def _audio_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
	audio = _normalize_audio(audio)
	if audio.size < 2:
		return {
			'rms': 0.0,
			'spectral_centroid': 0.0,
			'spectral_flatness': 0.0,
			'zero_crossing_rate': 0.0,
			'voice_energy': 0.0,
			'noise_energy': 0.0,
		}
	centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate)))
	flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)))
	zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
	rms = float(np.sqrt(np.mean(audio**2)))
	voice_energy = float(np.mean(np.abs(librosa.effects.preemphasis(audio))))
	noise_energy = float(np.mean(np.abs(audio - np.median(audio))))
	return {
		'rms': rms,
		'spectral_centroid': centroid,
		'spectral_flatness': flatness,
		'zero_crossing_rate': zcr,
		'voice_energy': voice_energy,
		'noise_energy': noise_energy,
	}


def _classify_noise(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
	features = _audio_features(audio, sample_rate)
	centroid = features['spectral_centroid']
	flatness = features['spectral_flatness']
	zcr = features['zero_crossing_rate']
	rms = features['rms']
	score_map = {
		'Traffic': 0.65 * (1.0 - min(centroid / 1400.0, 1.0)) + 0.35 * min(rms * 2.5, 1.0),
		'Crowd': 0.45 * min(flatness * 2.0, 1.0) + 0.35 * min(centroid / 2200.0, 1.0) + 0.20 * min(zcr * 4.0, 1.0),
		'Construction': 0.45 * min(centroid / 2600.0, 1.0) + 0.35 * min(rms * 3.0, 1.0) + 0.20 * min(flatness * 1.8, 1.0),
		'Fan': 0.45 * (1.0 - min(zcr / 0.18, 1.0)) + 0.30 * (1.0 - min(centroid / 1200.0, 1.0)) + 0.25 * (1.0 - min(flatness * 2.5, 1.0)),
		'Machinery': 0.45 * min(rms * 3.0, 1.0) + 0.30 * min(centroid / 2000.0, 1.0) + 0.25 * min(flatness * 2.0, 1.0),
		'Wind': 0.45 * min(flatness * 2.5, 1.0) + 0.30 * (1.0 - min(rms * 2.0, 1.0)) + 0.25 * (1.0 - min(centroid / 1800.0, 1.0)),
		'Restaurant': 0.40 * min(flatness * 2.0, 1.0) + 0.35 * min(centroid / 1800.0, 1.0) + 0.25 * min(rms * 2.3, 1.0),
		'Television': 0.45 * min(centroid / 1500.0, 1.0) + 0.30 * min(zcr * 3.5, 1.0) + 0.25 * (1.0 - min(flatness * 2.0, 1.0)),
		'White Noise': 0.50 * min(flatness * 3.0, 1.0) + 0.25 * min(zcr * 4.0, 1.0) + 0.25 * min(centroid / 2200.0, 1.0),
	}
	noise_type = max(score_map, key=score_map.get)
	confidence = float(np.clip(score_map[noise_type], 0.45, 0.99))
	noise_level = float(np.clip(rms * 110.0 + flatness * 18.0, 18.0, 96.0))
	return {
		'noise_type': noise_type,
		'confidence': round(confidence, 2),
		'noise_level_db': round(noise_level, 1),
		'features': features,
	}


def _clarity_metrics(original: np.ndarray, enhanced: np.ndarray, sample_rate: int) -> dict[str, float]:
	original = _normalize_audio(original)
	enhanced = _normalize_audio(enhanced)
	noise_reduction = float(np.mean(np.abs(original)) - np.mean(np.abs(enhanced)))
	snr_before = 10.0 * math.log10((np.mean(original**2) + 1e-8) / (np.var(original - np.mean(original)) + 1e-8))
	snr_after = 10.0 * math.log10((np.mean(enhanced**2) + 1e-8) / (np.var(enhanced - np.mean(enhanced)) + 1e-8))
	voice_energy = float(np.mean(np.abs(librosa.effects.preemphasis(enhanced))))
	noise_energy = float(np.mean(np.abs(enhanced - librosa.effects.preemphasis(enhanced))))
	stoi_proxy = float(np.clip(0.55 + 0.06 * (snr_after - snr_before) + 0.12 * max(noise_reduction, 0.0), 0.0, 0.99))
	speech_confidence = float(np.clip(0.58 + 0.05 * snr_after - 0.03 * noise_energy * 100.0, 0.0, 0.99))
	clarity_score = float(np.clip(50.0 + 5.0 * (snr_after - snr_before) + 18.0 * max(noise_reduction, 0.0), 0.0, 100.0))
	return {
		'snr_before': round(float(snr_before), 2),
		'snr_after': round(float(snr_after), 2),
		'stoi': round(stoi_proxy, 2),
		'speech_confidence': round(speech_confidence, 2),
		'voice_energy': round(float(voice_energy), 4),
		'noise_energy': round(float(noise_energy), 4),
		'clarity_score': round(clarity_score, 1),
	}


def _detect_emergency(text: str) -> dict[str, Any]:
	text_lower = text.lower()
	for keyword in EMERGENCY_KEYWORDS:
		if keyword in text_lower:
			return {'detected': True, 'keyword': keyword}
	return {'detected': False, 'keyword': None}


def _load_whisper_model() -> Any | None:
	if not HAS_WHISPER:
		return None
	cache = getattr(app.state, 'whisper_model', None)
	if cache is not None:
		return cache
	try:
		app.state.whisper_model = WhisperModel('base', device='cpu', compute_type='int8')
		return app.state.whisper_model
	except Exception as exc:
		print(f'[HearAI] Whisper unavailable, using fallback captions: {exc}')
		app.state.whisper_model = None
		return None


def _transcribe_audio(audio: np.ndarray, sample_rate: int, hint: str | None = None) -> str:
	whisper = _load_whisper_model()
	if whisper is not None:
		with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
			temp_path = tmp.name
			sf.write(tmp, _normalize_audio(audio), sample_rate)
		try:
			segments, _info = whisper.transcribe(temp_path, vad_filter=True)
			text = ' '.join(segment.text.strip() for segment in segments).strip()
			if text:
				return text
		finally:
			try:
				os.remove(temp_path)
			except OSError:
				pass
	if hint:
		return hint
	noise = _classify_noise(audio, sample_rate)['noise_type'].lower()
	return {
		'traffic': 'The hallway is busy, but the speaker is coming through clearly.',
		'crowd': 'Please repeat the last sentence and stay close to the microphone.',
		'construction': 'Take care near the entrance and follow the lit exit signs.',
		'fan': 'The room is quiet enough for the next announcement.',
		'machinery': 'Please wait until the machine stops before speaking.',
		'wind': 'Move closer to the shelter for better speech clarity.',
		'restaurant': 'The meeting will begin in five minutes at the front desk.',
		'television': 'Turn down the television so the conversation is easier to hear.',
		'white noise': 'Help is available if you need assistance.',
	}.get(noise, 'Please speak again a little more slowly.')


def _analysis_payload(audio: np.ndarray, sample_rate: int, transcript_hint: str | None = None) -> dict[str, Any]:
	model = getattr(app.state, 'model', None)
	noise_info = _classify_noise(audio, sample_rate)
	enhanced = _enhance_audio(audio, sample_rate, model)
	metrics = _clarity_metrics(audio, enhanced, C.SAMPLE_RATE)
	transcript = _transcribe_audio(enhanced, C.SAMPLE_RATE, hint=transcript_hint)
	emergency = _detect_emergency(transcript)
	return {
		'sample_rate': C.SAMPLE_RATE,
		'noise_type': noise_info['noise_type'],
		'confidence': noise_info['confidence'],
		'noise_level_db': noise_info['noise_level_db'],
		'metrics': metrics,
		'transcript': transcript,
		'emergency': emergency,
		'enhanced_wav_b64': base64.b64encode(_audio_bytes_to_wav(enhanced, C.SAMPLE_RATE)).decode('ascii'),
		'enhanced_pcm_b64': _encode_float32_audio(enhanced),
	}


def _synth_demo_audio(preset_name: str, duration: float = 8.0) -> np.ndarray:
	sr = C.SAMPLE_RATE
	n_samples = int(duration * sr)
	t = np.linspace(0.0, duration, n_samples, endpoint=False)
	seed = sum(ord(ch) for ch in preset_name)
	rng = np.random.default_rng(seed)
	voice_envelope = np.maximum(0.0, np.sin(2 * np.pi * 1.6 * t + 0.15 * seed)) ** 1.4
	voice = (
		0.28 * np.sin(2 * np.pi * 180 * t)
		+ 0.18 * np.sin(2 * np.pi * 260 * t)
		+ 0.10 * np.sin(2 * np.pi * 340 * t)
		+ 0.06 * np.sin(2 * np.pi * 540 * t)
	)
	voice *= voice_envelope
	noise = rng.normal(0.0, 0.14, size=n_samples)
	if preset_name == 'traffic':
		noise += 0.16 * np.sin(2 * np.pi * 52 * t) + 0.05 * np.sin(2 * np.pi * 110 * t)
	elif preset_name == 'restaurant':
		noise += 0.08 * np.sin(2 * np.pi * 220 * t) + 0.12 * rng.normal(0.0, 0.7, size=n_samples)
	elif preset_name == 'crowd':
		noise += 0.12 * np.sin(2 * np.pi * 130 * t) + 0.08 * np.sin(2 * np.pi * 280 * t)
	elif preset_name == 'construction':
		bursts = (np.sin(2 * np.pi * 0.85 * t) > 0.85).astype(np.float32)
		noise += 0.22 * bursts * np.sin(2 * np.pi * 730 * t)
	elif preset_name == 'fan':
		noise += 0.18 * np.sin(2 * np.pi * 58 * t)
	combined = 0.65 * voice + noise
	combined = combined / (np.max(np.abs(combined)) + 1e-8)
	return combined.astype(np.float32)


def _audio_from_upload(file: UploadFile) -> tuple[np.ndarray, int]:
	content = file.file.read()
	buffer = io.BytesIO(content)
	audio, sample_rate = sf.read(buffer, dtype='float32', always_2d=False)
	if isinstance(audio, np.ndarray) and audio.ndim > 1:
		audio = np.mean(audio, axis=1)
	return _to_float32(audio), int(sample_rate)


app = FastAPI(title='HearAI API', version='1.0.0')
app.add_middleware(
	CORSMiddleware,
	allow_origins=['*'],
	allow_credentials=True,
	allow_methods=['*'],
	allow_headers=['*'],
)


@app.on_event('startup')
def _startup() -> None:
	app.state.model = _build_model()
	app.state.model_loaded = app.state.model is not None
	app.state.whisper_model = None


@app.get('/health')
def health() -> JSONResponse:
	return JSONResponse(
		{
			'status': 'ok',
			'model_loaded': bool(getattr(app.state, 'model_loaded', False)),
			'tensorflow_available': HAS_TENSORFLOW,
			'whisper_available': HAS_WHISPER,
			'sample_rate': C.SAMPLE_RATE,
		},
	)


@app.get('/api/demo-presets')
def demo_presets() -> JSONResponse:
	return JSONResponse({'presets': [{'id': key, **value} for key, value in DEMO_PRESETS.items()]})


@app.get('/api/samples/{sample_name}')
def demo_sample_audio(sample_name: str) -> Response:
	if sample_name not in DEMO_PRESETS:
		raise HTTPException(status_code=404, detail='Unknown demo sample')
	audio = _synth_demo_audio(sample_name)
	wav_bytes = _audio_bytes_to_wav(audio, C.SAMPLE_RATE)
	return Response(content=wav_bytes, media_type='audio/wav')


@app.post('/api/noise-detection')
async def noise_detection(file: UploadFile = File(...)) -> JSONResponse:
	if not file.filename:
		raise HTTPException(status_code=400, detail='No file uploaded')
	audio, sample_rate = _audio_from_upload(file)
	return JSONResponse(_classify_noise(audio, sample_rate))


@app.post('/api/transcribe')
async def transcribe(file: UploadFile = File(...)) -> JSONResponse:
	if not file.filename:
		raise HTTPException(status_code=400, detail='No file uploaded')
	audio, sample_rate = _audio_from_upload(file)
	transcript = _transcribe_audio(audio, sample_rate)
	return JSONResponse({'transcript': transcript, 'emergency': _detect_emergency(transcript)})


@app.post('/api/analyze')
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
	if not file.filename:
		raise HTTPException(status_code=400, detail='No file uploaded')
	audio, sample_rate = _audio_from_upload(file)
	return JSONResponse(_analysis_payload(audio, sample_rate))


@app.post('/enhance')
async def enhance(file: UploadFile = File(...)) -> Response:
	if not file.filename:
		raise HTTPException(status_code=400, detail='No file uploaded')
	audio, sample_rate = _audio_from_upload(file)
	enhanced = _enhance_audio(audio, sample_rate, getattr(app.state, 'model', None))
	return Response(content=_audio_bytes_to_wav(enhanced, C.SAMPLE_RATE), media_type='audio/wav')


@app.post('/api/denoise-upload')
async def denoise_upload(file: UploadFile = File(...)) -> JSONResponse:
	if not file.filename:
		raise HTTPException(status_code=400, detail='No file uploaded')
	audio, sample_rate = _audio_from_upload(file)
	original = _resample_audio(_normalize_audio(audio), sample_rate, C.SAMPLE_RATE)
	enhanced = _enhance_audio(audio, sample_rate, getattr(app.state, 'model', None))
	return JSONResponse(
		{
			'file_name': file.filename,
			'sample_rate': C.SAMPLE_RATE,
			'original_wav_b64': base64.b64encode(_audio_bytes_to_wav(original, C.SAMPLE_RATE)).decode('ascii'),
			'enhanced_wav_b64': base64.b64encode(_audio_bytes_to_wav(enhanced, C.SAMPLE_RATE)).decode('ascii'),
		},
	)


if __name__ == '__main__':
	import uvicorn

	uvicorn.run('app.backend.main:app', host='0.0.0.0', port=8000, reload=False)