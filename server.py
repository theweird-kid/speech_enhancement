from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

import librosa
import numpy as np
import soundfile as sf


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

import config as C
from data_tools import (
	audio_to_frames,
	frames_to_spectrograms,
	spectrograms_to_frames,
	scale_input,
	inv_scale_output,
)
from model import build_unet


def _load_model_runtime():
	"""Load the pretrained U-Net once at server startup."""
	weights_path = Path(C.PRETRAINED_WEIGHTS)
	if not weights_path.exists():
		raise RuntimeError(
			f"Pretrained weights not found: {weights_path}. "
			"Place a trained .h5 model there before starting the server."
		)
	return build_unet(pretrained_weights=str(weights_path))


def _run_enhancement(model, input_path: str) -> bytes:
	"""Run inference on one uploaded WAV and return the enhanced WAV bytes."""
	audio, _ = librosa.load(input_path, sr=C.SAMPLE_RATE)
	if audio.shape[0] < C.FRAME_LENGTH:
		raise ValueError(f"Audio is too short for inference. Need at least {C.FRAME_LENGTH} samples.")

	frames = audio_to_frames(audio, C.FRAME_LENGTH, C.HOP_LENGTH_FRAME)
	m_noisy_db, m_phase = frames_to_spectrograms(frames, C.DIM_SQUARE_SPEC, C.N_FFT, C.HOP_LENGTH_FFT)

	X_in = scale_input(m_noisy_db).astype(np.float32)[..., np.newaxis]
	X_pred = model.predict(X_in, verbose=0)

	noise_model = inv_scale_output(X_pred[..., 0])
	X_denoised = m_noisy_db - noise_model
	frames_denoised = spectrograms_to_frames(X_denoised, m_phase, C.HOP_LENGTH_FFT, C.FRAME_LENGTH)
	audio_out = frames_denoised.reshape(-1) * 10.0

	buffer = io.BytesIO()
	sf.write(buffer, audio_out, C.SAMPLE_RATE, format="WAV")
	buffer.seek(0)
	return buffer.read()


app = FastAPI(title="Speech Enhancement API", version="1.0.0")


@app.on_event("startup")
def startup_event() -> None:
	app.state.model = _load_model_runtime()


@app.get("/health")
def health() -> JSONResponse:
	loaded = hasattr(app.state, "model") and app.state.model is not None
	return JSONResponse({"status": "ok", "model_loaded": loaded})


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)) -> StreamingResponse:
	if not file.filename:
		raise HTTPException(status_code=400, detail="No file uploaded")

	suffix = Path(file.filename).suffix or ".wav"
	with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
		temp_input = tmp.name
		content = await file.read()
		tmp.write(content)

	try:
		enhanced_audio = _run_enhancement(app.state.model, temp_input)
		return StreamingResponse(
			io.BytesIO(enhanced_audio),
			media_type="audio/wav",
			headers={"Content-Disposition": f'attachment; filename="enhanced_{Path(file.filename).name}"'},
		)
	except HTTPException:
		raise
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc))
	finally:
		try:
			os.remove(temp_input)
		except OSError:
			pass


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
