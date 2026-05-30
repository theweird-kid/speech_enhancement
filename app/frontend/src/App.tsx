import { useEffect, useRef, useState, type ChangeEvent, type ReactNode } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  Activity,
  AlertTriangle,
  Bot,
  Ear,
  Mic,
  PanelTop,
  Play,
  Shield,
  Sparkles,
  Volume2,
  ChartNoAxesCombined,
} from 'lucide-react';

type Page = 'landing' | 'dashboard' | 'demo';

type StreamMetrics = {
  snr_before: number;
  snr_after: number;
  stoi: number;
  speech_confidence: number;
  voice_energy: number;
  noise_energy: number;
  clarity_score: number;
};

type DemoPreset = {
  id: string;
  label: string;
  noise_type: string;
  transcript: string;
};

type DemoResult = {
  transcript: string;
  noise_type: string;
  confidence: number;
  metrics: StreamMetrics;
  emergency: { detected: boolean; keyword: string | null };
  enhanced_wav_b64: string;
};

type UploadDenoiseResponse = {
  file_name: string;
  sample_rate: number;
  original_wav_b64: string;
  enhanced_wav_b64: string;
};

type CaptureStatus = 'idle' | 'recording' | 'uploading' | 'ready' | 'error';

const API_BASE = "http://localhost:8000";
const API = (path: string) => `${API_BASE}${path}`;
const TARGET_UPLOAD_SAMPLE_RATE = 16_000;
const MAX_RECORD_SECONDS = 180;

const DEMO_PRESETS: DemoPreset[] = [
  { id: 'traffic', label: 'Traffic Noise Demo', noise_type: 'Traffic', transcript: 'Please speak clearly into the microphone and stay near the entrance.' },
  { id: 'restaurant', label: 'Restaurant Noise Demo', noise_type: 'Restaurant', transcript: 'The meeting will start in five minutes near the front desk.' },
  { id: 'crowd', label: 'Crowd Noise Demo', noise_type: 'Crowd', transcript: 'Help is available at the reception if you need anything.' },
  { id: 'construction', label: 'Construction Noise Demo', noise_type: 'Construction', transcript: 'Take the right hallway to the lobby. Watch for an accident near the exit.' },
  { id: 'fan', label: 'Fan Noise Demo', noise_type: 'Fan', transcript: 'Emergency exit is on the left side of the hallway.' },
];

function App() {
  const [page, setPage] = useState<Page>(readPage());
  const [backendReady, setBackendReady] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [highContrast, setHighContrast] = useState(false);

  const [captureStatus, setCaptureStatus] = useState<CaptureStatus>('idle');
  const [captureMessage, setCaptureMessage] = useState('Press record, speak for a few seconds, then stop.');
  const [captureError, setCaptureError] = useState('');
  const [captureDuration, setCaptureDuration] = useState(0);
  const [originalAudioUrl, setOriginalAudioUrl] = useState('');
  const [enhancedAudioUrl, setEnhancedAudioUrl] = useState('');
  const [uploadBusy, setUploadBusy] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState('');

  const [demoPreset, setDemoPreset] = useState(DEMO_PRESETS[0].id);
  const [demoLoading, setDemoLoading] = useState(false);
  const [demoResult, setDemoResult] = useState<DemoResult | null>(null);
  const [demoOriginalUrl, setDemoOriginalUrl] = useState('');
  const [demoEnhancedUrl, setDemoEnhancedUrl] = useState('');

  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const silentGainRef = useRef<GainNode | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const sampleRateRef = useRef(48_000);
  const stopTimerRef = useRef<number | null>(null);
  const recordingActiveRef = useRef(false);

  useEffect(() => {
    const handleHash = () => setPage(readPage());
    window.addEventListener('hashchange', handleHash);
    return () => window.removeEventListener('hashchange', handleHash);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.contrast = highContrast ? 'high' : 'normal';
  }, [highContrast]);

  useEffect(() => {
    void fetchBackendHealth();
  }, []);

  useEffect(() => {
    const sample = DEMO_PRESETS.find((item) => item.id === demoPreset);
    if (sample) {
      void loadDemo(sample.id);
    }
  }, [demoPreset]);

  useEffect(() => {
    return () => {
      if (stopTimerRef.current !== null) {
        window.clearTimeout(stopTimerRef.current);
        stopTimerRef.current = null;
      }
      void stopCapture(true);
      if (originalAudioUrl) URL.revokeObjectURL(originalAudioUrl);
      if (enhancedAudioUrl) URL.revokeObjectURL(enhancedAudioUrl);
      if (demoOriginalUrl) URL.revokeObjectURL(demoOriginalUrl);
      if (demoEnhancedUrl) URL.revokeObjectURL(demoEnhancedUrl);
    };
  }, []);

  async function fetchBackendHealth() {
    try {
      const response = await fetch(API('/health'));
      const health = await response.json();
      setBackendReady(true);
      setModelLoaded(Boolean(health.model_loaded));
    } catch (error) {
      console.error(error);
      setBackendReady(false);
      setModelLoaded(false);
    }
  }

  async function startCapture() {
    if (captureStatus === 'recording' || captureStatus === 'uploading') return;

    setCaptureError('');
    setCaptureMessage('Recording sample...');
    setCaptureStatus('recording');
    setCaptureDuration(0);
    recordingActiveRef.current = true;

    if (originalAudioUrl) {
      URL.revokeObjectURL(originalAudioUrl);
      setOriginalAudioUrl('');
    }
    if (enhancedAudioUrl) {
      URL.revokeObjectURL(enhancedAudioUrl);
      setEnhancedAudioUrl('');
    }

    chunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      const context = new AudioContext();
      await context.resume();

      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);
      const silentGain = context.createGain();
      silentGain.gain.value = 0;

      processor.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        chunksRef.current.push(new Float32Array(input));
        setCaptureDuration((current) => {
          const next = current + input.length / context.sampleRate;
          if (next >= MAX_RECORD_SECONDS) {
            recordingActiveRef.current = false;
            window.setTimeout(() => {
              void stopCapture();
            }, 0);
          }
          return next;
        });
      };

      source.connect(processor);
      processor.connect(silentGain);
      silentGain.connect(context.destination);

      streamRef.current = stream;
      audioContextRef.current = context;
      sourceRef.current = source;
      processorRef.current = processor;
      silentGainRef.current = silentGain;
      sampleRateRef.current = context.sampleRate;
      stopTimerRef.current = window.setTimeout(() => {
        if (recordingActiveRef.current) {
          recordingActiveRef.current = false;
          void stopCapture();
        }
      }, MAX_RECORD_SECONDS * 1000);
    } catch (error) {
      console.error(error);
      setCaptureStatus('error');
      setCaptureMessage('Microphone access failed.');
      setCaptureError('Could not access the microphone. Check permissions and try again.');
      await stopCapture(true);
    }
  }

  async function stopCapture(skipUpload = false) {
    recordingActiveRef.current = false;
    const hadRecording = captureStatus === 'recording';

    try {
      if (stopTimerRef.current !== null) {
        window.clearTimeout(stopTimerRef.current);
        stopTimerRef.current = null;
      }
      processorRef.current?.disconnect();
      silentGainRef.current?.disconnect();
      sourceRef.current?.disconnect();
      processorRef.current = null;
      silentGainRef.current = null;
      sourceRef.current = null;

      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;

      if (audioContextRef.current) {
        await audioContextRef.current.close();
      }
      audioContextRef.current = null;
    } catch (error) {
      console.error(error);
    }

    if (skipUpload || !hadRecording) {
      if (!skipUpload && captureStatus !== 'uploading') {
        setCaptureStatus('idle');
        setCaptureMessage('Press record, speak for a few seconds, then stop.');
      }
      return;
    }

    const recorded = mergeFloat32Chunks(chunksRef.current);
    chunksRef.current = [];

    if (recorded.length === 0) {
      setCaptureStatus('error');
      setCaptureMessage('No audio captured.');
      setCaptureError('Record a sample before stopping.');
      return;
    }

    const sampleRate = sampleRateRef.current;
    const downsampled = sampleRate > TARGET_UPLOAD_SAMPLE_RATE
      ? downsampleFloat32(recorded, sampleRate, TARGET_UPLOAD_SAMPLE_RATE)
      : recorded;
    const uploadSampleRate = sampleRate > TARGET_UPLOAD_SAMPLE_RATE ? TARGET_UPLOAD_SAMPLE_RATE : sampleRate;
    const originalBlob = float32ToWavBlob(downsampled, uploadSampleRate);
    const originalUrl = URL.createObjectURL(originalBlob);

    try {
      setCaptureStatus('uploading');
      setCaptureMessage('Uploading to backend...');

      const formData = new FormData();
      formData.append('file', originalBlob, 'sample.wav');

      const response = await fetch(API('/enhance'), {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`);
      }

      const enhancedBlob = await response.blob();
      const enhancedUrl = URL.createObjectURL(enhancedBlob);

      if (originalAudioUrl) URL.revokeObjectURL(originalAudioUrl);
      if (enhancedAudioUrl) URL.revokeObjectURL(enhancedAudioUrl);

      setOriginalAudioUrl(originalUrl);
      setEnhancedAudioUrl(enhancedUrl);
      setCaptureStatus('ready');
      setCaptureMessage('Denoised audio is ready.');
      setCaptureDuration(downsampled.length / uploadSampleRate);
    } catch (error) {
      console.error(error);
      URL.revokeObjectURL(originalUrl);
      setCaptureStatus('error');
      setCaptureMessage('Backend processing failed.');
      setCaptureError('The denoising request failed. Check the backend and try again.');
    }
  }

  async function toggleCapture() {
    if (captureStatus === 'recording') {
      await stopCapture();
      return;
    }
    await startCapture();
  }

  async function loadDemo(sampleId: string) {
    setDemoLoading(true);
    try {
      const sampleResponse = await fetch(API(`/api/samples/${sampleId}`));
      const sampleBlob = await sampleResponse.blob();
      const originalUrl = URL.createObjectURL(sampleBlob);

      const formData = new FormData();
      formData.append('file', sampleBlob, `${sampleId}.wav`);
      const analyzeResponse = await fetch(API('/api/analyze'), {
        method: 'POST',
        body: formData,
      });
      const result = (await analyzeResponse.json()) as DemoResult;

      const enhancedBlob = wavBlobFromBase64(result.enhanced_wav_b64);
      const enhancedUrl = URL.createObjectURL(enhancedBlob);

      if (demoOriginalUrl) URL.revokeObjectURL(demoOriginalUrl);
      if (demoEnhancedUrl) URL.revokeObjectURL(demoEnhancedUrl);
      setDemoOriginalUrl(originalUrl);
      setDemoEnhancedUrl(enhancedUrl);
      setDemoResult(result);
    } catch (error) {
      console.error(error);
    } finally {
      setDemoLoading(false);
    }
  }

  async function onUploadAudio(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    setCaptureError('');
    setCaptureStatus('uploading');
    setCaptureMessage('Uploading selected file to backend...');
    setUploadBusy(true);
    setUploadedFileName(file.name);

    try {
      const formData = new FormData();
      formData.append('file', file, file.name);
      const response = await fetch(API('/api/denoise-upload'), {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`);
      }

      const payload = (await response.json()) as UploadDenoiseResponse;
      const originalBlob = wavBlobFromBase64(payload.original_wav_b64);
      const enhancedBlob = wavBlobFromBase64(payload.enhanced_wav_b64);
      const originalUrl = URL.createObjectURL(originalBlob);
      const enhancedUrl = URL.createObjectURL(enhancedBlob);

      if (originalAudioUrl) URL.revokeObjectURL(originalAudioUrl);
      if (enhancedAudioUrl) URL.revokeObjectURL(enhancedAudioUrl);

      setOriginalAudioUrl(originalUrl);
      setEnhancedAudioUrl(enhancedUrl);
      setCaptureStatus('ready');
      setCaptureMessage(`Denoised file ready: ${payload.file_name}`);
      setCaptureDuration(0);
    } catch (error) {
      console.error(error);
      setCaptureStatus('error');
      setCaptureMessage('File upload processing failed.');
      setCaptureError('Could not denoise the uploaded file. Please try another file.');
    } finally {
      setUploadBusy(false);
      event.target.value = '';
    }
  }

  return (
    <div className="hearai-shell min-h-screen text-slate-100">
      <header className="sticky top-0 z-50 border-b border-white/10 bg-slate-950/60 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-6 lg:px-8">
          <button className="flex items-center gap-3 focus-ring" onClick={() => navigate('#/')}>
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-cyan-400/15 text-cyan-300 ring-1 ring-cyan-400/30">
              <Ear size={20} />
            </div>
            <div className="text-left">
              <div className="text-lg font-semibold tracking-wide text-white">HearAI</div>
              <div className="text-xs uppercase tracking-[0.28em] text-cyan-200/80">AI hearing assistance</div>
            </div>
          </button>

          <nav className="hidden items-center gap-2 md:flex">
            <NavButton active={page === 'landing'} onClick={() => navigate('#/')}>Landing</NavButton>
            <NavButton active={page === 'dashboard'} onClick={() => navigate('#/dashboard')}>Dashboard</NavButton>
            <NavButton active={page === 'demo'} onClick={() => navigate('#/demo')}>Demo</NavButton>
          </nav>

          <div className="flex items-center gap-2">
            <Pill tone={backendReady ? 'positive' : 'warning'}>{backendReady ? 'Backend Ready' : 'Checking Backend'}</Pill>
            <Pill tone={modelLoaded ? 'positive' : 'neutral'}>{modelLoaded ? 'Model Ready' : 'Fallback Mode'}</Pill>
          </div>
        </div>
      </header>

      <main>
        <AnimatePresence mode="wait">
          {page === 'landing' && <LandingPage key="landing" />}
          {page === 'dashboard' && (
            <DashboardPage
              key="dashboard"
              captureStatus={captureStatus}
              captureMessage={captureMessage}
              captureError={captureError}
              captureDuration={captureDuration}
              originalAudioUrl={originalAudioUrl}
              enhancedAudioUrl={enhancedAudioUrl}
              backendReady={backendReady}
              modelLoaded={modelLoaded}
              highContrast={highContrast}
              onHighContrastChange={setHighContrast}
              onToggleCapture={toggleCapture}
              onUploadAudio={onUploadAudio}
              uploadBusy={uploadBusy}
              uploadedFileName={uploadedFileName}
            />
          )}
          {page === 'demo' && (
            <DemoPage
              key="demo"
              demoPreset={demoPreset}
              demoLoading={demoLoading}
              demoResult={demoResult}
              demoOriginalUrl={demoOriginalUrl}
              demoEnhancedUrl={demoEnhancedUrl}
              onSelectPreset={setDemoPreset}
              presets={DEMO_PRESETS}
            />
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

function LandingPage() {
  return (
    <motion.section initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.35 }} className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8 lg:py-16">
      <div className="grid gap-8 lg:grid-cols-[1.15fr_0.85fr] lg:items-center">
        <div className="space-y-8">
          <Pill tone="info">HearAI prototype</Pill>
          <div className="space-y-5">
            <h1 className="max-w-3xl text-5xl font-semibold tracking-tight text-white sm:text-6xl lg:text-7xl">
              Hear Better.
              <span className="block bg-gradient-to-r from-cyan-300 to-sky-400 bg-clip-text text-transparent">Understand More.</span>
            </h1>
            <p className="max-w-2xl text-lg leading-8 text-slate-300 sm:text-xl">
              A simple prototype that records a microphone sample, sends it to the backend, and returns a denoised playback clip.
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <PrimaryButton onClick={() => navigate('#/dashboard')} icon={<Mic size={18} />}>Record Sample</PrimaryButton>
            <SecondaryButton onClick={() => navigate('#/demo')} icon={<Play size={18} />}>Try Demo</SecondaryButton>
          </div>

          <div className="grid gap-4 sm:grid-cols-3">
            <LandingFeature title="Record" value="Mic -> Upload -> Denoise" icon={<Activity size={18} />} />
            <LandingFeature title="Captions" value="Demo mode with transcripts" icon={<Bot size={18} />} />
            <LandingFeature title="Clarity" value="Noise-aware analysis" icon={<ChartNoAxesCombined size={18} />} />
          </div>
        </div>

        <div className="glass relative overflow-hidden rounded-[2rem] p-6 shadow-glass">
          <div className="absolute inset-0 bg-hearai-radial opacity-80" />
          <div className="relative space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs uppercase tracking-[0.28em] text-cyan-200/80">Signal flow</div>
                <div className="mt-1 text-2xl font-semibold text-white">Record, upload, enhance</div>
              </div>
              <div className="rounded-full border border-cyan-400/20 bg-cyan-400/10 px-3 py-1 text-xs text-cyan-100">Prototype</div>
            </div>

            <div className="grid grid-cols-4 gap-3">
              {[22, 48, 76, 34, 58, 92, 44, 68, 88, 42, 56, 80].map((height, index) => (
                <div key={index} className="flex h-28 items-end rounded-2xl border border-white/10 bg-white/5 p-3">
                  <div className="w-full rounded-full bg-gradient-to-t from-cyan-400 via-sky-400 to-blue-500 animate-pulseWave" style={{ height: `${height}%`, animationDelay: `${index * 90}ms` }} />
                </div>
              ))}
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <StatCard label="Microphone capture" value="One-button record" />
              <StatCard label="Backend output" value="Denoised WAV playback" />
              <StatCard label="Processing mode" value="TensorFlow or fallback" />
              <StatCard label="Demo flow" value="Pre-recorded samples" />
            </div>
          </div>
        </div>
      </div>
    </motion.section>
  );
}

type DashboardPageProps = {
  captureStatus: CaptureStatus;
  captureMessage: string;
  captureError: string;
  captureDuration: number;
  originalAudioUrl: string;
  enhancedAudioUrl: string;
  backendReady: boolean;
  modelLoaded: boolean;
  highContrast: boolean;
  onHighContrastChange: (value: boolean) => void;
  onToggleCapture: () => void;
  onUploadAudio: (event: ChangeEvent<HTMLInputElement>) => void;
  uploadBusy: boolean;
  uploadedFileName: string;
};

function DashboardPage(props: DashboardPageProps) {
  return (
    <motion.section initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }} className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <div className="mb-8 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="text-sm uppercase tracking-[0.32em] text-cyan-200/80">Main Dashboard</div>
          <h2 className="mt-2 text-3xl font-semibold text-white sm:text-4xl">Record a sample and denoise it</h2>
          <p className="mt-2 max-w-3xl text-slate-300">Click once to start recording, click again to upload the sample to the backend, and play back the enhanced audio.</p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <PrimaryButton onClick={props.onToggleCapture} icon={<Mic size={18} />}>
            {props.captureStatus === 'recording' ? 'Stop and Denoise' : props.captureStatus === 'uploading' ? 'Processing...' : 'Record Sample'}
          </PrimaryButton>
          <label className="focus-ring inline-flex cursor-pointer items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-5 py-3 text-sm font-semibold text-slate-100 transition hover:bg-white/10">
            Upload Audio
            <input className="hidden" type="file" accept="audio/*" onChange={props.onUploadAudio} disabled={props.uploadBusy || props.captureStatus === 'recording'} />
          </label>
          <button className="glass rounded-2xl px-4 py-3 text-sm font-medium text-slate-100 focus-ring" onClick={() => props.onHighContrastChange(!props.highContrast)}>
            {props.highContrast ? 'High Contrast On' : 'High Contrast Off'}
          </button>
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        <div className="space-y-6">
          <Card title="Capture Status" icon={<Mic size={18} />}>
            <div className="grid gap-4 md:grid-cols-2">
              <ControlTile label="Backend" value={props.backendReady ? 'Online' : 'Offline'} status={props.backendReady ? 'positive' : 'warning'} />
              <ControlTile label="Model" value={props.modelLoaded ? 'Loaded' : 'Fallback mode'} status={props.modelLoaded ? 'positive' : 'warning'} />
              <ControlTile label="Status" value={props.captureStatus.toUpperCase()} status={props.captureStatus === 'ready' ? 'positive' : props.captureStatus === 'error' ? 'warning' : 'neutral'} />
              <ControlTile label="Captured" value={`${props.captureDuration.toFixed(1)}s`} status="info" />
            </div>

            <div className="mt-6 rounded-3xl border border-white/10 bg-white/5 p-4 text-sm text-slate-300">
              {props.captureMessage}
            </div>
            {props.uploadedFileName && (
              <div className="mt-4 rounded-3xl border border-white/10 bg-black/20 p-4 text-sm text-slate-300">
                Selected file: {props.uploadedFileName}
              </div>
            )}
            {props.captureError && (
              <div className="mt-4 rounded-3xl border border-rose-400/30 bg-rose-500/10 p-4 text-sm text-rose-100">
                {props.captureError}
              </div>
            )}
          </Card>

          <div className="grid gap-6 md:grid-cols-2">
            <Card title="Original Audio" icon={<Shield size={18} />}>
              <AudioPanel label="Captured sample" url={props.originalAudioUrl} emptyLabel="No sample recorded yet." />
            </Card>

            <Card title="Denoised Output" icon={<Sparkles size={18} />}>
              <AudioPanel label="Backend result" url={props.enhancedAudioUrl} emptyLabel="Record a sample to get the enhanced result." accent />
            </Card>
          </div>
        </div>

        <div className="space-y-6">
          <Card title="How it works" icon={<Activity size={18} />}>
            <div className="space-y-3 text-sm text-slate-300">
              <div className="rounded-2xl border border-white/10 bg-white/5 p-4">1. Press record and speak into your microphone.</div>
              <div className="rounded-2xl border border-white/10 bg-white/5 p-4">2. Stop recording to upload the clip to the backend.</div>
              <div className="rounded-2xl border border-white/10 bg-white/5 p-4">3. The backend denoises the sample and returns a WAV file for playback.</div>
            </div>
          </Card>

          <Card title="Accessibility" icon={<Volume2 size={18} />}>
            <div className="grid gap-3 md:grid-cols-2">
              <Pill tone="info">Large buttons</Pill>
              <Pill tone="info">High contrast mode</Pill>
              <Pill tone="info">Simple workflow</Pill>
              <Pill tone="info">Browser native audio</Pill>
            </div>
          </Card>

          <Card title="Emergency Keyword Detection" icon={<AlertTriangle size={18} />}>
            <div className="rounded-3xl border border-white/10 bg-white/5 p-4 text-sm text-slate-300">
              The demo mode still shows the speech analysis pipeline and emergency keyword output from the backend.
            </div>
          </Card>
        </div>
      </div>
    </motion.section>
  );
}

type DemoPageProps = {
  demoPreset: string;
  demoLoading: boolean;
  demoResult: DemoResult | null;
  demoOriginalUrl: string;
  demoEnhancedUrl: string;
  onSelectPreset: (value: string) => void;
  presets: DemoPreset[];
};

function DemoPage(props: DemoPageProps) {
  return (
    <motion.section initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }} className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <div className="mb-8 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="text-sm uppercase tracking-[0.32em] text-cyan-200/80">Demo Mode</div>
          <h2 className="mt-2 text-3xl font-semibold text-white sm:text-4xl">Pre-recorded hearing scenarios</h2>
          <p className="mt-2 max-w-3xl text-slate-300">Compare original audio, enhanced audio, transcript, noise classification, and clarity score without a microphone.</p>
        </div>
      </div>

      <Card title="Sample Library" icon={<PanelTop size={18} />}>
        <div className="grid gap-3 md:grid-cols-5">
          {props.presets.map((preset) => (
            <button
              key={preset.id}
              className={`rounded-3xl border px-4 py-4 text-left focus-ring ${props.demoPreset === preset.id ? 'border-cyan-400/40 bg-cyan-400/10 text-cyan-50' : 'border-white/10 bg-white/5 text-slate-200 hover:bg-white/10'}`}
              onClick={() => props.onSelectPreset(preset.id)}
            >
              <div className="text-sm font-semibold">{preset.label}</div>
              <div className="mt-1 text-xs text-slate-400">{preset.noise_type}</div>
            </button>
          ))}
        </div>
      </Card>

      <div className="mt-6 grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
        <Card title="Original vs Enhanced" icon={<Activity size={18} />}>
          <div className="space-y-4">
            <AudioPanel label="Original Audio" url={props.demoOriginalUrl} emptyLabel="Select a sample to load audio." />
            <AudioPanel label="Enhanced Audio" url={props.demoEnhancedUrl} emptyLabel="Select a sample to generate the enhanced output." accent />
          </div>
          {props.demoLoading && <div className="mt-4 text-sm text-cyan-200">Processing sample...</div>}
        </Card>

        <div className="space-y-6">
          <Card title="Transcript" icon={<Bot size={18} />}>
            <div className="rounded-3xl border border-white/10 bg-black/20 p-5 text-slate-100">
              {props.demoResult?.transcript || 'Select a sample to generate captions.'}
            </div>
            {props.demoResult?.emergency.detected && (
              <div className="mt-4 rounded-3xl border border-rose-400/30 bg-rose-500/10 p-4 text-rose-100">
                Potential Emergency Detected: {props.demoResult.emergency.keyword}
              </div>
            )}
          </Card>

          <Card title="Demo Metrics" icon={<ChartNoAxesCombined size={18} />}>
            <div className="grid gap-4 md:grid-cols-2">
              <MetricTile label="Noise" value={props.demoResult ? `${props.demoResult.noise_type} (${Math.round(props.demoResult.confidence * 100)}%)` : '---'} />
              <MetricTile label="Clarity score" value={props.demoResult ? `${props.demoResult.metrics.clarity_score.toFixed(1)}%` : '---'} />
              <MetricTile label="SNR after" value={props.demoResult ? `${props.demoResult.metrics.snr_after.toFixed(1)} dB` : '---'} />
              <MetricTile label="STOI" value={props.demoResult ? props.demoResult.metrics.stoi.toFixed(2) : '---'} />
            </div>
          </Card>
        </div>
      </div>
    </motion.section>
  );
}

function Card({ title, icon, children }: { title: string; icon: ReactNode; children: ReactNode }) {
  return (
    <div className="glass rounded-[2rem] p-5 sm:p-6">
      <div className="mb-5 flex items-center gap-3">
        <div className="rounded-2xl bg-cyan-400/10 p-2.5 text-cyan-200 ring-1 ring-cyan-400/20">{icon}</div>
        <h3 className="text-lg font-semibold text-white">{title}</h3>
      </div>
      {children}
    </div>
  );
}

function Pill({ tone, children }: { tone: 'positive' | 'warning' | 'neutral' | 'info'; children: ReactNode }) {
  const classes = {
    positive: 'border-emerald-400/30 bg-emerald-400/10 text-emerald-100',
    warning: 'border-amber-400/30 bg-amber-400/10 text-amber-100',
    neutral: 'border-white/10 bg-white/5 text-slate-200',
    info: 'border-cyan-400/30 bg-cyan-400/10 text-cyan-100',
  }[tone];
  return <span className={`inline-flex rounded-full border px-3 py-1 text-xs uppercase tracking-[0.22em] ${classes}`}>{children}</span>;
}

function NavButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button className={`rounded-full px-4 py-2 text-sm font-medium focus-ring ${active ? 'bg-cyan-400/15 text-cyan-100 ring-1 ring-cyan-300/30' : 'text-slate-300 hover:bg-white/5'}`} onClick={onClick}>
      {children}
    </button>
  );
}

function PrimaryButton({ onClick, icon, children }: { onClick: () => void; icon: ReactNode; children: ReactNode }) {
  return (
    <button className="focus-ring inline-flex items-center gap-2 rounded-2xl bg-gradient-to-r from-cyan-400 to-sky-500 px-5 py-3 text-sm font-semibold text-slate-950 shadow-lg shadow-cyan-950/40 transition hover:brightness-110" onClick={onClick}>
      {icon}
      {children}
    </button>
  );
}

function SecondaryButton({ onClick, icon, children }: { onClick: () => void; icon: ReactNode; children: ReactNode }) {
  return (
    <button className="focus-ring inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-5 py-3 text-sm font-semibold text-slate-100 transition hover:bg-white/10" onClick={onClick}>
      {icon}
      {children}
    </button>
  );
}

function LandingFeature({ title, value, icon }: { title: string; value: string; icon: ReactNode }) {
  return (
    <div className="glass rounded-3xl p-4">
      <div className="flex items-center gap-2 text-cyan-200">
        {icon}
        <span className="text-sm font-medium uppercase tracking-[0.22em] text-cyan-100/80">{title}</span>
      </div>
      <div className="mt-3 text-sm text-slate-300">{value}</div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-3xl border border-white/10 bg-white/5 p-4">
      <div className="text-xs uppercase tracking-[0.22em] text-slate-400">{label}</div>
      <div className="mt-2 text-sm font-medium text-white">{value}</div>
    </div>
  );
}

function ControlTile({ label, value, status }: { label: string; value: string; status: 'positive' | 'warning' | 'neutral' | 'info' }) {
  return (
    <div className="rounded-3xl border border-white/10 bg-white/5 p-4">
      <div className="text-xs uppercase tracking-[0.22em] text-slate-400">{label}</div>
      <div className="mt-2 flex items-center gap-2">
        <div className="text-lg font-semibold text-white">{value}</div>
        <Pill tone={status}>{status}</Pill>
      </div>
    </div>
  );
}

function MetricTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-3xl border border-white/10 bg-black/20 p-4">
      <div className="text-xs uppercase tracking-[0.22em] text-slate-400">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-white">{value}</div>
    </div>
  );
}

function AudioPanel({ label, url, emptyLabel, accent = false }: { label: string; url: string; emptyLabel: string; accent?: boolean }) {
  return (
    <div className={`rounded-3xl border p-4 ${accent ? 'border-cyan-400/25 bg-cyan-400/10' : 'border-white/10 bg-white/5'}`}>
      <div className="mb-3 flex items-center justify-between">
        <div className="text-sm font-medium text-white">{label}</div>
        <Pill tone={accent ? 'info' : 'neutral'}>{accent ? 'Enhanced' : 'Original'}</Pill>
      </div>
      {url ? <audio className="w-full" controls src={url} /> : <div className="text-sm text-slate-400">{emptyLabel}</div>}
    </div>
  );
}

function readPage(): Page {
  const hash = window.location.hash.replace('#', '');
  if (hash === '/dashboard') return 'dashboard';
  if (hash === '/demo') return 'demo';
  return 'landing';
}

function navigate(hash: string) {
  window.location.hash = hash;
}

function wavBlobFromBase64(base64: string) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Blob([bytes], { type: 'audio/wav' });
}

function mergeFloat32Chunks(chunks: Float32Array[]) {
  const totalLength = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function float32ToWavBlob(samples: Float32Array, sampleRate: number) {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  let offset = 0;
  const writeString = (value: string) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
    offset += value.length;
  };
  const writeUint32 = (value: number) => {
    view.setUint32(offset, value, true);
    offset += 4;
  };
  const writeUint16 = (value: number) => {
    view.setUint16(offset, value, true);
    offset += 2;
  };

  writeString('RIFF');
  writeUint32(36 + dataSize);
  writeString('WAVE');
  writeString('fmt ');
  writeUint32(16);
  writeUint16(1);
  writeUint16(1);
  writeUint32(sampleRate);
  writeUint32(byteRate);
  writeUint16(blockAlign);
  writeUint16(16);
  writeString('data');
  writeUint32(dataSize);

  const pcm16 = new Int16Array(buffer, 44);
  for (let index = 0; index < samples.length; index += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[index]));
    pcm16[index] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

function downsampleFloat32(samples: Float32Array, sourceRate: number, targetRate: number) {
  if (sourceRate <= targetRate || samples.length === 0) {
    return samples;
  }
  const ratio = sourceRate / targetRate;
  const targetLength = Math.max(1, Math.floor(samples.length / ratio));
  const downsampled = new Float32Array(targetLength);
  for (let index = 0; index < targetLength; index += 1) {
    const start = Math.floor(index * ratio);
    const end = Math.min(samples.length, Math.floor((index + 1) * ratio));
    let sum = 0;
    let count = 0;
    for (let sampleIndex = start; sampleIndex < end; sampleIndex += 1) {
      sum += samples[sampleIndex];
      count += 1;
    }
    downsampled[index] = count > 0 ? sum / count : samples[start] ?? 0;
  }
  return downsampled;
}

export default App;