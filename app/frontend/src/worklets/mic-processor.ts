// Provide minimal AudioWorklet types for TypeScript (worklet runs in audio context)
declare abstract class AudioWorkletProcessor {
  port: any;
  constructor();
  process(inputs: any[][]): boolean;
}

declare function registerProcessor(name: string, ctor: any): void;

class MicCaptureProcessor extends AudioWorkletProcessor {
  private buffer = new Float32Array(0);
  private readonly chunkSize = 4096;

  process(inputs: Float32Array[][]): boolean {
    const input = inputs[0]?.[0];
    if (!input || input.length === 0) {
      return true;
    }

    const merged = new Float32Array(this.buffer.length + input.length);
    merged.set(this.buffer, 0);
    merged.set(input, this.buffer.length);
    this.buffer = merged;

    while (this.buffer.length >= this.chunkSize) {
      const chunk = this.buffer.slice(0, this.chunkSize);
      this.port.postMessage({ type: 'chunk', pcm: chunk }, [chunk.buffer]);
      this.buffer = this.buffer.slice(this.chunkSize);
    }

    return true;
  }
}

registerProcessor('mic-capture-processor', MicCaptureProcessor);