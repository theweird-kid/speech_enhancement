# Speech Enhancement — Simplified Colab Edition

A clean, minimal re-implementation of deep-learning speech enhancement using a **U-Net** that learns to subtract background noise from noisy speech spectrograms.

> Based on [vbelz/Speech-enhancement](https://github.com/vbelz/Speech-enhancement) — simplified for Google Colab with Google Drive storage.

---

## 🗂️ Project Structure

```
speech_enhancement_alt/
├── src/
│   ├── config.py          # All paths & hyper-parameters (edit DRIVE_ROOT here)
│   ├── data_tools.py      # Audio I/O, STFT, spectrogram helpers
│   ├── model.py           # U-Net architecture
│   ├── prepare_data.py    # Data-preparation pipeline
│   ├── train.py           # Training loop
│   └── predict.py         # Inference / denoising
├── notebooks/
│   ├── 01_Data_Preparation.ipynb   # Step 1: create spectrograms
│   ├── 02_Train_Model.ipynb        # Step 2: train U-Net
│   └── 03_Predict_Denoise.ipynb    # Step 3: denoise audio
└── requirements.txt
```

---

## 🚀 Quick Start (Google Colab)

### 1. Organise your Drive

```
MyDrive/speech_enhancement/
├── data/
│   ├── voice/    ← clean speech WAVs  (e.g. LibriSpeech)
│   └── noise/    ← background noise WAVs  (e.g. ESC-50)
```

### 2. Update `DRIVE_ROOT` in `src/config.py`

```python
DRIVE_ROOT = '/content/drive/MyDrive/speech_enhancement'
```

### 3. Update the `REPO` URL in each notebook

Replace `YOUR_USERNAME` with your GitHub username in the clone cell.

### 4. Run the notebooks in order

| Notebook | Purpose |
|---|---|
| `01_Data_Preparation.ipynb` | Load audio → blend → save spectrograms to Drive |
| `02_Train_Model.ipynb` | Load spectrograms → train U-Net → save .h5 model |
| `03_Predict_Denoise.ipynb` | Load .h5 model → denoise test WAVs → play & compare |

---

## 🧠 How It Works

```
Noisy voice WAV
      │
      ▼ STFT
Noisy spectrogram (128×128)
      │
      ▼ U-Net (predicts noise model)
Noise model spectrogram
      │
      ▼ Spectral subtraction
Clean spectrogram
      │
      ▼ iSTFT + original phase
Denoised WAV
```

The model targets **noise model = noisy - clean** (trained with Huber loss).  
At inference, the predicted noise is subtracted from the noisy spectrogram.

---

## 📦 Data Sources

| Type | Source |
|---|---|
| Clean speech | [LibriSpeech](http://www.openslr.org/12/) |
| Background noise | [ESC-50 dataset](https://github.com/karoldvl/ESC-50) |

---

## 🔧 Key Fixes vs. Original

| Issue | Fix |
|---|---|
| `librosa.output.write_wav` removed in librosa ≥ 0.10 | Replaced with `soundfile.write` |
| Indentation bug in `model_unet.py` | Fixed encoder/decoder block alignment |
| `librosa.get_duration(y, sr)` deprecation | Updated API call |
| Monolithic `args.py` | Replaced with clean `config.py` |
| No Colab Drive mount | Added to all notebooks |

---

## 📄 License

MIT — see [original repo](https://github.com/vbelz/Speech-enhancement).
# speech_enhancement
