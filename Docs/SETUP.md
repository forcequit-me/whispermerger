# WhisperMerger — Setup (native Windows)

v1 runs **natively on Windows**. The PRD chose WSL2+conda only to dodge native-Windows
WhisperX dependency hell, but CUDA torch already runs natively on this machine, so that
pain is already solved. WSL2 is kept as a fallback only (bottom of this doc).

## Verified machine target

| Item | Value |
|---|---|
| GPU | RTX 3080 Ti, 12 GB |
| Python | 3.12.7 |
| torch | 2.6.0+cu124 (`torch.cuda.is_available() == True`) |
| ffmpeg / ffprobe | 5.1, on PATH |

## Why these pins

- **whisperx==3.4.2** — requires `torch>=2.5.1` (no upper cap), so it keeps the working
  `torch 2.6.0+cu124`. The latest whisperx (3.8.x) pins `torch~=2.8.0` and would replace it.
- **torch / torchaudio 2.6.0+cu124** installed from the cu124 index, then `constraints.txt`
  blocks any transitive dep from bumping them.

## Install

```powershell
cd C:\Users\plagi\Desktop\whispermerger
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip

# 1. torch + torchaudio matched to CUDA 12.4
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 2. app deps; -c keeps torch pinned
pip install -r backend/requirements.txt -c backend/constraints.txt
```

## First-run model downloads

- `large-v3` (faster-whisper / CTranslate2) — ~3 GB, into `~/.cache/huggingface`
- wav2vec2 forced-alignment model — ~360 MB
- `ctranslate2 4.4.x` (whisperx caps `<4.5.0`) ships its own CUDA libs.
- `pyannote-audio` installs as a hard whisperx dep but is **unused** (diarization is NG1 —
  the stems approach replaces it).

## torch 2.6 + Windows gotchas (already handled)

whisperx 3.4.2 on torch 2.6.0+cu124 needs four fixes, all pinned/coded so a clean rebuild
just works:

1. **`setuptools<81`** — ctranslate2 4.4.0 imports `pkg_resources`, removed in setuptools 81.
2. **`transformers<5`** — transformers 5.x references `torch.float8_e8m0fnu` (torch 2.7+).
3. **`weights_only=False`** — torch 2.6 made `torch.load` default `weights_only=True`, which
   rejects the omegaconf globals in whisperx's pyannote VAD checkpoint. `transcribe._allow_full_torch_load`
   forces it off (checkpoints are from trusted HF repos).
4. **cuDNN 8 on PATH** — ctranslate2 4.4.0 needs cuDNN 8, but torch bundles cuDNN 9. We pin
   `nvidia-cudnn-cu12==8.9.7.29`; `transcribe._ensure_cuda_dlls` adds its bin dir (plus cuBLAS/
   nvRTC) to the DLL search path **and** PATH at import.

## GPU smoke test (the §11.3 gate)

```powershell
python backend\smoke_test.py            # optionally pass a short speech clip:
python backend\smoke_test.py path\to\clip.wav
```

Pass = whisperx imports, large-v3 loads on `cuda`/`float16` without OOM, and the alignment
pass prints per-word `{word, start, end}` timestamps.

## Fallback: WSL2 + conda

Only if installing whisperx ever fights the native torch: Ubuntu 24.04 under WSL2, conda
env with PyTorch+CUDA, then `pip install whisperx`. Documented here so the environment can
be rebuilt cleanly if the native path breaks (NFR-4).
