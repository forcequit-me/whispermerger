# WhisperMerger

Local desktop app that turns raw multi-track recordings (OBS MKV with mic + desktop
streams, or several audio files) into accurate, speaker-labeled transcripts with
**word-level** timestamps, via WhisperX on a local GPU. A FastAPI backend drives WhisperX
and a pure-Python merge engine; a vanilla HTML/JS frontend handles input, labeling,
editing, and export.

- **Product spec:** [`Docs/whispermerger-prd.md`](Docs/whispermerger-prd.md)
- **Build plan:** [`Docs/whispermerger-build-plan.md`](Docs/whispermerger-build-plan.md)
- **Setup:** [`Docs/SETUP.md`](Docs/SETUP.md)

Status: v1 in progress. Currently scaffolding + environment (build-plan step 1).
