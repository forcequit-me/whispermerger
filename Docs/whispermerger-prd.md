# WhisperMerger
### Product Requirements Document

| Field | Value |
|---|---|
| Status | Draft |
| Version | 1.1 |
| Author | Force Quit |
| Last updated | June 8, 2026 |
| Related work | Transcript Merger (existing web tool), Recording-to-Video Skill (downstream, see Appendix C) |

---

## 1. Summary

WhisperMerger is a local desktop application that turns raw multi-track recordings into accurate, speaker-labeled transcripts with word-level timestamps. It replaces the current "transcribe in DaVinci Resolve, export each track, drop into Transcript Merger" workflow with a single drag-and-drop pipeline that runs WhisperX locally on a GPU.

The existing Transcript Merger web tool stays as the general-purpose, no-install option. WhisperMerger is the local power-user version for anyone with a GPU who needs precise timestamps across long recordings.

---

## 2. Background and problem

Resolve transcription is segment-level only. On long sparse-speech sessions, such as multi-hour vibe-coding recordings with frequent gaps, segment timestamps drift by seconds. Finding a specific moment across a 5-plus-hour file becomes unusable. Word-level timestamps remove that drift and make real semantic search through long footage possible.

The current workflow also carries too many manual steps. To transcribe one multi-track recording today, the operator splits audio in Resolve, exports each track, transcribes each track separately, then drops the results into Transcript Merger. Every recording pays that cost. WhisperMerger collapses the whole sequence to three actions: drop the file, label the tracks, run.

---

## 3. Goals

| ID | Goal |
|---|---|
| G1 | Produce acoustically accurate word-level timestamps for recordings up to and beyond 5 hours. |
| G2 | Reduce the multi-track-to-transcript workflow to drag, label, run. |
| G3 | Reuse the Transcript Merger frontend and export logic wherever it already works. |
| G4 | Run entirely locally on the target GPU, with no cloud dependency. |
| G5 | Make repeated reprocessing cheap through output caching. |

---

## 4. Non-goals

| ID | Non-goal | Reason |
|---|---|---|
| NG1 | Model-based diarization in v1. | The stems-based recording approach separates speakers better than any diarization model. Single-track manual labeling is a planned v2 feature (Section 11.4). |
| NG2 | A packaged installer in v1. | v1 ships as a FastAPI server plus a browser frontend. A Tauri-wrapped .exe is future work. |
| NG3 | Cross-platform support beyond the target Windows plus WSL2 environment for v1. | Keeps the v1 environment a single known target. |
| NG4 | Replacing the Transcript Merger web tool. | The two products coexist and serve different users. |
| NG5 | Word-level click-to-edit in the UI. | Segments stay the editing unit. Editing individual words would be miserable to use. Word-level data is stored underneath and exported, not edited directly. |

---

## 5. Target user

The primary user is a solo content creator who records multi-track sessions (OBS multi-track MKV with separate mic and desktop audio streams) and needs them transcribed accurately for editing and search. This user has an RTX-class GPU and is comfortable running a local server in a browser.

The secondary user is anyone with a GPU who already relies on Transcript Merger and wants better timing without changing their editing habits.

---

## 6. Key use cases

| ID | Use case |
|---|---|
| UC1 | Drop a single OBS MKV with mic and desktop tracks, label the two tracks, get one merged transcript. |
| UC2 | Drop several sequential recordings (multiple files, one speaker each), confirm order, get one continuous merged transcript. |
| UC3 | Re-run the merge after tweaking settings, without re-transcribing unchanged audio. |
| UC4 | Edit a transcript that came from elsewhere (Resolve, Premiere, Descript) using the carried-over Edit mode. |
| UC5 | Export a clean SRT whose line splits land on real word boundaries. |

---

## 7. Solution overview

### 7.1 Approach

WhisperMerger runs a local FastAPI backend that drives WhisperX for transcription and a pure-Python merge engine for combining tracks. A vanilla HTML and JavaScript frontend, served on a known local port and opened in the browser, handles input, labeling, editing, and export. Most of that frontend ports directly from Transcript Merger.

### 7.2 Why WhisperX rather than vanilla Whisper

Vanilla Whisper derives word timestamps from cross-attention heuristics, which can drift by seconds. That drift is the exact failure the product exists to fix, so vanilla Whisper is unusable here.

WhisperX uses faster-whisper (CTranslate2) for transcription, then runs a separate wav2vec2 forced-alignment pass to produce acoustic-aligned word timestamps. Its VAD preprocessing strips silence at the transcription stage, which suits sparse-speech sessions precisely.

Alternatives considered and ruled out:

- **faster-whisper alone**: fast and accurate transcription, but word timestamps remain Whisper-quality, which is not good enough for navigation.
- **insanely-fast-whisper**: optimized for raw throughput on high-end GPUs, not for word-level accuracy.
- **WhisperX with built-in pyannote diarization**: skipped. The stems-based recording approach already solves diarization better than a model can.

---

## 8. Functional requirements

Priority key: **P0** is required for the first usable v1, **P1** is in v1 scope but does not block the first working pipeline, **P2** is v2 or later.

### 8.1 Input handling

| ID | Requirement | Priority |
|---|---|---|
| FR-IN-1 | Accept audio files (.wav, .mp3, .flac, and similar). One file maps to one speaker. | P0 |
| FR-IN-2 | Accept multi-track video files (.mkv, .mov, .mp4 with multiple audio streams). One stream maps to one speaker. | P0 |
| FR-IN-3 | Accept a mix of audio files and multi-track video files in a single drop. | P0 |
| FR-IN-4 | Run ffprobe on each input to enumerate audio streams and read duration. | P0 |
| FR-IN-5 | Extract and convert audio streams on the fly with ffmpeg to mono, 16 kHz, PCM s16le (the format WhisperX expects). See Appendix A. | P0 |
| FR-IN-6 | Write extracted audio to a temp directory and delete it after transcription completes. | P0 |

### 8.2 Track identification

| ID | Requirement | Priority |
|---|---|---|
| FR-TK-1 | Represent every input through one Input Source model (Section 9.1), so audio files and video files share the same data structure. | P0 |
| FR-TK-2 | Pre-fill track labels by OBS convention when an input looks like an OBS recording: Track 1 defaults to "Mic", Track 2 defaults to "Desktop". Labels stay editable. | P0 |
| FR-TK-3 | Provide a per-track audition button. On file load, extract a 5 to 10 second sample per track. A play control next to each track lets the user listen and confirm the label. | P1 |
| FR-TK-4 | Render the track selector only when an input exposes more than one audio stream. | P0 |

### 8.3 Transcription

| ID | Requirement | Priority |
|---|---|---|
| FR-TR-1 | Transcribe each selected track with WhisperX (large-v3) plus wav2vec2 forced alignment. | P0 |
| FR-TR-2 | Use compute type float16. | P0 |
| FR-TR-3 | Apply WhisperX VAD preprocessing to strip silence at the transcription stage. | P0 |
| FR-TR-4 | Show a progress indicator per file or track during transcription. | P1 |
| FR-TR-5 | Trigger the merge automatically once all transcriptions complete. | P0 |

### 8.4 Merge engine

The merge engine is a pure-Python module. It takes a list of WhisperX outputs plus speaker labels plus sequence order, and returns merged JSON (Section 9.2). It applies the following conflict hierarchy in order.

| ID | Requirement | Priority |
|---|---|---|
| FR-MG-1 | **No time overlap:** interleave segments by start time. | P0 |
| FR-MG-2 | **Backchannel** (segment B sits fully inside segment A, B is short, under N words): treat as an interjection. Honor a user preference to render it as a parenthetical, render it as a separate line, or skip it. | P0 |
| FR-MG-3 | **Genuine overlap** (both speakers talking simultaneously for several seconds): interleave at word boundaries by start time, and mark the overlapping range so the editor can highlight it. | P0 |
| FR-MG-4 | **Bleed** (similar text in both streams within a small time window, caused by a mic picking up desktop audio or the reverse): flag for review and prefer the louder source by computing RMS from the original audio at that timestamp. | P1 |
| FR-MG-5 | Surface conflicts in the UI and let the user choose the resolution. Do not over-automate. | P0 |

Note: bleed detection (FR-MG-4) is high value. Bleed happens on every recording with any voice over speakers and silently ruins merges when it is not caught. It carries P1 only because it needs threshold tuning on real recordings before it is reliable.

### 8.5 Word-level data

| ID | Requirement | Priority |
|---|---|---|
| FR-WD-1 | Store word-level timestamps under each segment, while keeping the segment as the editing unit. | P0 |
| FR-WD-2 | Preserve word-level timestamps in the JSON export. | P0 |
| FR-WD-3 | When subtitle limits force a long entry to split, split at real word timestamps rather than estimating from segment boundaries. | P1 |

### 8.6 Editing and conflict resolution (ported from Transcript Merger)

| ID | Requirement | Priority |
|---|---|---|
| FR-UI-1 | Inline edit: click an entry to edit speaker, text, or timing. | P0 |
| FR-UI-2 | Rename a speaker globally. | P0 |
| FR-UI-3 | Add and remove entries with smart closed-caption placement. | P0 |
| FR-UI-4 | Subtitle limit settings: characters per line, max lines, minimum duration. | P0 |
| FR-UI-5 | Output toggles: Show Speakers, Show Transcription, Show CC, Strip Silence. | P0 |
| FR-UI-6 | Per-file framerate and timecode handling (still relevant for Edit mode). | P0 |
| FR-UI-7 | Keyboard shortcuts: Enter, Ctrl+Enter, Escape, arrow up and down. | P0 |
| FR-UI-8 | Custom filename for exports. | P0 |
| FR-UI-9 | Per-tab settings preservation. | P0 |
| FR-UI-10 | Highlight conflict ranges (overlap, bleed, backchannel) in the editor using the `flags` field. | P1 |

### 8.7 Export

| ID | Requirement | Priority |
|---|---|---|
| FR-EX-1 | Export SRT, TXT, CSV, and JSON, reusing the existing Transcript Merger logic. | P0 |
| FR-EX-2 | SRT line splits use word-level timing (depends on FR-WD-3). | P1 |

### 8.8 Caching

| ID | Requirement | Priority |
|---|---|---|
| FR-CA-1 | Hash input audio with SHA-256 of file contents. | P1 |
| FR-CA-2 | Cache WhisperX JSON output keyed by hash plus model version plus alignment model. | P1 |
| FR-CA-3 | Skip re-transcription when a cached entry matches. | P1 |

Caching saves 30-plus minutes of compute every time the merge logic is tweaked and the same session is reprocessed. It is cheap to add upfront and painful to bolt on later, so it sits inside v1 scope even though the first working pipeline does not strictly require it.

### 8.9 Modes

v1 ships two tabs.

| ID | Requirement | Priority |
|---|---|---|
| FR-MD-1 | **Transcribe and Merge** tab: the new primary path, audio in. Fuses the old Merge Speakers and Sequence Transcripts modes into one, because audio files carry continuous time and ffprobe reads durations automatically. | P0 |
| FR-MD-2 | **Edit Transcript** tab: the carryover from the web tool, file in. Handles transcripts from Resolve, Premiere, or Descript that did not come through the stems pipeline. | P0 |
| FR-MD-3 | Retire the standalone Sequence-only mode. | P0 |

What the new pipeline removes from the old workflow:

- Manual segment duration entry (ffprobe reads duration from the file).
- The "(Silence)" marker problem (WhisperX VAD strips silence at transcription, so no placeholder markers are emitted).
- The Strip Silence toggle becomes a no-op on the audio-in path, though it stays useful for Edit mode.

---

## 9. Data model

### 9.1 Input Source model

An **Input Source** has a sequence position, a duration (read by ffprobe), and one or more **tracks**. Each track has a speaker name and a selected-or-skip flag.

- Audio files have exactly one track.
- Video files expose one track per audio stream.

The same structure drives both. The UI renders the track selector only when an input has more than one stream. This model generalizes once more in v2 (Section 11.4), where a single-track recording is represented as one virtual track per speaker, populated from labeled regions. In the v2 UI, each speaker color is one virtual track, so the regions painted with a speaker's color are exactly that speaker's labeled regions, whether the user is looking at the single-track view or the split-lane view.

### 9.2 Merged output JSON schema

```json
{
  "version": "1.0",
  "sources": [{
    "filename": "session_2026_01.mkv",
    "duration": 18253.4,
    "tracks": [
      { "track_index": 0, "speaker": "Max" },
      { "track_index": 1, "speaker": "Desktop" }
    ]
  }],
  "segments": [
    {
      "speaker": "Max",
      "start": 12.34,
      "end": 15.67,
      "text": "And then I realized something...",
      "words": [
        { "word": "And", "start": 12.34, "end": 12.45 },
        { "word": "then", "start": 12.46, "end": 12.62 }
      ],
      "flags": []
    }
  ]
}
```

`flags` holds conflict markers, for example `["overlap"]`, `["bleed_suspected", "preferred"]`, or `["backchannel"]`. The editor UI reads these for highlighting and filtering.

---

## 10. Technical approach and environment

### 10.1 Hardware and runtime targets

| Item | Target |
|---|---|
| GPU | RTX 3080 Ti, 12 GB VRAM |
| Expected speed | 15 to 25x realtime with large-v3 plus alignment. A 5-hour stem finishes in 12 to 20 minutes. |
| Compute type | float16 (Ampere handles it well, fits easily in VRAM, no int8 quantization needed). |
| OS | Windows 11. |
| Python environment | WSL2 plus Ubuntu 24.04 plus conda. Native Windows pip with WhisperX is dependency hell. WSL2 sidesteps it and makes rebuilding the environment trivial when something breaks. |

### 10.2 Stack

- **Backend:** FastAPI. Python is required for WhisperX. Runs a local server on a known port.
- **Frontend:** vanilla HTML and JavaScript, the same approach as the existing index.html. Opened in the browser at localhost on the configured port.
- **Future polish (later, not v1):** a Tauri wrapper with the FastAPI backend as a sidecar, shipping as a real .exe.

Electron, PyQt, Streamlit, and Gradio are all rejected as bloated, ugly, or both.

### 10.3 Audio extraction

ffmpeg extracts and converts each stream to the format WhisperX expects. See Appendix A for the exact invocation. Extraction runs at many times realtime and is mostly disk-bound. Output runs about 30 KB per second, so a 5-hour mono stream is roughly 540 MB, written to a temp directory and deleted once transcription completes.

---

## 11. Release plan

### 11.1 Build order (engine first, UI second)

1. Get WhisperX running in the WSL2 conda environment. Confirm GPU acceleration. One stem in, JSON out.
2. Wire ffprobe plus ffmpeg extraction. Multi-track video in, multiple stems out.
3. Build the merge engine as a pure-Python module. It takes a list of WhisperX outputs plus speaker labels plus sequence order and returns merged JSON. Test with real session data.
4. Add the caching layer (hash plus model version keyed JSON cache).
5. Wrap in FastAPI. Endpoints: upload and select files, get track info, start a transcription job, poll progress, get the merged result.
6. Build the frontend on top. Start by porting the existing index.html, swap in the new Transcribe and Merge mode, and wire it to the FastAPI endpoints.
7. Polish: audition buttons, progress UI, conflict highlighting.

Engine first matters because a UI-first build risks getting halfway through the frontend before discovering the merge logic needs a different data shape. Locking the shape in the engine means the UI is built against a stable contract.

### 11.2 v1 scope

v1 is the multi-track stems pipeline. It ships fast and matches the current real-world recording workflow. v1 includes everything in Section 8 marked P0 and P1.

### 11.3 First 90 minutes (validation before architecture)

1. Install and verify WSL2 plus Ubuntu 24.04.
2. Create the conda environment: PyTorch with CUDA, then `pip install whisperx`.
3. Run `nvidia-smi` inside WSL and confirm the 3080 Ti is visible. This is the gating check before any code.
4. CLI test: push one real 5-hour stem through WhisperX large-v3 plus alignment.
5. Inspect the JSON output and confirm word timestamps actually solve the navigation problem before architecting around them.

If step 2 turns rough, bail to native Windows conda before sinking more time into WSL2.

### 11.4 v2: single-track manual diarization

**Purpose.** Some recordings have no stems: an interview the user did not record, a podcast appearance, recovered audio, or a collaboration where the other person did not separate their tracks. Those files do not fit the v1 workflow at all. Diarization models exist but are brittle, and their failure modes are exactly what motivated the stems pipeline. v2 sidesteps the model entirely by letting the user label segments by hand on a waveform, then running WhisperX only on the labeled regions per speaker. Same end-state data structure, different way to populate it.

**Labeling interface.** The source recording renders as a single waveform with a row of speaker buttons above it, one per named speaker, each carrying its own color. Labeling is one motion: select a span on the waveform, click a speaker button, and the span takes that speaker's color and assignment. Number keys (1, 2, 0 for skip) act as shortcuts for the buttons. VAD pre-detects speech and lays down starting selections, so most of the work is confirming and clicking rather than drawing spans by hand. Each speaker's set of colored regions is exactly the virtual track from the Input Source model (Section 9.1), and WhisperX transcribes each speaker on only that speaker's audio.

**View toggle.** The same labeled data renders two ways, switchable at any time:

- **Single track (default):** every speaker's colored regions on one waveform. Compact, and the whole timeline stays visible at once.
- **Split lanes:** one horizontal lane per speaker, take-lane style, with each speaker's regions in their own lane. Better for dense back-and-forth where colors on a single track get hard to read.

Both views are renderings of the same region-to-speaker assignment, so switching never changes the underlying data.

**Workflow.**

1. Drop a single-track audio file in.
2. Enter speaker names (for example "Max", "Guest"). Each name creates a speaker button with its own color.
3. The app runs VAD, auto-detects speech, and lays the segments down as unassigned selections on the waveform.
4. Assign each selection: click it, then click a speaker button (or press its number key). The selection takes that speaker's color. Bulk-assign a run of selections to one speaker in a single action.
5. Refine where VAD got it wrong: drag a selection's edges to fix boundaries, or draw a free span on the waveform for speech VAD missed.
6. Click Transcribe. WhisperX runs per speaker, on only that speaker's regions, producing per-speaker segments.
7. The same merge engine as the stems path produces the final transcript.

For a 2-hour podcast with clean turn-taking, that is 200 to 400 segments. With keyboard shortcuts and bulk operations ("next 5 are Speaker A"), it is a few minutes of work, far less than fixing diarization mistakes after the fact.

**Value framing.** The original assumption was that skipping silence would massively speed up transcription. WhisperX VAD already strips silence at the transcription stage, so that speed win is smaller than expected. The real wins are:

1. Single-track recordings become a viable input type (the big one).
2. Diarization brittleness disappears (the hard problem in speech AI is sidestepped).
3. Explicit control over what gets transcribed (sponsor breaks, confidential asides, and tangents can be marked as skip).

Frame v2 as expanded use cases plus better accuracy, not primarily as faster.

**Honest limitation.** Genuinely simultaneous speech on a single track cannot be separated by labeling. The interface can show it, by coloring the same span for two speakers, but WhisperX still transcribes the shared audio for each and produces one garbled line per speaker. The user either picks one speaker for the range and loses the other, or accepts the overlap. The stems approach stays superior whenever simultaneous speech matters. v2 is for clean turn-taking on a single track.

**Tech.**

- **Waveform and lanes UI:** WaveSurfer.js. It handles colored regions and selections, playback, and keyboard shortcuts, and renders both the single-track view and the split-lane view from the same region set. Drops into the existing HTML and JavaScript frontend.
- **VAD:** silero-vad or pyannote VAD on the Python side. Results go to the frontend as a list of (start, end) regions, laid down as unassigned selections for the user to color.

**Why v2 and not v1.** v1 ships fast and matches the current workflow. Single-track manual diarization is a meaningful UI build: waveform component, VAD integration, colored region assignment, the single-track and split-lane views with a toggle between them, boundary editing, keyboard shortcuts, playback controls, and bulk operations. It is worth doing once the core pipeline works on real OBS files.

**v2 labeling requirements.**

| ID | Requirement | Priority |
|---|---|---|
| FR-V2-1 | Render the single-track source as one waveform with a row of per-speaker buttons above it, each with its own color. | P2 |
| FR-V2-2 | Pre-populate selections from VAD output as unassigned regions. | P2 |
| FR-V2-3 | Assign a selection to a speaker by clicking the speaker button or pressing its number key. The selection takes that speaker's color. | P2 |
| FR-V2-4 | Provide a view toggle between single-track (all speakers colored on one waveform) and split-lane (one lane per speaker). Both render from the same data. | P2 |
| FR-V2-5 | Support free-span selection on the waveform to create and assign a region VAD missed. | P2 |
| FR-V2-6 | Support boundary refinement by dragging selection edges. | P2 |
| FR-V2-7 | Support bulk assignment of a contiguous run of selections to one speaker. | P2 |
| FR-V2-8 | Run WhisperX per speaker on that speaker's regions, populating one virtual track per speaker (Section 9.1). | P2 |
| FR-V2-9 | Provide per-region playback so the user can audition before and after assigning. | P2 |

---

## 12. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-1 | A 5-hour stem completes transcription plus alignment in 12 to 20 minutes on the target GPU. |
| NFR-2 | The pipeline runs fully offline with no network dependency. |
| NFR-3 | Extracted temp audio is deleted after transcription. |
| NFR-4 | The environment rebuilds cleanly from a documented conda setup when it breaks. |
| NFR-5 | The frontend reuses Transcript Merger code wherever existing logic applies, to minimize new surface area. |

---

## 13. Success metrics

| ID | Metric | Target |
|---|---|---|
| SM-1 | Word-timestamp accuracy | Semantic search across long footage returns the correct moment, validated against real 5-hour stems. |
| SM-2 | Processing time | A 5-hour stem finishes within 20 minutes end to end. |
| SM-3 | Workflow time | Time from file drop to exported transcript is materially lower than the current Resolve-plus-export-plus-merge workflow. |
| SM-4 | Cache effectiveness | Reprocessing an unchanged session after a merge-logic tweak skips transcription and saves 30-plus minutes. |
| SM-5 | Conflict surfacing | Overlap, backchannel, and bleed cases are flagged and visible in the editor rather than corrupting the merge silently. |

---

## 14. Open questions

| ID | Question | Current lean |
|---|---|---|
| OQ-1 | Input source variety: only OBS MKVs, or also collab files, Zoom recordings, separate field recorders? External sources may have track sync drift, sample-rate mismatches, and odd codec combinations, which affects how much robustness goes into the v1 input pipeline. | OBS is the easy, well-formed case. Decide how much external robustness v1 takes on. |
| OQ-2 | Backchannel default behavior: parenthetical, separate line, or skip? | User-configurable, default to be chosen. |
| OQ-3 | Bleed RMS threshold: how loud counts as "louder"? | Needs tuning on real recordings. |
| OQ-4 | Cache invalidation: hash the audio file or the decoded PCM content? File hash is faster but breaks on re-encode. Content hash is stable but slower. | Probably file hash for v1. |
| OQ-5 | Tauri timing: when does the browser-plus-FastAPI build get wrapped into a real .exe? | After the engine is solid and there is a reason to ship an installable to other people. |
| OQ-6 | v2 default view: open in single-track colored view or split-lane view? | Lean single-track as default for the whole-timeline overview, with split-lane on demand for dense sections. |
| OQ-7 | v2 selections: snap to VAD boundaries by default with free-span as an override, or always free? And how is a contested range represented, two overlapping colored regions or a dedicated overlap marker? | Lean VAD-snapped by default with free-span override; represent overlap as a span colored for two speakers that inherits the single-track overlap limitation. |

---

## 15. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Native Windows WhisperX dependency hell. | Use WSL2 plus conda. If WSL2 setup turns rough during the first 90 minutes, bail to native Windows conda before sinking more time in. |
| Word timestamps do not actually solve navigation. | Validate on a real 5-hour stem before architecting around them (Section 11.3). |
| Bleed silently corrupting merges. | Bleed detection with RMS comparison and a review flag (FR-MG-4), tuned on real recordings. |
| Merge data shape changes after the UI is built. | Engine-first build order locks the data contract before the frontend exists. |
| Scope creep from external file formats. | Resolve OQ-1 explicitly and keep v1 focused on well-formed OBS MKVs. |

---

## 16. Future work

- **Tauri packaging:** wrap v1 (FastAPI plus browser) into an installable .exe with the backend as a sidecar, once the engine is solid.
- **v2 single-track manual diarization:** detailed in Section 11.4.
- **Recording-to-Video Skill:** a downstream product that depends on WhisperMerger word-level output. Summarized in Appendix C. It warrants its own PRD and should not start until WhisperMerger v1 reliably produces word-level JSON on real footage.

---

## What this unlocks

The end state: drop a folder of raw OBS recordings, label the tracks once, and get a merged transcript with word-level timestamps and a clean SRT. AI semantic search across 200 hours of footage that returns accurate timestamps. Resolve becomes optional for the transcription step entirely.

The video story follows from that. Not "I added more file formats to my tool," but "I built a faster, more accurate version that does not need DaVinci at all."

---

## Appendix A: ffmpeg and ffprobe usage

ffprobe enumerates audio streams and reads duration per input.

ffmpeg extracts a stream to mono, 16 kHz, PCM s16le:

```
ffmpeg -i input.mkv -map 0:a:1 -ac 1 -ar 16000 -c:a pcm_s16le speaker_a.wav
```

Output is roughly 30 KB per second. A 5-hour mono stream is about 540 MB, written to a temp directory and deleted after transcription completes.

---

## Appendix B: Merged output schema

See Section 9.2 for the full JSON schema and the `flags` semantics.

---

## Appendix C: Downstream product summary, Recording-to-Video Skill

> This is a separate product, documented here for context only. It is out of scope for the WhisperMerger PRD and should get its own PRD before any build work begins. It cannot exist without WhisperMerger word-level output underneath it, which is the strongest argument for shipping WhisperMerger v1 first.

**What it is.** A Claude skill plus a DaVinci Resolve MCP pipeline that turns raw recordings into edited YouTube videos, using WhisperMerger as the upstream dependency.

**Pipeline.**

1. Record a session, run it through WhisperMerger, get merged transcript JSON with word-level timestamps.
2. Drop the transcript into Claude with the skill loaded.
3. The skill interviews the creator: what is the story, what is the central conflict, who is the audience, what is the one takeaway?
4. The skill generates a draft script from the transcript, the answers, and the creator's editing principles.
5. The script carries timestamp anchors back to source files. Every line maps to (filename, start, end).
6. Hand the script to Claude with the Resolve MCP loaded.
7. First auto-edit pass: import source media, cut clips at the script timestamps, color-code by inclusion. Green is in the script, red is cut, yellow is uncertain or an alternative take.
8. The creator opens Resolve and scrubs the color-coded timeline to see at a glance what was kept versus cut.
9. The creator recolors anything Claude got wrong and explains the change.
10. The skill regenerates the script reflecting the new selections. Iterate until the cut feels right.
11. Final pass: assemble in script order and prep for B-roll insertion.

**Why color-coding is the unlock.** Most AI editing tools are black boxes that hand back a finished cut, after which the creator spends hours fixing what the AI misunderstood about the story. Color-coded selection makes the AI's choices visible at a glance. Scrubbing a color-coded timeline is faster than reading a script and imagining the cut. When the creator disagrees, they recolor, and the AI reads the new state and updates its model. The AI proposes, the creator reviews and adjusts, the AI reads the changes. Neither side does all of it, and neither side does nothing.

**Do not rebuild the MCP layer.** Two Resolve MCP servers already exist: samuelgursky/davinci-resolve-mcp (broad API coverage, the likely starting point) and apvlv/davinci-resolve-mcp (simpler, project, timeline, and media-pool focused). The work is the skill, the script-to-timeline mapping logic, and the iteration workflow on top. If a specific capability is missing (for example, setting clip colors programmatically), fork or PR.

**What the skill encodes.** The creator's voice and editing principles, drawn from their own scripts and the Ed Lawrence framework: story over spectacle (lead with the problem, then the solution), hook with conflict before explaining, clean A-roll (cut filler, kill lines that do not advance the point), jump cuts only at natural sentence breaks, three consistent text styles only, make abstract concepts visual through metaphor or on-screen step sequences, simpler edits over fancy ones for trust and retention, and a first-person, casual, direct voice with no marketing tone. The interview pattern of 5 to 10 questions up front is what separates this from "summarize the transcript."

**Open questions (own PRD).** Color scheme granularity, visualization approach (clip colors versus markers versus separate tracks), multi-take handling, B-roll handling, script format (JSON plus a human-readable markdown view), the interview question template, the iteration mechanism (chat versus MCP reading timeline state directly), and the skill name.

**Build order.** Strictly downstream of WhisperMerger v1. Do not start until the merge engine reliably produces word-level JSON on real footage and the output has driven a couple of real videos.
