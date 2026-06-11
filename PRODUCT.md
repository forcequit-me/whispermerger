# Product

## Register

product

## Users

Primary: a solo content creator who records multi-track sessions (OBS multi-track MKV with
separate mic/desktop streams) and needs accurate, speaker-labeled, word-level transcripts for
editing and search. Has an RTX-class GPU, runs a local server in the browser, and is comfortable
with audio/video tooling (DaVinci Resolve, OBS). Often working with multi-hour files where a few
seconds of timestamp drift makes a moment unfindable.

Secondary: anyone with a GPU who already relies on the existing Transcript Merger web tool and
wants better timing without changing their editing habits.

Context of use: long focused sessions at a desktop, dark room, frequently a multi-hour file in
flight. The user is mid-workflow (record → transcribe → edit → cut), not browsing. Every screen
is a step in getting raw audio to a clean transcript with as few manual actions as possible.

## Product Purpose

WhisperMerger turns raw multi-track recordings into accurate, speaker-labeled transcripts with
word-level timestamps, running WhisperX locally on a GPU. It collapses the old "split in Resolve,
export each track, transcribe each, drop into Transcript Merger" sequence into three actions: drop
the file, label the tracks, run.

Four surfaces: **Batch** (queue projects, auto-export), **Transcribe & Merge** (one-off pipeline),
**Label Track** (single-track manual diarization on a waveform), **Edit Transcript** (tweak any
WhisperMerger JSON). Success = time from file-drop to exported transcript is materially lower than
the Resolve workflow, conflicts (overlap / backchannel / bleed) surface in the editor instead of
silently corrupting the merge, and a 5-hour stem finishes end-to-end in minutes.

## Brand Personality

A **precise DAW tool**: technical, exact, audio-engineer feel. Reads like a focused instrument,
not an app trying to be friendly. Voice is direct, first-person-adjacent, no marketing tone, no
filler. Numbers are first-class (timecodes, durations, percentages, counts) and should read like
instrument readouts. The interface earns trust by being legible and honest about state (what stage,
what percent, how long) rather than by decoration. Three-word feel: **precise, local, unfussy.**

## Anti-references

- **Generic AI/SaaS dashboard** — purple/blue gradients, hero-metric cards, identical
  icon+heading card grids, decorative glassmorphism. The default "AI made this" look.
- **Bloated GUI toolkit** — Gradio / Streamlit / Electron / PyQt aesthetics. The PRD explicitly
  rejects these as "bloated, ugly, or both." The whole point is a lean vanilla HTML/JS frontend.
- **Consumer-cute** — rounded mascots, playful emoji, oversized friendly onboarding. Wrong tone
  for a power-user instrument. (Emoji-as-icons already removed in favor of SVG.)
- **Cluttered enterprise** — gray Bootstrap-era admin panels, boxes inside boxes, no hierarchy.

## Design Principles

1. **Instrument, not app.** Every control maps to a real audio/transcript operation. Numbers read
   like readouts (monospace, aligned). Nothing decorative competes with the waveform or the data.
2. **Honest state.** Long GPU jobs must never look stuck: always show stage, percent, and elapsed.
   Errors surface inline and specific, never swallowed.
3. **Width belongs to the waveform.** This is timeline work. Give horizontal space to waveforms,
   lanes, and the transcript table; don't cap the working area like a marketing page.
4. **Borrow from Transcript Merger, don't reinvent.** Reuse its editor/export patterns and visual
   tidiness wherever they already work; new surface area is a cost.
5. **Surface conflicts, don't auto-resolve.** Overlap/backchannel/bleed are flagged for the user
   to decide. The tool proposes; the user judges.

## Accessibility & Inclusion

- **Target WCAG 2.1 AA**: contrast ratios for text and UI on the dark theme, visible focus states
  on every interactive control.
- **Keyboard-first**: the editor and labeling already lean on shortcuts (Enter, Ctrl+Enter, Escape,
  arrows, number keys for speaker assignment); keep all primary actions reachable without a mouse.
- **Reduced motion**: honor `prefers-reduced-motion` for progress, stage-stepper, and lane
  animations.
- **Colorblind-safe**: speaker color is never the only signal. Lanes and assignments always carry
  a text label and region count alongside the color swatch, so identity survives without hue.
