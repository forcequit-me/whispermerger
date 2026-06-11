---
name: WhisperMerger
description: Local DAW-style instrument that turns multi-track recordings into word-level transcripts.
colors:
  accent-signal-blue: "#5b9dff"
  conflict-amber: "#d9a441"
  conflict-red: "#e0625e"
  status-green: "#57b894"
  surface-base: "#14161a"
  surface-panel: "#1c1f26"
  surface-raised: "#232732"
  hairline: "#2e3340"
  text-primary: "#e6e9ef"
  text-muted: "#9aa3b2"
typography:
  headline:
    fontFamily: "system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "17px"
    fontWeight: 600
    lineHeight: 1.5
    letterSpacing: "0.2px"
  body:
    fontFamily: "system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "13px"
    fontWeight: 600
    letterSpacing: "0.6px"
  mono:
    fontFamily: "ui-monospace, JetBrains Mono, Cascadia Code, Menlo, Consolas, monospace"
    fontSize: "12px"
    fontWeight: 400
rounded:
  sm: "6px"
  md: "8px"
  lg: "10px"
spacing:
  xs: "6px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "22px"
components:
  button-primary:
    backgroundColor: "{colors.accent-signal-blue}"
    textColor: "#0b1220"
    rounded: "{rounded.sm}"
    padding: "6px 9px"
  button-ghost:
    backgroundColor: "{colors.surface-raised}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.sm}"
    padding: "6px 9px"
  panel:
    backgroundColor: "{colors.surface-panel}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.lg}"
    padding: "{spacing.lg}"
  input:
    backgroundColor: "{colors.surface-raised}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.sm}"
    padding: "6px 9px"
---

# Design System: WhisperMerger

## 1. Overview

**Creative North Star: "The Workbench"**

WhisperMerger is a maker's workbench for audio, not a dashboard. Tools are laid out where the hand
expects them, nothing is decorative, and everything is within reach. The surface is a dark, tinted
charcoal so that waveforms, lanes, and conflict colors read like the real material on the bench;
the chrome around them stays quiet and gets out of the way. It is an instrument the user works *at*
for hours, not a page they look *at*.

The system rejects, by name, the four things the product is not: the **generic AI/SaaS dashboard**
(no purple gradients, hero-metric cards, identical icon-card grids, or decorative glass); the
**bloated GUI toolkit** look of Gradio / Streamlit / Electron (the whole point is a lean vanilla
frontend); **consumer-cute** (no mascots, no emoji — icons are crisp SVG); and **cluttered
enterprise** admin panels (no boxes-inside-boxes, no gray Bootstrap soup). Depth comes from tonal
layering of three near-black surfaces, never from drop shadows.

Numbers are first-class. Timecodes, durations, percentages, and region counts are set in monospace
and column-aligned so they read like instrument readouts. Color is reserved almost entirely for
meaning — the signal-blue accent and the three conflict hues — and is never spent on decoration.

**Key Characteristics:**
- Dark tinted-charcoal workbench; three tonal surfaces carry all depth (flat, no shadows).
- Width belongs to the waveform — timeline surfaces run wide, not capped like a marketing page.
- Monospace numerics everywhere a number is a readout.
- One accent (signal blue) plus a fixed conflict palette (amber / red / green); nothing else colored.
- Crisp inline-SVG icons, never emoji.

## 2. Colors

A near-monochrome dark workbench where color is spent only on the accent and on conflict meaning.

### Primary
- **Signal Blue** (#5b9dff): the single interactive accent. Primary buttons, focus, progress fill,
  the active stage dot, waveform progress, and the default speaker color. Carries "this is live /
  this is the action." Used sparingly so it stays loud.

### Secondary
The fixed **conflict palette** — these are semantic, not decorative, and map 1:1 to merge flags:
- **Conflict Amber** (#d9a441): genuine speaker overlap.
- **Conflict Red** (#e0625e): suspected mic bleed, and all error/destructive states.
- **Status Green** (#57b894): edited / preferred / done.

### Neutral
- **Surface Base** (#14161a): the app background, the workbench top. Tinted toward blue, never pure black.
- **Surface Panel** (#1c1f26): panels, header, sticky lane labels — one step up from base.
- **Surface Raised** (#232732): inputs, buttons, source rows, lane-track wells — the topmost tonal step.
- **Hairline** (#2e3340): 1px borders and dividers. The only separator the system uses.
- **Text Primary** (#e6e9ef): body and headings.
- **Text Muted** (#9aa3b2): labels, helper text, secondary metadata.

### Named Rules
**The Meaning-Only Color Rule.** Hue is reserved for the accent and the three conflict states.
Amber always means overlap, red always means bleed/error, green always means done/preferred. Never
introduce a color that doesn't carry one of those meanings. A neutral that needs to stand out gets a
tonal step (base → panel → raised), not a tint.

**The No-Pure-Black Rule.** #000 and #fff are prohibited. Every neutral is tinted toward the blue
hue (the base is `#14161a`, not `#101010`). Pure black reads as a hole, not a surface.

## 3. Typography

**Display/Body Font:** system-ui (with -apple-system, Segoe UI, sans-serif fallback)
**Label/Mono Font:** ui-monospace (with JetBrains Mono, Cascadia Code, Menlo, Consolas)

**Character:** Native and unfussy. The system font keeps the tool feeling local and instant (zero
font load on an offline app), and all visual personality comes from the monospace numerics, not from
a branded display face. The pairing is "OS-native chrome, instrument-grade readouts."

### Hierarchy
- **Headline** (600, 17px, 1.5 line-height, 0.2px tracking): the app title only. Deliberately small
  — hierarchy comes from weight and color, the title does not scream.
- **Body** (400, 14px, 1.5): all editable content, transcript text, controls.
- **Label** (600, 13px, uppercase, 0.6px tracking, muted): panel section headers ("PROGRESS",
  "MERGE OPTIONS"). Also an 11px muted variant for field captions.
- **Mono** (400, 12px): every numeric readout — timecodes (HH:MM:SS.ss), durations, percentages,
  region counts, ruler ticks. Tabular figures, column-aligned.

### Named Rules
**The Readout Rule.** Any value that is a measurement — time, duration, percent, count — is set in
mono with tabular numerics so digits align in a column. Words stay in the system sans; numbers
become readouts. Never set a measurement in the proportional body face.

**The Quiet Title Rule.** No oversized H1. The largest type in the app is 17px. Control hierarchy
with weight, color, and tonal surface, never with a giant hero heading.

## 4. Elevation

Flat by default. The system uses **tonal layering**, not shadows: three stacked near-black surfaces
(base #14161a → panel #1c1f26 → raised #232732), each one step lighter, plus 1px hairline borders
(#2e3340) to separate regions. Depth is read from the tonal step and the border, never from a cast
shadow. There is no ambient or drop-shadow vocabulary, and none should be added.

The only "glow" in the system is a 3px soft ring on the **active** progress stage dot
(`rgba(91,157,255,.2)`) — a state signal, not decoration. It appears only on the in-progress stage
and disappears when the stage completes.

### Named Rules
**The Flat-Bench Rule.** Surfaces are flat. If something needs to stand out, raise it one tonal step
or give it a hairline border — do not add a shadow. If a panel looks like it's floating, the design
is wrong.

## 5. Components

### Buttons
- **Shape:** gently rounded (6px radius), compact 6px/9px padding.
- **Primary:** signal-blue fill (#5b9dff) with near-black text (#0b1220), weight 600. The single
  loud element on any screen — used once per panel (Run, Export, Transcribe). Disabled drops to 50%
  opacity with `not-allowed`.
- **Ghost:** raised-surface fill (#232732), hairline border, primary text. The default for every
  secondary action. Icon-only ghost buttons (play, close, reorder) center a 15px SVG with 6px padding.
- **Hover/Focus:** subtle. The shutdown button is the one colored exception — it shifts to conflict
  red on hover to signal a destructive/irreversible action.

### Chips
- **Style:** tiny (10px), pill (10px radius), tinted background at ~20% of the conflict hue with the
  hue as text color. One chip per flag.
- **State:** overlap → amber, backchannel → blue, bleed_suspected → red, edited/preferred → green.
  Read-only meaning markers, not interactive filters.

### Cards / Containers
- **Panel** is the only container. Corner 10px, panel surface (#1c1f26), 1px hairline border, 16px
  internal padding. Used to group one logical step (Input, Merge options, Progress, Transcript).
- **Shadow Strategy:** none — see Elevation. Panels sit flat on the base surface.
- **Nesting is prohibited.** Source rows and lane wells inside a panel use the raised surface
  (#232732) and a hairline, never another bordered card.

### Inputs / Fields
- **Style:** raised surface (#232732), 1px hairline border, 6px radius, 6px/9px padding. Field
  captions sit *above* the input (11px muted label, 4px gap).
- **Numeric inputs** (timecodes, durations) use the mono stack so digits align.
- **Focus:** accent-colored focus ring required on every control (accessibility — see Do's).

### Navigation
- **Tabs** in the header: ghost pill buttons, muted at rest, raised-surface + hairline + primary
  text when active. Four tabs (Batch, Transcribe & Merge, Label Track, Edit Transcript). No
  underline, no accent bar.

### Signature Component — Split-Lane / Waveform (Label Track)
The distinctive surface. A WaveSurfer waveform (180px) over a stacked **split-lane** view: one row
per speaker, each lane a dark well (#14161a) with a center gridline, regions drawn as mini-waveform
canvases tinted with the speaker color. Above the lanes sits a **time ruler** with snapped m:ss
ticks. Each lane label is sticky and carries a color **swatch + name + region count** so identity
never depends on color alone. This is where width and height are spent; it runs full-bleed wide.

## 6. Do's and Don'ts

### Do:
- **Do** keep color meaningful: signal blue for action/live, amber=overlap, red=bleed/error,
  green=done. If a color doesn't carry one of those meanings, remove it.
- **Do** set every measurement (time, duration, percent, count) in the mono stack with tabular
  numerics, column-aligned.
- **Do** convey depth with tonal steps (base → panel → raised) and 1px hairlines. Flat by default.
- **Do** give width to waveforms, lanes, and the transcript table — these surfaces run wide, not
  capped at a marketing-page max-width.
- **Do** keep one signal-blue primary button per panel; everything else is a ghost button.
- **Do** carry a text label + region count alongside every speaker color swatch (colorblind-safe).
- **Do** put an accent focus ring on every interactive control, and honor `prefers-reduced-motion`.

### Don't:
- **Don't** ship the **generic AI/SaaS dashboard** look: no purple/blue gradients as decoration, no
  hero-metric cards, no identical icon+heading card grids, no decorative glassmorphism.
- **Don't** look like a **bloated GUI toolkit** (Gradio / Streamlit / Electron / PyQt). Stay lean.
- **Don't** go **consumer-cute**: no mascots, no emoji (icons are inline SVG, already enforced), no
  oversized friendly onboarding.
- **Don't** build **cluttered enterprise** panels: no boxes inside boxes, no nested cards, no gray
  Bootstrap soup.
- **Don't** use `#000` or `#fff`; every neutral is tinted toward the blue hue.
- **Don't** add drop shadows to convey elevation — raise a tonal step or add a hairline instead.
- **Don't** use a colored side-stripe (`box-shadow: inset 3px 0 0`, `border-left > 1px`) as a row
  accent. The conflict flag rows currently do this and should migrate to a full hairline-tint or a
  leading chip. Flagged for cleanup.
- **Don't** introduce a web/display font — it breaks the offline-instant feel for zero gain.
