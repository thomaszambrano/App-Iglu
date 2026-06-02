# Iglu — Monitor de Fatiga Facial

Real-time drowsiness detection that runs entirely in the browser. No backend, no data uploads, no stored frames — all inference happens locally via WebAssembly.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Frontend — React App](#frontend--react-app)
  - [Tech Stack](#tech-stack)
  - [Component Tree](#component-tree)
  - [Detection Pipeline](#detection-pipeline)
  - [Fatigue Scoring Model](#fatigue-scoring-model)
  - [Settings & Thresholds](#settings--thresholds)
- [Python Vision Module](#python-vision-module)
  - [Preprocessing Pipeline](#preprocessing-pipeline)
  - [Ambient Mode Detection](#ambient-mode-detection)
- [Privacy Model](#privacy-model)
- [Local Development](#local-development)
- [Configuration Reference](#configuration-reference)

---

## Overview

Iglu detects facial fatigue signals in real time using the device camera. It tracks eye closure, yawning, head tilt, and movement stillness, combines them into a continuous fatigue score (0–100), and triggers an audio alert when the score crosses a configurable threshold.

The system is designed for **driver and operator safety** use cases where latency and privacy are both critical. There is no server component — the MediaPipe Face Landmarker model runs via WASM directly in the user's browser.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser (Client)                    │
│                                                         │
│  WebRTC Camera ──► MediaPipe WASM ──► Landmark Parser   │
│                          │                    │         │
│                   Face Landmarker        478 3D pts      │
│                       (GPU/CPU)               │         │
│                                        Metric Extractor  │
│                                     (EAR · MAR · Head)   │
│                                               │         │
│                                        Fatigue Scorer    │
│                                     (multi-signal EMA)   │
│                                               │         │
│                                        Alert Engine      │
│                                     (audio · cooldown)   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              Python Vision Module (optional)            │
│                                                         │
│  OpenCV Frame ──► Ambient Detector ──► Preprocessor     │
│                   (hysteresis fsm)    Night / Backlit    │
│                                       CLAHE · Gamma      │
└─────────────────────────────────────────────────────────┘
```

The frontend and Python module are **independent subsystems**. The React app is the production-ready deliverable; the Python module is a preprocessing pipeline designed for embedded or native environments where frame quality needs correction before inference.

---

## Repository Structure

```
App-Iglu/
├── Iglu/                        # React 19 frontend (main product)
│   ├── public/
│   │   └── mediapipe/
│   │       ├── models/          # Bundled face_landmarker.task model
│   │       └── wasm/            # MediaPipe WASM runtime binaries
│   ├── src/
│   │   ├── components/          # UI layer (presentational)
│   │   ├── hooks/               # State & side-effect layer
│   │   ├── services/            # Core detection logic (pure / class-based)
│   │   ├── types/               # TypeScript domain types
│   │   └── utils/               # Landmark math, signal math
│   ├── package.json
│   └── vite.config.ts
│
├── vision/                      # Python preprocessing module
│   ├── ambient.py               # Ambient light FSM
│   ├── pipeline.py              # Orchestration entry point
│   ├── config.py                # Numeric constants
│   ├── processors/              # Night vision, backlight correction
│   ├── detection/               # Landmark utils, alert, metrics
│   ├── utils/                   # Blur check, timing, simulation
│   └── tests/                   # pytest suite
│
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Frontend — React App

### Tech Stack

| Layer | Technology |
|---|---|
| UI Framework | React 19 |
| Language | TypeScript 5.7 |
| Build Tool | Vite 6 |
| Styling | Tailwind CSS v4 |
| ML Runtime | MediaPipe Tasks Vision (WASM) |
| Icons | Lucide React |
| Linting | ESLint 9 + typescript-eslint |
| Formatting | Prettier 3 |

### Component Tree

```
App
├── AlertBanner          — dismissable overlay when alert fires
├── CameraView           — WebRTC video feed + fullscreen toggle
└── (aside)
    ├── DetectionStatus  — live status chip (awake / fatigue / alert / no-face)
    ├── FatiguePanel     — score bar + active signal tags
    ├── SettingsPanel    — user-configurable thresholds
    ├── InfoSection [Engine]   — model name + runtime state
    ├── InfoSection [Metrics]  → MetricsPanel (EAR, MAR, head angles, movement)
    └── InfoSection [Privacy]  — static privacy notice
```

All components are **presentational**. Business logic lives exclusively in hooks and services.

### Detection Pipeline

Each animation frame runs the following pipeline:

```
requestAnimationFrame
  └─ video.currentTime changed?
       └─ FaceDetectionService.detect(video, timestampMs)
            └─ MediaPipe FaceLandmarker.detectForVideo()
                 └─ 478 normalized 3D landmarks + transformation matrix
       └─ analyzeFacialMetrics({ result, previousLandmarks })
            ├─ Eye Aspect Ratio  (left + right, averaged)
            ├─ Mouth Aspect Ratio
            ├─ Head Euler angles (pitch, yaw, roll from 4×4 matrix)
            └─ Movement delta   (Euclidean mean shift vs previous frame)
       └─ analyzeDrowsinessFrame({ metrics, previousState, settings, timestampMs })
            └─ → DrowsinessAnalysis + next runtime state
```

`FaceDetectionService` attempts GPU delegation first; falls back to CPU silently on failure. The model file and WASM binaries are served from `public/mediapipe/` so inference never requires a network call after initial page load.

### Fatigue Scoring Model

The fatigue score is a **continuous 0–100 value** that accumulates when fatigue signals are active and decays at rest. It is computed per-frame using elapsed time (capped at 250 ms to avoid jumps after tab switches).

**Decay:** −10 points/second when no signals are active.

**Signal contributions (points/second):**

| Signal | Trigger condition | Onset rate | Sustained rate |
|---|---|---|---|
| Eyes closed | EAR < threshold | +34 /s | +46 /s after `eyeClosureDurationMs` |
| Yawning | MAR > threshold | +18 /s | +26 /s after 1.4 s |
| Head tilt | pitch or roll > threshold | +18 /s | +22 /s after 1.2 s |
| Stillness | movement < threshold | — | +12 /s after `stillnessDurationMs` |
| Blink bursts | ≥ 5 blinks in 20 s window | +20 /s | — |

**Thresholds:**

- `POSSIBLE_FATIGUE` status: score ≥ 38
- `DROWSINESS_DETECTED` status + alert: score ≥ 70, **or** eyes closed ≥ 1.35 × `eyeClosureDurationMs`

**Smoothing:** All facial metrics are passed through an EMA filter (α = 0.35) before thresholding to suppress per-frame noise from landmark jitter.

**Sensitivity scaling:** A global sensitivity multiplier (`low = 1.16×`, `medium = 1×`, `high = 0.86×`) is applied to all numeric thresholds, making detection more or less aggressive without changing individual parameters.

### Settings & Thresholds

All settings are runtime-configurable via `SettingsPanel`. Defaults:

| Setting | Default | Description |
|---|---|---|
| `eyeClosureThreshold` | `0.19` | EAR below which eyes are considered closed |
| `eyeClosureDurationMs` | `1500` | Duration before eye-closure triggers sustained penalty |
| `yawnThreshold` | `0.48` | MAR above which mouth is considered open for yawning |
| `headTiltThresholdDegrees` | `24` | Pitch/roll degrees for head-tilt signal |
| `stillnessThreshold` | `0.0018` | Mean landmark movement below which stillness is detected |
| `stillnessDurationMs` | `12000` | Duration before stillness contributes to score |
| `alertCooldownMs` | `8000` | Minimum ms between consecutive audio alerts |
| `soundEnabled` | `true` | Enable/disable audio alert |
| `sensitivity` | `medium` | Global threshold scaling (`low` / `medium` / `high`) |

---

## Python Vision Module

The `vision/` module is a standalone Python preprocessing pipeline for environments where raw camera frames need quality correction before being passed to an inference engine (e.g., embedded devices running OpenCV, not a browser).

### Preprocessing Pipeline

`PreprocessingPipeline` is the single entry point. Call `pipeline.process(frame, face_box)` once per frame:

```python
result = pipeline.process(frame, face_box=(x, y, w, h))
# result keys:
#   frame          — preprocessed BGR frame ready for inference
#   mode           — AmbientMode (NORMAL / NIGHT_VISION / BACKLIT)
#   skip_inference — True during the 5-frame transition freeze
#   blurry         — True if Laplacian variance < blur_threshold
#   ambient_ms     — ambient detection elapsed time
#   preprocess_ms  — preprocessing elapsed time
#   frame_idx      — monotonic frame counter
```

**Transition freeze:** When ambient mode switches, inference is suppressed for 5 frames (`transition_freeze_frames`) to prevent false detections during the visual transition.

### Ambient Mode Detection

`AmbientDetector` classifies lighting conditions using a hysteresis-filtered state machine to avoid mode flickering.

**Sampling:** Runs every 15 frames (configurable) plus unconditionally on the first frame.

**Classification logic (on a 0.25× downsampled LAB frame):**

```
global_luma = mean(L channel)

if global_luma < night_vision_trigger_luma (55):
    → NIGHT_VISION

if face_box provided:
    center_luma = mean(L within scaled face_box)
else:
    center_luma = mean(L within center 35% crop)

ratio = global_luma / center_luma

if ratio >= backlit_trigger_ratio (2.4):
    → BACKLIT

→ NORMAL
```

**Hysteresis:** Each mode accumulates votes; a mode is only adopted after reaching `hysteresis_buffer` (8) consecutive votes, preventing single-frame misclassifications from causing mode switches.

**Processors:**

| Mode | Processor | Method |
|---|---|---|
| `NIGHT_VISION` | `NightVisionProcessor` | Gamma correction (γ=1.8) + CLAHE |
| `BACKLIT` | `BacklightingCorrector` | CLAHE on L channel (LAB) |
| `NORMAL` | — | Frame passed through unchanged |

---

## Privacy Model

- **No backend.** The app has no server component. There is no API, no database, no cloud function.
- **No frame transmission.** Video frames are processed in-memory by MediaPipe WASM and discarded immediately after landmark extraction.
- **No persistence.** No facial data, metrics, or scores are stored anywhere — not in `localStorage`, not in cookies, nowhere.
- **WebRTC only.** The camera stream is handled by the browser's native WebRTC API. The app never has access to raw JPEG or encoded bytes.
- **Model served locally.** The `face_landmarker.task` model is bundled in `public/mediapipe/` and served as a static asset. No model download occurs at runtime.

---

## Local Development

### Frontend

```bash
cd Iglu
npm install
npm run dev          # starts Vite dev server at http://localhost:5173
npm run build        # TypeScript compile + Vite production build
npm run lint         # ESLint
npm run format       # Prettier
```

### Python Module

```bash
# Create and activate virtualenv
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest vision/tests/

# Run dev script
python vision/scripts/run_dev.py
```

---

## Configuration Reference

### Python `DEFAULT_CONFIG`

```python
DEFAULT_CONFIG = {
    "night_vision_trigger_luma": 55,    # LAB L* threshold for night mode
    "backlit_trigger_ratio": 2.4,       # bg/center luma ratio for backlit mode
    "hysteresis_buffer": 8,             # votes needed to confirm mode switch
    "blur_threshold": 80.0,             # Laplacian variance below = blurry
    "gamma": 1.8,                       # night vision gamma exponent
    "clahe_clip": 2.0,                  # CLAHE clip limit for both processors
    "transition_freeze_frames": 5,      # frames to skip inference after mode switch
}
```

A `LOCAL_TEST_CONFIG` variant is available with relaxed thresholds for development and calibration runs.
