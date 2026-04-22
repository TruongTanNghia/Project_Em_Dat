---
name: eeg-project
description: Project context for the EEG Brain Analysis repo (Node 3000 + Python Flask 5000 + PyTorch CNN+BiGRU+Attention, CHB-MIT seizure detection). Use whenever the user asks questions about this project — its architecture, APIs, ML model, file layout, how to run/debug it — or is editing files under backend/, frontend/, training/, or models/. Loading this skill avoids re-reading README/source to rediscover the same facts.
---

# EEG Brain Analysis — Project Context

Seizure detection + qEEG visualization app. CHB-MIT pediatric EEG dataset.
GitHub: `git@github.com:TruongTanNghia/Project_Em_Dat.git` (was `EEG-Brain-Analysis` — remote updated 2026-04-21).

## Architecture (two services run together)

| Service | File | Port | Stack |
|---|---|---|---|
| Web + chat + proxy | [frontend/server.js](../../../frontend/server.js) | 3000 | Node.js, Express, express-ws, OpenAI GPT-4o |
| ML inference API | [backend/python_api.py](../../../backend/python_api.py) | 5000 | Flask, PyTorch, pyedflib, scipy |
| MCP standalone | [frontend/mcp-server.js](../../../frontend/mcp-server.js) | stdio | 4 tools: analyze_eeg, generate_report, get_analysis_history, compare_analyses |
| UI assets | [frontend/](../../../frontend/) | — | Vanilla HTML/JS + Chart.js + Three.js |

**Convention**: `backend/` is Python-only (Flask ML service). Node lives inside `frontend/` because its primary job is serving the UI + proxying to Python — Node isn't a standalone API server here. Static serving is whitelisted to `/css` and `/js` so `server.js` / `package.json` are not exposed over HTTP.

Node proxies `/api/predict-edf` to Python via `PYTHON_API` env (default `http://localhost:5000`). Image analysis (`/api/analyze`) and chat (`/api/chat`) stay inside Node via OpenAI.

## Run commands

```bash
# Terminal 1 — ML API
python backend/python_api.py

# Terminal 2 — Web
cd frontend && npm start          # = node server.js

# MCP only
cd frontend && npm run mcp        # = node mcp-server.js
```

`.env` lives at **project root** and is loaded by both services:
- `server.js` uses `dotenv.config({ path: <PROJECT_ROOT>/.env })`
- Python reads it via `os.environ` (process env at shell level) OR you add `python-dotenv` if needed

Required keys: `OPENAI_API_KEY`, `PYTHON_API=http://localhost:5000`, `PORT=3000`.

## ML Model — IMPORTANT: README is out of date

README still describes GradientBoosting + 33 qEEG features. **Actual current model is PyTorch deep learning** — see [backend/python_api.py](../../../backend/python_api.py):

- **Architecture**: `EEG_CNN_BiGRU_Attention` = 1D-CNN (3 blocks) → 2-layer BiGRU(64) → Temporal Attention → MLP classifier
- **Input**: 23 channels × 1024 samples (4 s windows @ 256 Hz)
- **Bandpass**: 0.5–40 Hz (Butterworth order 4, SOS filtfilt)
- **Weights**: `models/best_cnn_model.pth` (committed, ~880 KB)
- **Metadata**: `models/model_metadata.json` (contains `best_threshold`, `roc_auc`, `pr_auc`, `architecture` — loaded at startup)
- **Device**: auto CUDA → CPU fallback
- **DataParallel safe**: strips `module.` prefix on load

### 4-tier per-window classification ([backend/python_api.py:171-187](../../../backend/python_api.py#L171-L187))

| Prob (p) | Label | Color |
|---|---|---|
| ≥ 0.70 | Nguy hiểm (severe) | `#ef4444` |
| 0.40 – 0.70 | Bất thường (moderate) | `#f97316` |
| 0.20 – 0.40 | Nghi ngờ (mild) | `#f59e0b` |
| < 0.20 | Bình thường (normal) | `#10b981` |

The file-level verdict uses `best_threshold` from metadata (default 0.4). Windows are non-overlapping (`step = WINDOW_SAMPLES = 1024`) during inference. The legacy 33-feature qEEG pipeline is kept only for **visualization** (band power charts, radar/doughnut) — NOT for classification.

## API endpoints

### Node (3000)
- `POST /api/predict-edf` → multipart `edfFile` (≤500 MB) → proxied to Python
- `POST /api/analyze` → multipart image (≤100 MB, jpg/png/gif/bmp/webp/tiff) → GPT-4o Vision
- `POST /api/chat` → JSON chat, session-scoped history
- `GET /api/history`, `GET /api/model-status`, `GET /api/mcp/status`, `POST /api/mcp/execute`
- `/uploads/*` served as static

### Python (5000)
- `POST /api/predict-edf` → form field `edfFile`
  - Filters out `ECG`/`VNS`/empty label channels; needs ≥3 EEG channels; pads to 23
  - Returns per-window probabilities + file-level verdict + band-power stats for charts
- `GET /api/model-info`, `GET /health`

## File layout

```
backend/                    ← Python ONLY
├── python_api.py           Flask ML API — PyTorch inference (port 5000)
├── train_eeg_kaggle.py     Training script for Kaggle/Colab
└── requirements.txt

frontend/                   ← UI + Node server (Node serves the UI)
├── server.js               Web + proxy + OpenAI + WebSocket (port 3000)
├── mcp-server.js           MCP stdio server
├── package.json            Node deps (run npm install / npm start from here)
├── package-lock.json
├── node_modules/           (gitignored)
├── index.html              Routed explicitly by Express
├── css/                    Served via app.use('/css', static)
└── js/                     Served via app.use('/js', static)
    ├── app.js
    └── vendor/{three.min.js, OrbitControls.js}

models/        Model weights + metadata + charts (loaded by python_api.py)
training/      Notebooks — train_eeg_ver3.ipynb is current
docs/          bao_cao_do_an.{md,pdf} — Vietnamese thesis report
uploads/       Runtime uploads (gitignored, can grow to GB)
dataset/       CHB-MIT raw data (gitignored, ~900 MB)
1/             Extra pretrained models (gitignored — contains 126 MB .pt)
.env           OPENAI_API_KEY etc — at project root, loaded by both services
```

**Path resolution**: Both Node and Python resolve PROJECT_ROOT by walking up **one** level from their file — `path.resolve(__dirname, '..')` / `os.path.join(os.path.dirname(__file__), '..')`. Don't hardcode relative paths like `./models` inside service code.

**Static serving whitelist**: `server.js` sits in `frontend/` next to `index.html`. Serving `__dirname` with `express.static` would also expose `server.js` and `package.json` over HTTP. Instead only `/css` and `/js` are served; `/` routes to `index.html`; everything else falls through to the SPA `*` handler (also `index.html`). **Do NOT add `app.use(express.static(__dirname))`** — it bypasses this isolation.

## Git / gitignore rules (important when adding files)

`.gitignore` excludes: `node_modules/`, `.env`, `uploads/`, `dataset/`, `1/`, `*.pt`, `*.h5`, `*.pth`, `*.edf`, `models/*.{pkl,csv,zip,txt}`, `models/.virtual_documents/`.

**Consequences:**
- Adding a new `.pth`/`.h5`/`.pt` will NOT be tracked. If you need to commit a new small weight file, force with `git add -f <path>` or rename.
- `best_cnn_model.pth` is already tracked from before the rule — it stays tracked (gitignore only affects untracked files).
- Never `git add .` without reviewing — `uploads/` was already gitignored but double-check if adding new top-level dirs.
- GitHub hard limit is 100 MB/file; 50 MB triggers a warning. The 126 MB `1/dental-prosthetic-crown-detection-model-x-rays.pt` is the reason pushes used to fail — keep it ignored.

## Conventions / gotchas

- **Language**: All user-facing strings (labels, errors, chatbot replies) are Vietnamese. Keep that consistent when editing.
- **Node 22 fetch conflict**: Proxy uses `form-data`'s `.submit()` instead of `fetch` — see [frontend/server.js](../../../frontend/server.js). Don't "modernize" this to `fetch`/`FormData` without testing.
- **Chart.js + Three.js**: Frontend builds 4 chart types (waveform, anomaly timeline, band-power bar/radar/doughnut, band-power timeline) plus a Three.js 3D brain viewer with OrbitControls. Anomaly windows highlighted red. The 3D viewer has a full RAF lifecycle (dispose + teardown on re-init, pause on tab hidden / section offscreen) — see `teardown3DScene()` and `animate3D()` in [frontend/js/app.js](../../../frontend/js/app.js).
- **OpenAI model**: `gpt-4o` for vision and chat. API key read from `.env` via `dotenv` with explicit path to root.
- **MCP**: stdio-based, see [frontend/mcp-server.js](../../../frontend/mcp-server.js). Tools are also reachable via `POST /api/mcp/execute` from the web UI.
- **Training**: Do NOT try to train locally — use [backend/train_eeg_kaggle.py](../../../backend/train_eeg_kaggle.py) or [training/train_eeg_ver3.ipynb](../../../training/train_eeg_ver3.ipynb) on Kaggle with the CHB-MIT dataset. The script outputs `.pkl` artifacts for the legacy sklearn pipeline; the current `.pth` comes from the notebook.
- **README drift**: README's ML section still mentions Gradient Boosting + 33 qEEG features + `eeg_seizure_model.pkl`. That path is legacy. If the user asks about the "current model", answer with the CNN+BiGRU+Attention description from `python_api.py`, not the README.

## Quick diagnosis cheats

- `python_api.py` won't start → check `models/best_cnn_model.pth` and `models/model_metadata.json` both exist.
- `/api/predict-edf` returns "File EDF không đủ kênh EEG" → file has <3 non-ECG/VNS channels.
- Web works but prediction hangs → Python API down or `PYTHON_API` env wrong.
- `npm start` EADDRINUSE → something already on 3000; `PORT=xxxx npm start`.
- Charts empty → open devtools; `app.js` expects the exact JSON shape from `/api/predict-edf` (window_results with `probability`, `classification.color`, band powers).
