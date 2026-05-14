# Project — Medical AI Suite

Multi-module medical AI: EEG seizure detection, brain MRI tumor
segmentation, lung CT nodule analysis, blood test insights.

## Architecture

```
frontend-next/        Next.js 16 + React 19 wrapper (Vercel-ready)
  app/                Server + client React components (landing page)
  public/             Static assets — legacy.html + css/ + js/ + models/
  next.config.mjs     Proxy /api/* → BACKEND_URL (set in Vercel env)

frontend/             LEGACY Express server (local dev only)
backend/              Python Flask + PyTorch + TensorFlow (port 5000)
models/               .keras + .hdf5 checkpoints (gitignored — too big)
dataset/              BraTS + LIDC samples (gitignored)
```

Production setup: Vercel hosts `frontend-next/`, Python BE runs locally
on the user's machine, exposed via ngrok HTTPS tunnel. The `BACKEND_URL`
env var in Vercel project settings points at the ngrok URL.

## ⚠️ Frontend / UI work — ALWAYS load the impeccable skill first

This project ships a custom design-system skill at
[.agents/skills/impeccable/SKILL.md](.agents/skills/impeccable/SKILL.md).

**Before ANY UI / styling / design task** (landing page, components,
forms, CSS, layout, animation, typography, color, theme, the legacy
medical app's visuals, etc.), the assistant MUST:

1. Read `.agents/skills/impeccable/SKILL.md` to refresh the design laws.
2. Read the matching register reference:
   - `reference/brand.md` for landing pages, marketing, hero sections
   - `reference/product.md` for the 4 medical app modules (EEG / Brain
     / Lung / Blood) and their internal UI
3. If the user invokes a sub-command (`craft`, `audit`, `polish`,
   `bolder`, `quieter`, `distill`, etc.) ALSO read that command's
   `reference/<command>.md` before doing anything.

Hard rules from the skill that catch most slop (non-negotiable):

- Never `#000` / `#fff` — tint every neutral toward the brand hue.
- Use OKLCH for colors, not HSL / hex.
- **Banned**: gradient text (`background-clip: text` + gradient
  background), glassmorphism as a default, identical card grids,
  side-stripe borders > 1px, hero-metric template, modal as first
  thought, em dashes in copy.
- Hierarchy via scale + weight contrast ≥ 1.25 ratio (no flat scales).
- Body line length 65–75ch.
- Motion: ease out exponential, never bounce/elastic, never animate
  CSS layout properties.

The current landing page at `frontend-next/app/page.tsx` +
`frontend-next/public/index.html` was built BEFORE the skill was
installed and violates 3 rules (gradient title, glass cards, identical
4-card grid). Fix those when next iterating the landing.

## Backend

- Python Flask on port 5000 (`backend/python_api.py`).
- Loaded models: EEG (PyTorch CNN+BiGRU+Attention), Lung DeepLabV3,
  Brain 3D U-Net (Keras), YOLOv8 (Ultralytics).
- 21 endpoints — `/health`, `/api/predict-*`, `/api/brain-models`,
  `/api/brain-model-switch`, `/api/*-status` aliases (Express-compat).
- All endpoints return JSON; CORS is wide open (`flask_cors`).

## Deployment

- **Frontend**: Vercel, project `project-em-dat` at
  https://project-em-dat.vercel.app (rename via dashboard if desired).
- **Backend**: user's local machine + ngrok tunnel
  `https://kathryn-nitrocellulosic-martine.ngrok-free.dev` (ngrok URL
  changes on each restart — must update Vercel env var BACKEND_URL).
- **No vercel.json** at repo root — Vercel CLI 53.x has a bug with
  outputDirectory. Use the dashboard Root Directory = `frontend-next`
  + Framework Preset = Next.js instead.

## Commands

```powershell
# Local dev (3 terminals)
cd backend; python python_api.py                    # :5000
cd frontend-next; npm run dev                       # :3000 (or 3001)
ngrok http 5000                                     # HTTPS tunnel

# Git workflow — only commit when user explicitly asks
git add <files>
git commit -m "..."
git push origin main
```

## Conventions for this codebase

- Don't push `models/` or `dataset/` to git (large binaries / datasets).
- `frontend-next/public/models/` IS pushed (small GLB files for 3D viz).
- `.vercelignore` patterns must use leading `/` to anchor to repo root
  so they don't accidentally match `frontend-next/public/models/`.
- Legacy `frontend/` Express server is local-only — not deployed.
