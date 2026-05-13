# Medical AI Suite — Frontend (Next.js 16)

Next.js 16 (React 19) wrapper around the existing vanilla-JS Medical AI
app. The legacy `index.html` + CSS + JS files live under `public/` and
are served at `/` via a rewrite rule in `next.config.mjs`.

Requires **Node.js 20+** (Next.js 16 dropped Node 18 support).

## Why Next.js?

The previous setup used a plain static `frontend/` folder served by Express.
Deploying that to Vercel hit a bug in Vercel CLI 53.x with `outputDirectory`
in `vercel.json` (`Cannot read properties of undefined (reading 'fsPath')`).

Switching to Next.js solves it because Vercel natively detects Next.js and
deploys it without needing custom config. Zero code changes to the actual
app — only the deployment wrapper changes.

## Project structure

```
frontend-next/
├── app/
│   ├── layout.tsx     # Minimal HTML wrapper
│   └── page.tsx       # Stub (rewrite serves /legacy.html instead)
├── public/
│   ├── legacy.html    # The real app (copy of frontend/index.html)
│   ├── css/styles.css
│   ├── js/app.js, config.js, vendor/*
│   └── models/*.glb
├── next.config.mjs    # Rewrites / → /legacy.html
├── package.json
└── tsconfig.json
```

## Local dev

```bash
cd frontend-next
npm install
npm run dev          # http://localhost:3000
```

The app talks to the Python AI backend via the URL configured in
`public/js/config.js` (`window.APP_CONFIG.API_BASE`). For local dev, leave
it as `''` and run the Express proxy from `frontend/server.js` on the same
host. For Vercel deploys, set `BACKEND_URL` in config.js to an ngrok HTTPS
tunnel pointing at your local Python BE.

## Deploy on Vercel

1. Push this repo to GitHub.
2. Vercel → **Add New Project** → import the repo.
3. **Root Directory:** `frontend-next`.
4. **Framework Preset:** *Next.js* (auto-detected).
5. Build / Output / Install commands: leave default (Vercel handles it).
6. Click **Deploy**.

That's it. Vercel will run `npm install && next build`, then host the static
output globally on its edge network.

## Backend setup (run locally on your machine)

Vercel hosts the FE; the Python AI backend runs on your own machine and is
exposed via an ngrok tunnel.

```bash
# Terminal 1 — Python backend
cd backend
python python_api.py

# Terminal 2 — Express proxy
cd frontend
node server.js

# Terminal 3 — ngrok tunnel
ngrok http 3000
# Copy the https://xxx.ngrok-free.app URL it prints
```

Then edit `frontend-next/public/js/config.js`:

```js
var BACKEND_URL = 'https://xxx.ngrok-free.app';
```

Commit + push → Vercel auto-redeploys with the new backend URL.
