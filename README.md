<h1 align="center">ADA Group · Medical AI Suite</h1>

<p align="center">
  <em>Bốn pipeline AI lâm sàng đã triển khai, trên một phòng đọc duy nhất.</em>
</p>

<p align="center">
  <a href="https://project-em-dat.vercel.app">
    <img src="https://img.shields.io/badge/live-project--em--dat.vercel.app-FF5A1F?style=flat-square&labelColor=0a0a0a" alt="Live demo"/>
  </a>
  <img src="https://img.shields.io/badge/frontend-Next.js%2016-000000?style=flat-square&logo=next.js" alt="Next.js"/>
  <img src="https://img.shields.io/badge/backend-Flask%20%2B%20PyTorch-3776AB?style=flat-square&logo=python" alt="Flask + PyTorch"/>
  <img src="https://img.shields.io/badge/deploy-Vercel%20%2B%20ngrok-1f6feb?style=flat-square&logo=vercel" alt="Vercel + ngrok"/>
</p>

---

## Tổng quan

ADA Group dựng pipeline AI cho 4 chuyên khoa và phơi kết quả qua một
giao diện duy nhất. Mỗi case là pipeline production đã trained trên
dataset chuẩn (BraTS, LIDC, CHB-MIT), không phải mockup.

| Case | Module | Dataset | Method | Output chính |
|:---:|:---|:---|:---|:---|
| **01** | NEURO · Phát hiện động kinh EEG | CHB-MIT (24 BN nhi) | CNN + BiGRU + Attention | `p(seizure)` per window 4s, ROC-AUC 0.84 |
| **02** | ONCOLOGY · Phân đoạn u não MRI | BraTS 2020, 4-channel | 3D U-Net + 4-way TTA | mask NCR / ED / ET, Dice WT 0.83 |
| **03** | PULMONOLOGY · Định vị nốt phổi CT | LIDC-IDRI | DeepLabV3 | mask + ø mm + malignancy 1–5 |
| **04** | HEMATOLOGY · Đọc xét nghiệm máu | reference range chuẩn lab | Rule-based engine | flagged values + risk score |

---

## Kiến trúc

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend  ·  Next.js 16 + React 19  ·  hosted on Vercel    │
│                                                             │
│   App Router · OKLCH design tokens · model-viewer 3D · TSX  │
└──────────────────────────┬──────────────────────────────────┘
                           │  /api/*  (proxied via next.config.mjs)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  ngrok HTTPS tunnel        →   BACKEND_URL env var (Vercel) │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Backend  ·  Flask + PyTorch + TF/Keras  ·  localhost:5000  │
│                                                             │
│   21 endpoints  ·  EEG CNN+BiGRU  ·  3D U-Net  ·  DeepLabV3 │
└─────────────────────────────────────────────────────────────┘
```

**Production:** Vercel host `frontend-next/`. Python backend chạy trên
máy local, expose qua ngrok HTTPS. Khi ngrok URL đổi (mỗi lần restart),
cập nhật env var `BACKEND_URL` trong Vercel dashboard rồi redeploy.

---

## Live demo

→ **[project-em-dat.vercel.app](https://project-em-dat.vercel.app)**

> Lưu ý: 4 pipeline AI cần backend Flask đang chạy trên máy host và
> ngrok đang sống. Nếu URL trả về 502, backend đang offline.

---

## Local development

Mở 3 terminal (PowerShell trên Windows / bash trên Unix):

```powershell
# 1. Backend Flask  →  :5000
cd backend
python python_api.py

# 2. Frontend Next.js  →  :3000
cd frontend-next
npm install   # lần đầu
npm run dev

# 3. ngrok tunnel  (chỉ cần khi muốn Vercel gọi vào BE local)
ngrok http 5000
```

Sau khi ngrok in URL HTTPS:

1. Vercel dashboard → project `project-em-dat` → Settings → Environment
   Variables → cập nhật `BACKEND_URL`.
2. Deployments → Redeploy bản mới nhất, **bỏ tick** *Use existing Build
   Cache* để env var được pick up.

---

## Layout

```
.
├── frontend-next/         Next.js wrapper (deploy trên Vercel)
│   ├── app/               Server + client React components
│   ├── public/models/     GLB anatomical models (brain, lung)
│   ├── scripts/           Utility scripts (logo chroma-key, etc.)
│   └── next.config.mjs    Rewrite /api/* → process.env.BACKEND_URL
│
├── .claude/skills/        Project skills cho Claude Code
├── CLAUDE.md              Project context, conventions, deploy notes
├── README.md              (file này)
│
├── backend/               Flask + PyTorch + TF/Keras   (local only)
├── frontend/              Legacy Express server         (local only)
├── models/                .keras / .pth / .pkl weights  (gitignored)
├── training/              Notebook huấn luyện           (gitignored)
├── docs/                  Báo cáo đồ án                 (gitignored)
└── dataset/               BraTS / LIDC / CHB-MIT samples (gitignored)
```

> GitHub repo chỉ track `frontend-next/` (cái Vercel deploy). Backend,
> model weights, dataset đều giữ local — file vẫn còn trên máy nhưng
> không push lên GitHub để repo nhẹ và không leak dataset bệnh án.

---

## Tech stack

| Layer | Stack |
|:---|:---|
| Frontend framework | Next.js 16 (App Router), React 19, TypeScript |
| Design tokens | OKLCH colors, fluid `clamp()` typography, custom motion easings |
| 3D viz | Google `<model-viewer>` v4.1 (GLB anatomical models) |
| Fonts | Mona Sans (display + body), JetBrains Mono (annotations) |
| Backend | Python 3.10+, Flask, flask-cors |
| ML — EEG | PyTorch · CNN + BiGRU + Attention |
| ML — Brain | Keras 3D U-Net · marching-cubes mesh extraction |
| ML — Lung | PyTorch DeepLabV3 |
| ML — Blood | Rule-based engine, reference range chuẩn lab |
| Deploy | Vercel (frontend) + ngrok HTTPS tunnel (backend) |

---

## API endpoints (Flask, :5000)

| Endpoint | Method | Mô tả |
|:---|:---:|:---|
| `/health` | GET | Health check, trả về status + loaded models |
| `/api/predict-edf` | POST | Upload `.edf` → seizure prediction per 4s window |
| `/api/predict-brain` | POST | Upload MRI 4-channel → mask NCR/ED/ET + mesh GLB |
| `/api/predict-lung` | POST | Upload CT → nodule mask + diameter + malignancy |
| `/api/predict-blood` | POST | JSON CBC/Lipid/Glucose → flagged values + risk score |
| `/api/brain-models` | GET | Danh sách brain model variants có sẵn |
| `/api/brain-model-switch` | POST | Switch active brain model checkpoint |
| `/api/*-status` | GET | Aliases compat với legacy Express frontend |

Tổng ~21 endpoints. Toàn bộ trả JSON, CORS mở rộng cho frontend Vercel.

---

## Datasets & metrics

| Module | Dataset | Train size | Best metric |
|:---|:---|:---:|:---:|
| EEG seizure | CHB-MIT (PhysioNet) | 52,851 windows × 4s | ROC-AUC **0.84** |
| Brain tumor | BraTS 2020 | 369 cases × 4 channels | Dice WT **0.83** |
| Lung nodule | LIDC-IDRI | ~1,018 CT scans | mIoU **0.71** |
| Blood panel | rule-based, không train | n/a | n/a |

EEG class imbalance ~0.3% seizure: AUC là metric tin cậy hơn F1 vì F1
nhạy với precision-recall ở scale dataset mất cân bằng cao.

---

## Design system

UI tuân theo skill nội bộ tại `.agents/skills/impeccable/` — design
rules được enforce trên mọi UI change:

- **Color space:** OKLCH only (không hex / HSL / RGB).
- **Brand:** orange-red hexagon mark + cobalt centre dot. Anti-reference:
  blue+white+mint stock healthcare aesthetic.
- **Surface mix:** warm cream paper `oklch(0.96 0.010 70)` cho page,
  drenched dark cards `oklch(0.13 0.006 25)` cho monitor visuals
  (aesthetic: "monitor on a paper desk").
- **Typography:** scale ≥ 1.25× ratio, body 65–75ch, không gradient text.
- **Motion:** ease-out exponential only, không bounce / elastic.

---

## Status

| Item | State |
|:---|:---|
| Frontend deploy | ✓ `https://project-em-dat.vercel.app` |
| Backend deploy | local + ngrok (URL đổi mỗi restart, set lại `BACKEND_URL`) |
| Model weights | trained, giữ local, không push GitHub |
| Datasets | gitignored, vài GB raw |

---

## License & disclaimer

Mục đích **nghiên cứu** + **học tập**.
Không dùng cho chẩn đoán lâm sàng.

<p align="center">
  © 2026 <strong>ADA Group</strong> · <a href="https://github.com/TruongTanNghia">Trương Tấn Nghĩa</a>
</p>
