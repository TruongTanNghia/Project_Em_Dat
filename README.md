<p align="center">
  <img src="https://img.shields.io/badge/🧠-EEG%20Brain%20Analysis-8B5CF6?style=for-the-badge&labelColor=1e1b4b" alt="EEG Brain Analysis"/>
</p>

<h1 align="center">🧠 EEG Brain Analysis</h1>
<p align="center">
  <strong>Hệ Thống Phân Tích Điện Não Đồ Bằng AI</strong><br/>
  <em>Seizure Detection & Quantitative EEG Analysis</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Node.js-22+-339933?style=flat-square&logo=node.js&logoColor=white" alt="Node.js"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="sklearn"/>
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat-square&logo=openai&logoColor=white" alt="OpenAI"/>
  <img src="https://img.shields.io/badge/Flask-API-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/Chart.js-Visualization-FF6384?style=flat-square&logo=chart.js&logoColor=white" alt="Chart.js"/>
</p>

---

## 📋 Giới Thiệu

**EEG Brain Analysis** là hệ thống phân tích điện não đồ (EEG) sử dụng Machine Learning và AI, được xây dựng cho mục đích nghiên cứu y khoa. Hệ thống có khả năng:

- 🔬 **Phát hiện cơn động kinh (Seizure Detection)** từ file `.edf` bằng mô hình ML đã huấn luyện
- 🖼️ **Phân tích ảnh EEG** bằng GPT-4 Vision với đánh giá chuyên sâu
- 📊 **Trực quan hóa tín hiệu não** theo thời gian thực với highlight vùng bất thường
- 💬 **Chatbot tư vấn thần kinh học** hỗ trợ giải thích kết quả

## 🏗️ Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (HTML/JS/CSS)                │
│    Upload .EDF / Ảnh → Biểu đồ → Chatbot → MCP        │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
    ┌──────────▼──────────┐  ┌───────▼────────────┐
    │  Node.js Server     │  │  Python Flask API   │
    │  (Port 3000)        │  │  (Port 5000)        │
    │                     │  │                     │
    │  • Proxy requests   │  │  • Load ML Model    │
    │  • OpenAI GPT-4     │  │  • Read .edf files  │
    │  • Chat & History   │  │  • Extract qEEG     │
    │  • MCP Server       │  │  • Predict Normal/  │
    │  • WebSocket        │  │    Abnormal          │
    └─────────────────────┘  └─────────────────────┘
                                      │
                             ┌────────▼────────┐
                             │  ML Model (.pkl) │
                             │  GradientBoosting│
                             │  33 qEEG features│
                             │  CHB-MIT Dataset │
                             └─────────────────┘
```

## ⚡ Tính Năng Chính

### 1. 📁 Phân Tích File EDF (Machine Learning)
- Upload file `.edf` (European Data Format)
- Trích xuất **33 đặc trưng qEEG** (Quantitative EEG):
  - Công suất 5 dải sóng: Delta (0.5-4Hz), Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz), Gamma (30-50Hz)
  - Tỉ lệ giữa các dải sóng (Theta/Alpha, Delta/Beta, ...)
  - Chỉ số thống kê (Mean, Std, Kurtosis, Skewness, RMS, Zero-crossings)
- Phân loại bằng **Gradient Boosting Classifier**: **Bình thường** ✅ hoặc **Bất thường** 🔴
- Hiển thị chi tiết từng window 4 giây

### 2. 🖼️ Phân Tích Ảnh EEG (GPT-4 Vision)
- Upload ảnh chụp điện não đồ
- GPT-4o phân tích chuyên sâu: sóng não, vùng não, bất thường
- Trả về đánh giá chi tiết dạng JSON có cấu trúc

### 3. 📊 Trực Quan Hóa
- **🧠 EEG Waveform**: Biểu đồ tín hiệu đa kênh với vùng đỏ = bất thường
- **⚠️ Anomaly Timeline**: Xác suất bất thường theo thời gian (bar chart)
- **📊 Band Power**: Bar chart, Radar chart, Doughnut chart cho 5 dải sóng
- **🧠 Band Power Timeline**: Sự thay đổi năng lượng sóng theo thời gian

### 4. 💬 Chatbot Tư Vấn
- Chatbot AI chuyên gia thần kinh học
- Tự động liên kết context từ kết quả phân tích
- Hỗ trợ tiếng Việt, câu hỏi gợi ý

### 5. 🔌 MCP Server
- Model Context Protocol server tích hợp
- 3 tools: `analyze_eeg`, `generate_report`, `get_analysis_history`

## 🚀 Cài Đặt & Chạy

### Yêu Cầu
- **Python** 3.10+
- **Node.js** 18+
- **OpenAI API Key** (cho GPT-4 Vision & Chatbot)

### Bước 1: Clone & Cài đặt

```bash
git clone https://github.com/TruongTanNghia/Project_Em_Dat.git
cd Project_Em_Dat

# Cài Node.js dependencies
cd frontend && npm install && cd ..

# Cài Python dependencies
pip install -r backend/requirements.txt
```

### Bước 2: Cấu hình

Tạo file `.env`:
```env
OPENAI_API_KEY=sk-your-api-key-here
PYTHON_API=http://localhost:5000
PORT=3000
```

### Bước 3: Chuẩn bị Model

> ⚠️ **Model files (`.pkl`, `.csv`) không được upload lên GitHub** vì file quá lớn.

Có 2 cách:

**Cách 1: Huấn luyện trên Kaggle** (khuyến nghị)
1. Upload `train_eeg_kaggle.ipynb` lên [Kaggle](https://www.kaggle.com/)
2. Thêm dataset: [CHB-MIT EEG Dataset](https://www.kaggle.com/datasets/abhishekinnvonix/seizure-epilepcy-chb-mit-eeg-dataset-pediatric)
3. Chạy notebook → Tải các file output về thư mục `models/`:
   - `eeg_seizure_model.pkl`
   - `eeg_scaler.pkl`
   - `eeg_feature_names.pkl`
   - `eeg_features.csv`
   - `eeg_threshold.pkl`

**Cách 2: Auto re-train** 
- Đặt file `eeg_features.csv` vào `models/`
- Python API sẽ tự động train model khi khởi động lần đầu

### Bước 4: Chạy

Mở **2 terminal** (chạy từ project root):

```bash
# Terminal 1: Python API (ML Model)
python backend/python_api.py
# ✅ Output: 🧠 EEG Python API starting on port 5000...

# Terminal 2: Node.js Server (Web)
cd frontend && npm start
# ✅ Output: 🧠 EEG Analysis Server running on http://localhost:3000
```

Mở trình duyệt → **http://localhost:3000** 🎉

## 📁 Cấu Trúc Dự Án

```
Project_Em_Dat/
├── 📂 backend/                  # Python ML service (port 5000)
│   ├── python_api.py            # Flask API — PyTorch inference
│   ├── train_eeg_kaggle.py      # Script train (Kaggle/Colab)
│   └── requirements.txt
│
├── 📂 frontend/                 # UI + Node server (port 3000)
│   ├── server.js                # Express + WebSocket + OpenAI proxy
│   ├── mcp-server.js            # MCP Server (stdio)
│   ├── package.json
│   ├── package-lock.json
│   ├── index.html               # Giao diện chính
│   ├── 📂 css/
│   │   └── styles.css           # Dark theme UI
│   └── 📂 js/
│       ├── app.js               # Logic upload/charts/chat + Three.js 3D viewer
│       └── 📂 vendor/
│           ├── three.min.js
│           └── OrbitControls.js
│
├── 📂 models/                   # Model weights + metadata
│   ├── best_cnn_model.pth       # PyTorch CNN+BiGRU+Attention
│   ├── model_metadata.json
│   └── evaluation_dl_charts.png
│
├── 📂 training/                 # Notebooks huấn luyện
│   └── train_eeg_ver3.ipynb
│
├── 📂 docs/                     # Báo cáo đồ án
│   ├── bao_cao_do_an.md
│   └── bao_cao_do_an.pdf
│
├── 📂 uploads/                  # File upload runtime (gitignored)
├── 📂 dataset/                  # CHB-MIT raw data (gitignored)
├── 📄 .env                      # API keys (gitignored, root)
├── 📄 .gitignore
└── 📄 README.md
```

## 🧬 Chi Tiết Kỹ Thuật

### Dataset
- **CHB-MIT Scalp EEG Database** — PhysioNet
- 24 bệnh nhân nhi khoa bị động kinh
- ~900 file `.edf`, mỗi file ~1 tiếng
- Tổng: **883,982 windows** (mỗi window 4 giây)

### Feature Engineering (qEEG)

| # | Nhóm | Features | Mô tả |
|---|-------|----------|--------|
| 1-10 | Band Power | `delta_power_mean/std`, `theta_*`, `alpha_*`, `beta_*`, `gamma_*` | Welch PSD cho 5 dải |
| 11-14 | Ratios | `theta_alpha_ratio`, `delta_alpha_ratio`, `delta_beta_ratio`, `theta_beta_ratio` | Tỉ lệ chéo |
| 15-19 | Relative | `delta_relative`, ..., `gamma_relative` | Công suất tương đối |
| 20-33 | Statistics | `mean_mean/std`, `std_*`, `kurtosis_*`, `skewness_*`, `peak_to_peak_*`, `rms_*`, `zero_crossings_*` | Chỉ số thống kê |

### Machine Learning Pipeline

```
File .edf → Đọc kênh EEG → Cắt window 4s → Trích xuất 33 features
    → StandardScaler → GradientBoostingClassifier → Normal/Abnormal
```

- **Model**: Gradient Boosting (100 trees, depth=5)
- **Class balancing**: `compute_sample_weight('balanced')`
- **Training**: 52,851 samples (2,851 seizure + 50,000 normal)

### Metrics
| Model | Accuracy | AUC | F1 |
|-------|----------|-----|-----|
| Gradient Boosting | 99.6% | 0.926 | 0.244 |
| Random Forest | 62.7% | 0.901 | 0.016 |

> ⚠️ F1 thấp do dữ liệu rất mất cân bằng (seizure chỉ chiếm ~0.3%). AUC là metric tin cậy hơn.

## 🖥️ Screenshots

### Upload & Phân Tích
Hỗ trợ 2 chế độ upload:
- **📁 File .EDF** → Model AI phân loại Bình thường/Bất thường
- **🖼️ Ảnh EEG** → GPT-4 Vision phân tích chuyên sâu

### Biểu Đồ Trực Quan
- Tín hiệu EEG đa kênh với vùng bất thường tô đỏ
- Timeline xác suất bất thường với threshold line
- Phân bố năng lượng 5 dải sóng não

### Chatbot Tư Vấn
- Tự động liên kết context phân tích
- Trả lời bằng tiếng Việt chuyên nghiệp

## 🔧 API Endpoints

### Node.js (Port 3000)
| Method | Endpoint | Mô tả |
|--------|----------|--------|
| POST | `/api/predict-edf` | Upload .edf → predict (proxy to Python) |
| POST | `/api/analyze` | Upload ảnh → GPT-4 Vision |
| POST | `/api/chat` | Chatbot |
| GET | `/api/history` | Lịch sử phân tích |
| GET | `/api/model-status` | Trạng thái ML model |
| GET | `/api/mcp/status` | Trạng thái MCP server |
| POST | `/api/mcp/execute` | Thực thi MCP tool |

### Python Flask (Port 5000)
| Method | Endpoint | Mô tả |
|--------|----------|--------|
| POST | `/api/predict-edf` | Nhận .edf → trả về prediction + waveform |
| GET | `/api/model-info` | Thông tin model |
| GET | `/health` | Health check |

## 🤝 Đóng Góp

1. Fork repo
2. Tạo branch: `git checkout -b feature/ten-tinh-nang`
3. Commit: `git commit -m "Add feature"`
4. Push: `git push origin feature/ten-tinh-nang`
5. Tạo Pull Request

## 📝 Giấy Phép

Dự án này được phát triển cho mục đích nghiên cứu và học tập.

---

<p align="center">
  <strong>Phát triển bởi</strong><br/>
  <a href="https://github.com/TruongTanNghia">Trương Tấn Nghĩa</a><br/>
  <em>Đại học — Nghiên cứu Y tế AI</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-❤️-red?style=for-the-badge" alt="Made with love"/>
</p>
