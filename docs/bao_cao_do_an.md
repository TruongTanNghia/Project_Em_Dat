<![CDATA[# BÁO CÁO ĐỒ ÁN TỐT NGHIỆP

# HỆ THỐNG PHÂN TÍCH TÍN HIỆU ĐIỆN NÃO ĐỒ (EEG) ỨNG DỤNG HỌC SÂU TRONG HỖ TRỢ PHÁT HIỆN ĐỘNG KINH

---

## MỤC LỤC

- [Chương 1: Giới thiệu đề tài](#chương-1-giới-thiệu-đề-tài)
- [Chương 2: Cơ sở lý thuyết](#chương-2-cơ-sở-lý-thuyết)
- [Chương 3: Dữ liệu và tiền xử lý](#chương-3-dữ-liệu-và-tiền-xử-lý)
- [Chương 4: Kiến trúc mô hình học sâu](#chương-4-kiến-trúc-mô-hình-học-sâu)
- [Chương 5: Huấn luyện và tối ưu hóa](#chương-5-huấn-luyện-và-tối-ưu-hóa)
- [Chương 6: Kết quả thực nghiệm](#chương-6-kết-quả-thực-nghiệm)
- [Chương 7: Kiến trúc hệ thống phần mềm](#chương-7-kiến-trúc-hệ-thống-phần-mềm)
- [Chương 8: Giao diện người dùng và trực quan hóa](#chương-8-giao-diện-người-dùng-và-trực-quan-hóa)
- [Chương 9: Kết luận và hướng phát triển](#chương-9-kết-luận-và-hướng-phát-triển)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Chương 1: Giới thiệu đề tài

### 1.1. Đặt vấn đề

Động kinh (Epilepsy) là một rối loạn thần kinh mãn tính đặc trưng bởi các cơn co giật (seizure) tái phát do hoạt động điện bất thường trong não bộ. Theo Tổ chức Y tế Thế giới (WHO), ước tính khoảng **50 triệu người** trên toàn cầu mắc bệnh động kinh, trong đó gần 80% sống tại các quốc gia có thu nhập thấp và trung bình. Tại Việt Nam, tỷ lệ mắc động kinh ước tính khoảng 4,4-7,8 trên 1.000 dân, tương đương khoảng 350.000 – 600.000 bệnh nhân.

Phương pháp chẩn đoán tiêu chuẩn vàng cho động kinh là **điện não đồ (Electroencephalography – EEG)**. Bác sĩ chuyên khoa thần kinh phân tích bản ghi EEG dài hàng giờ đến hàng ngày để tìm kiếm các dạng sóng bất thường (spike, spike-wave, sharp wave). Quy trình này tốn rất nhiều thời gian, phụ thuộc kinh nghiệm chủ quan của bác sĩ, và dễ sai sót khi xử lý khối lượng dữ liệu lớn — đặc biệt trong các đơn vị theo dõi EEG liên tục (continuous EEG monitoring).

Sự phát triển mạnh mẽ của **Trí tuệ nhân tạo (AI)** và **Học sâu (Deep Learning)** trong thập kỷ gần đây mở ra cơ hội xây dựng các hệ thống tự động hỗ trợ bác sĩ phân tích EEG nhanh, chính xác, và nhất quán hơn.

### 1.2. Mục tiêu đề tài

Đề tài hướng đến xây dựng một **hệ thống phần mềm hoàn chỉnh** với các mục tiêu:

1. **Xây dựng mô hình Học sâu** phân loại tín hiệu EEG thành bình thường (Normal) và bất thường (Seizure) với độ chính xác cao, sử dụng kiến trúc lai **1D-CNN + BiGRU + Temporal Attention**.

2. **Thiết kế hệ thống 4 mức phân loại lâm sàng** (Bình thường → Cảnh báo nhẹ → Bất thường → Nghiêm trọng) phù hợp với quy trình chẩn đoán y khoa.

3. **Phát triển ứng dụng web** cho phép bác sĩ upload file EDF, nhận kết quả phân tích tự động, và trực quan hóa tín hiệu EEG bằng biểu đồ 2D (Chart.js) và 3D (Three.js).

4. **Tích hợp AI chatbot** (MCP Protocol) để hỗ trợ giải thích kết quả cho người dùng.

### 1.3. Phạm vi đề tài

| Thành phần | Công nghệ | Mô tả |
|---|---|---|
| Mô hình AI | PyTorch, 1D-CNN+BiGRU+Attention | Phân loại Seizure/Normal trên tín hiệu EEG thô |
| Dataset | CHB-MIT Scalp EEG (PhysioNet) | 24 bệnh nhân nhi, 198 cơn seizure, 983 giờ |
| Backend | Python Flask + Node.js Express | API inference + web server |
| Frontend | HTML/CSS/JS + Chart.js + Three.js | Dashboard phân tích + 3D visualization |
| Chatbot | MCP Server + GPT-4 Vision | Giải thích kết quả cho người dùng |

### 1.4. Cấu trúc dự án

```
DU_AN_Y_TE-VIP/
├── python_api.py              # Flask API — load CNN model, inference EDF
├── server.js                  # Node.js Express — web server (port 3000)
├── mcp-server.js              # MCP protocol — AI chatbot integration
├── models/
│   ├── best_cnn_model.pth     # Trọng số CNN đã train (879 KB)
│   ├── model_metadata.json    # Metadata: threshold, AUC, config
│   └── evaluation_dl_charts.png # Biểu đồ đánh giá
├── training/
│   └── train_eeg_ver3.ipynb   # Notebook huấn luyện trên Kaggle
├── public/
│   ├── index.html             # Giao diện dashboard
│   ├── app.js                 # Logic frontend + charts + 3D
│   ├── styles.css             # Dark theme, glassmorphism
│   ├── three.min.js           # Three.js r128 (local)
│   └── OrbitControls.js       # Camera controls cho 3D
├── package.json               # Node.js dependencies
└── requirements.txt           # Python dependencies
```

---

## Chương 2: Cơ sở lý thuyết

### 2.1. Điện não đồ (EEG)

#### 2.1.1. Khái niệm

Điện não đồ (EEG) là phương pháp ghi lại hoạt động điện của não bộ thông qua các điện cực đặt trên bề mặt da đầu theo hệ thống tiêu chuẩn quốc tế **10-20**. Tín hiệu EEG phản ánh tổng hợp các **điện thế hậu synap** (post-synaptic potentials) từ hàng triệu tế bào neuron vỏ não hoạt động đồng bộ.

Đặc điểm của tín hiệu EEG:
- **Biên độ thấp**: thường ở mức 10-100 μV (microvolt)
- **Tần số**: phần lớn nằm trong khoảng 0.5-100 Hz
- **Nhiều kênh**: hệ thống 10-20 sử dụng 21-23 điện cực đặt theo vị trí chuẩn trên da đầu
- **Liên tục theo thời gian**: ghi nhận liên tục, có thể kéo dài từ vài phút đến nhiều ngày

#### 2.1.2. Các dải tần số EEG

Tín hiệu EEG được phân thành 5 dải tần số chính, mỗi dải liên quan đến trạng thái sinh lý khác nhau:

| Dải tần | Tần số (Hz) | Biên độ | Trạng thái não bộ |
|---------|-------------|---------|-------------------|
| **Delta (δ)** | 0.5 – 4 | 20-200 μV | Giấc ngủ sâu (NREM giai đoạn 3-4). Ở người trưởng thành tỉnh táo, delta chiếm ưu thế là dấu hiệu bệnh lý. |
| **Theta (θ)** | 4 – 8 | 5-100 μV | Buồn ngủ, thiền định, trẻ em bình thường. Theta tăng cao bất thường có thể chỉ ra rối loạn nhận thức. |
| **Alpha (α)** | 8 – 13 | 20-60 μV | Thư giãn, nhắm mắt, nghỉ ngơi tỉnh táo. Biến mất khi mở mắt hoặc tập trung (alpha blocking). |
| **Beta (β)** | 13 – 30 | 2-20 μV | Tập trung, hoạt động nhận thức, lo âu. Beta nhanh đôi khi liên quan đến dùng thuốc (benzodiazepine). |
| **Gamma (γ)** | 30 – 50+ | 1-10 μV | Xử lý nhận thức cao cấp, liên kết thông tin giữa các vùng não, trí nhớ làm việc. |

#### 2.1.3. Biểu hiện EEG trong cơn động kinh

Trong cơn co giật (ictal), EEG thể hiện các đặc trưng bất thường rõ ràng:

- **Sóng nhọn (Spike)**: xung điện có thời gian 20-70ms, biên độ cao đột ngột
- **Sóng nhọn-chậm (Spike-and-wave)**: phức hợp spike theo sau bởi sóng chậm, tần số 3Hz đặc trưng cho vắng ý thức (absence seizure)
- **Sóng đa nhọn (Polyspike)**: nhiều spike liên tiếp, thường gặp trong cơn co giật toàn thể
- **Thay đổi đột ngột tần số và biên độ**: chuyển từ nền bình thường sang hoạt động nhịp nhanh hoặc chậm bất thường
- **Sự lan tỏa (Spreading)**: hoạt động bất thường có thể bắt đầu khu trú (focal) rồi lan ra toàn bộ vỏ não (generalized)

### 2.2. Mạng nơ-ron tích chập một chiều (1D-CNN)

#### 2.2.1. Nguyên lý hoạt động

Mạng nơ-ron tích chập (Convolutional Neural Network – CNN) là kiến trúc mạng nơ-ron sâu sử dụng phép tích chập (convolution) để tự động trích xuất đặc trưng từ dữ liệu đầu vào. Trong bài toán xử lý tín hiệu 1D (time series), chúng tôi sử dụng **1D-CNN** – biến thể của CNN truyền thống, trong đó kernel (bộ lọc) quét dọc theo **một chiều** (trục thời gian).

**Công thức phép tích chập 1D:**

```
y[n] = Σ(k=0 → K-1) w[k] · x[n + k] + b
```

Trong đó:
- `x[n]`: tín hiệu đầu vào tại vị trí thời gian n
- `w[k]`: các trọng số của kernel (bộ lọc) có kích thước K
- `b`: hệ số bias
- `y[n]`: giá trị feature map đầu ra tại vị trí n

#### 2.2.2. Vai trò trong bài toán EEG

1D-CNN hoạt động như **bộ trích xuất đặc trưng cục bộ (local feature extractor)** — tự động học cách nhận diện các mẫu sóng ngắn hạn mà mắt thường khó phát hiện, bao gồm: spike, spike-wave, sharp wave, và các biến đổi hình thái bất thường.

Trong mô hình của chúng tôi, 3 block CNN xếp chồng với kích thước kernel giảm dần:

| Block | Kernel size | Chức năng |
|-------|-------------|-----------|
| Block 1 | 15 | Bắt các đặc trưng tần số thấp: sóng delta, theta, dao động chậm |
| Block 2 | 7 | Bắt các đặc trưng tần số trung bình: alpha rhythm, spindles |
| Block 3 | 3 | Bắt các chi tiết tần số cao: spikes, sharp waves, gamma |

Mỗi block kết hợp: `Conv1d → BatchNorm → ReLU → MaxPool → Dropout`

- **BatchNorm (Batch Normalization)**: chuẩn hóa đầu ra mỗi layer, giúp ổn định gradient và tăng tốc hội tụ.
- **ReLU (Rectified Linear Unit)**: hàm kích hoạt `f(x) = max(0, x)`, tạo tính phi tuyến cho mạng.
- **MaxPool**: giảm chiều dữ liệu, giữ lại đặc trưng nổi bật nhất.
- **Dropout (Spatial Dropout)**: tắt ngẫu nhiên một tỷ lệ neuron trong quá trình train, ngăn hiện tượng **quá khớp (overfitting)** — khi model học thuộc training data thay vì học đặc trưng tổng quát.

### 2.3. Mạng GRU hai chiều (Bidirectional GRU)

#### 2.3.1. Mạng hồi quy (RNN) và vấn đề gradient

Mạng neuron hồi quy (Recurrent Neural Network – RNN) là kiến trúc chuyên xử lý dữ liệu tuần tự (sequential data) nhờ cơ chế **bộ nhớ nội bộ** — trạng thái ẩn (hidden state) được truyền từ bước thời gian trước sang bước tiếp theo.

Tuy nhiên, RNN truyền thống gặp vấn đề **vanishing gradient** (gradient biến mất) khi chuỗi dữ liệu dài — gradient truyền ngược qua nhiều time step bị thu nhỏ dần về 0, khiến model không thể học được phụ thuộc xa (long-range dependencies).

#### 2.3.2. GRU — Giải pháp cho gradient biến mất

**GRU (Gated Recurrent Unit)** do Cho et al. (2014) đề xuất, là biến thể cải tiến của RNN, sử dụng 2 cổng (gate) để kiểm soát luồng thông tin:

**Cổng Reset (Reset Gate) — quyết định bao nhiêu thông tin quá khứ cần "quên":**
```
r_t = σ(W_r · [h_{t−1}, x_t] + b_r)
```

**Cổng Update (Update Gate) — quyết định bao nhiêu thông tin mới cần "ghi nhớ":**
```
z_t = σ(W_z · [h_{t−1}, x_t] + b_z)
```

**Trạng thái ẩn ứng viên:**
```
h̃_t = tanh(W_h · [r_t ⊙ h_{t−1}, x_t] + b_h)
```

**Trạng thái ẩn cuối (cân bằng giữa giữ lại cũ và cập nhật mới):**
```
h_t = (1 − z_t) ⊙ h_{t−1} + z_t ⊙ h̃_t
```

Trong đó:
- `σ`: hàm sigmoid (output ∈ [0, 1])
- `⊙`: phép nhân element-wise (Hadamard product)
- `r_t ≈ 0`: quên gần hết quá khứ → trạng thái ẩn mới giống như reset
- `z_t ≈ 0`: giữ nguyên trạng thái cũ; `z_t ≈ 1`: thay hoàn toàn bằng ứng viên mới

So với LSTM (Long Short-Term Memory), GRU có ít tham số hơn (2 gate thay vì 3), tốc độ train nhanh hơn, nhưng hiệu suất tương đương trên nhiều bài toán.

#### 2.3.3. Bidirectional — Nhìn cả hai hướng thời gian

Chúng tôi sử dụng **BiGRU (Bidirectional GRU)** — chạy 2 GRU song song theo 2 hướng:
- **Forward GRU**: đọc chuỗi từ trái sang phải (quá khứ → hiện tại)
- **Backward GRU**: đọc chuỗi từ phải sang trái (tương lai → hiện tại)

```
→h_t = GRU_forward(x_t, →h_{t−1})
←h_t = GRU_backward(x_t, ←h_{t+1})
h_t = [→h_t ; ←h_t]    (nối 2 vector)
```

**Ý nghĩa trong EEG:** Khi phân tích một time step, model có thể tham khảo cả ngữ cảnh **trước và sau** — rất quan trọng vì một đoạn spike cần được đánh giá trong bối cảnh trước-sau để phân biệt seizure thật với artifact (nhiễu do cử động mắt, cơ bắp, v.v.).

Mô hình sử dụng **2 tầng BiGRU** (stacked) với `dropout=0.3` giữa các tầng để tăng depth biểu diễn và chống overfitting.

### 2.4. Cơ chế chú ý thời gian (Temporal Attention)

#### 2.4.1. Động lực

Trong một cửa sổ EEG 4 giây (1024 mẫu), hoạt động seizure thường chỉ chiếm một **phần nhỏ** — có thể là 0.5-2 giây chứa spike, phần còn lại là nền bình thường. Nếu model xử lý đồng đều tất cả time step, tín hiệu seizure sẽ bị "pha loãng" bởi nền, dẫn đến phân loại sai.

**Attention mechanism** giải quyết bằng cách cho model **tự học cách tập trung** vào các thời điểm quan trọng nhất, gán trọng số cao cho vùng seizure và trọng số thấp cho vùng nền.

#### 2.4.2. Công thức

Gọi `H = [h_1, h_2, ..., h_T]` là output của BiGRU tại T time steps:

**Bước 1: Tính năng lượng chú ý (attention energy):**
```
e_t = tanh(W_1 · h_t + b_1)     (W_1 ∈ ℝ^{d×d/2})
```

**Bước 2: Tính trọng số chú ý bằng softmax (đảm bảo tổng = 1):**
```
α_t = exp(W_2 · e_t) / Σ_{k=1}^{T} exp(W_2 · e_k)    (α_t ∈ [0,1])
```

**Bước 3: Tạo context vector (tổng có trọng số):**
```
c = Σ_{t=1}^{T} α_t · h_t      (c ∈ ℝ^d)
```

Trong đó:
- `α_t`: attention weight tại thời điểm t — cho biết model "chú ý" bao nhiêu vào time step đó
- `c`: context vector — tóm tắt toàn bộ chuỗi thời gian thành 1 vector biểu diễn, tập trung vào vùng quan trọng
- Vector `c` sau đó được đưa vào classifier (FC layers) để phân loại Seizure/Normal

#### 2.4.3. Ưu điểm trong phát hiện seizure

- **Phóng đại tín hiệu bất thường**: attention weights tự động tăng cao tại vùng spike/seizure
- **Giảm nhiễu nền**: vùng bình thường được gán trọng số thấp, ít ảnh hưởng đến quyết định
- **Khả năng giải thích (Interpretability)**: có thể trực quan hóa attention weights để biết *vùng nào* model chú ý — giúp bác sĩ hiểu lý do phân loại

### 2.5. Hàm mất mát Focal Loss với Label Smoothing

#### 2.5.1. Vấn đề mất cân bằng dữ liệu (Class Imbalance)

Trong dataset EEG, lớp **Normal chiếm khoảng 85-90%**, trong khi **Seizure chỉ chiếm 10-15%** tổng số cửa sổ. Nếu dùng hàm mất mát Binary Cross-Entropy (BCE) truyền thống, model sẽ bị chi phối bởi lớp đa số — nghĩa là chỉ cần dự đoán tất cả là "Normal" đã đạt accuracy 85-90%, nhưng hoàn toàn vô dụng trên lâm sàng.

#### 2.5.2. Focal Loss (Lin et al., 2017)

Focal Loss giải quyết vấn đề trên bằng cơ chế **giảm trọng số cho các mẫu dễ đoán**:

```
FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)
```

Trong đó:
- `p_t`: xác suất dự đoán đúng lớp thật
- `α_t = 0.75` cho Seizure (lớp thiểu số), `0.25` cho Normal: bù đắp sự mất cân bằng
- `γ = 2.0` (**focusing parameter**): yếu tố then chốt
  - Khi mẫu **dễ đoán** (p_t → 1): `(1−p_t)^2 → 0` → loss rất nhỏ → model bỏ qua
  - Khi mẫu **khó đoán** (p_t → 0): `(1−p_t)^2 → 1` → loss giữ nguyên → model tập trung học

**Hiệu quả**: model tập trung nguồn lực vào các **hard examples** — những mẫu nằm gần ranh giới phân loại, thay vì lãng phí vào các mẫu hiển nhiên.

#### 2.5.3. Label Smoothing (ε = 0.05)

Thay vì label cứng (0 hoặc 1), sử dụng label mềm:
```
y_smooth = y × (1 − ε) + 0.5 × ε
```
Với ε = 0.05: label 1 → 0.975, label 0 → 0.025

**Lý do**: ngăn model trở nên **quá tự tin (overconfident)** — khi model dự đoán xác suất gần 0 hoặc 1, gradient gần bằng 0, model ngừng học. Label smoothing giữ cho gradient luôn tồn tại, cải thiện khả năng tổng quát hóa.

### 2.6. Bộ lọc thông dải Butterworth (Bandpass Filter)

#### 2.6.1. Sự cần thiết của tiền xử lý

Tín hiệu EEG thô bị nhiễm nhiều loại nhiễu:
- **Nhiễu điện lưới (Power line noise)**: 50 Hz (châu Âu/Á) hoặc 60 Hz (Mỹ)
- **Nhiễu cơ bắp (EMG artifact)**: > 30-40 Hz, do co cơ mặt, nhai, nuốt
- **Nhiễu mắt (EOG artifact)**: < 4 Hz, do chớp mắt, cử động nhãn cầu
- **Drift điện cực (Baseline wander)**: < 0.5 Hz, do tiếp xúc điện cực kém

#### 2.6.2. Bộ lọc Butterworth bậc 4

Chúng tôi áp dụng bộ lọc **Butterworth bậc 4**, dải thông **0.5-40 Hz**:

```
H(s) = 1 / √(1 + (s/ω_c)^{2n})
```

Trong đó:
- n = 4: bậc bộ lọc (rolloff -24 dB/octave)
- ω_c: tần số cắt (0.5 Hz cho high-pass, 40 Hz cho low-pass)

**Lý do chọn Butterworth:** đáp ứng tần số **phẳng nhất trong dải thông** (maximally flat magnitude) — không gây méo biên độ tín hiệu EEG trong khoảng 0.5-40 Hz.

**Kỹ thuật lọc**: sử dụng `sosfiltfilt` (zero-phase forward-backward filtering) — lọc 2 lần theo 2 chiều để **không gây trễ pha (zero-phase distortion)**, rất quan trọng trong phân tích EEG vì trễ pha làm sai lệch thời điểm xuất hiện spike.

**Hiệu quả:**
- **High-pass 0.5 Hz**: loại bỏ drift điện cực, giữ sóng delta
- **Low-pass 40 Hz**: loại bỏ nhiễu điện lưới 50/60 Hz và EMG, giữ toàn bộ dải EEG lâm sàng

---

## Chương 3: Dữ liệu và tiền xử lý

### 3.1. Dataset CHB-MIT Scalp EEG

#### 3.1.1. Tổng quan

**CHB-MIT Scalp EEG Database** là bộ dữ liệu EEG công khai nổi tiếng nhất trong lĩnh vực nghiên cứu phát hiện seizure, được thu thập tại **Children's Hospital Boston (CHB)** và **Massachusetts Institute of Technology (MIT)**, phân phối qua **PhysioNet**.

| Thuộc tính | Chi tiết |
|---|---|
| Đối tượng | 24 bệnh nhân nhi (5 nam, 17 nữ, 2 chưa xác định) |
| Độ tuổi | 1.5 – 22 tuổi |
| Tần số lấy mẫu | 256 Hz |
| Số kênh EEG | 23 kênh (hệ thống 10-20 quốc tế) |
| Tổng thời lượng | ~983 giờ ghi liên tục |
| Số cơn Seizure | 198 cơn (có ghi chú thời điểm bắt đầu/kết thúc chính xác) |
| Định dạng file | EDF (European Data Format) |
| Nguồn | https://physionet.org/content/chbmit/1.0.0/ |

#### 3.1.2. Cấu trúc dữ liệu

Mỗi bệnh nhân có một thư mục riêng (vd: `chb01/`) chứa:
- Nhiều file `.edf` (mỗi file 1-4 giờ ghi EEG liên tục)
- File `chb01-summary.txt`: ghi chú thời điểm bắt đầu/kết thúc của từng cơn seizure

### 3.2. Quy trình tiền xử lý (Data Pipeline)

#### 3.2.1. Tổng quan pipeline

```
File EDF → Đọc (pyedflib) → Lọc kênh EEG → Padding/Truncate (23 kênh)
→ Bandpass Filter (0.5-40Hz) → Cắt cửa sổ (4 giây) → Gán nhãn
→ Asymmetric Overlap + Subsampling → Lưu HDF5 → Patient-Aware Split
```

#### 3.2.2. Đọc file EDF và lọc kênh

1. Sử dụng thư viện `pyedflib` để đọc file EDF
2. Loại bỏ các kênh không phải EEG: ECG (điện tâm đồ), VNS (kích thích thần kinh phế vị), kênh trống
3. Giữ tối đa **23 kênh EEG** — nếu file có ít hơn, padding zero

#### 3.2.3. Cắt cửa sổ (Windowing)

Tín hiệu EEG liên tục được cắt thành các **cửa sổ (windows)** cố định:
- **Kích thước**: 4 giây = 1024 mẫu (ở tần số 256 Hz)
- **Hình dạng mỗi window**: (23 kênh × 1024 mẫu)

#### 3.2.4. Gán nhãn (Labeling)

Dựa trên thông tin annotation từ file summary:
- Window nằm hoàn toàn trong khoảng `[seizure_start, seizure_end]` → **label = 1 (Seizure)**
- Window không chứa bất kỳ phần nào thuộc seizure → **label = 0 (Normal)**
- Window nằm giữa (overlap một phần) → **bỏ qua** (tránh nhãn nhiễu)

### 3.3. Xử lý mất cân bằng lớp (Class Imbalance)

Vấn đề nghiêm trọng: trong dataset CHB-MIT, tỷ lệ Seizure/Normal rất chênh lệch (~1:50). Chúng tôi áp dụng **4 chiến lược kết hợp**:

#### 3.3.1. Asymmetric Overlap (Chồng lấp bất đối xứng)

| Lớp | Overlap | Hiệu quả | Giải thích |
|-----|---------|-----------|------------|
| **Seizure** | 75% | ×4 augmentation | Cửa sổ seizure trượt chồng nhau 75%, tạo 4× dữ liệu từ cùng 1 đoạn seizure gốc. Đây là augmentation **tự nhiên** — không thêm nhiễu hay biến dạng nhân tạo. |
| **Normal** | 0% | ×1 (không chồng) | Cửa sổ normal không chồng nhau, giữ nguyên. |

#### 3.3.2. Normal Subsampling

Chỉ giữ **15%** cửa sổ Normal (chọn ngẫu nhiên, `random_seed=42` đảm bảo reproducible). Giảm tỷ lệ từ 1:50 xuống còn khoảng 1:6-8, khả thi hơn cho training.

#### 3.3.3. Focal Loss (α = 0.75, γ = 2.0)

Xem phần 2.5 — hàm loss tự động tập trung vào mẫu khó.

#### 3.3.4. Online Data Augmentation

Augmentation thực hiện **trực tiếp trên GPU** (bằng torch tensor operations) trong quá trình train:

| Loại augmentation | Xác suất | Chi tiết |
|---|---|---|
| **Gaussian Noise** | 50% | Thêm nhiễu σ=0.01, mô phỏng nhiễu sensor |
| **Amplitude Scaling** | 50% | Nhân biên độ ×[0.8, 1.2], mô phỏng biến thiên giữa các phiên ghi |
| **Channel Masking** | 30% | Zero-out 1-2 kênh ngẫu nhiên, buộc model học đặc trưng từ các kênh còn lại |
| **Time Masking** | 30% | Zero-out 10% thời gian, tăng robustness với dữ liệu bị mất |

### 3.4. Patient-Aware Data Split (Chia dữ liệu theo bệnh nhân)

#### 3.4.1. Tại sao không dùng Random Split?

Random split (chia ngẫu nhiên cửa sổ vào train/test) gây **data leakage** nghiêm trọng:
- Các cửa sổ từ cùng một bệnh nhân có đặc điểm EEG tương tự nhau (do giải phẫu não, điện cực tiếp xúc)
- Model sẽ **học nhận diện bệnh nhân** thay vì học đặc trưng seizure phổ quát
- Kết quả test cao giả tạo, nhưng thất bại khi gặp bệnh nhân mới

#### 3.4.2. Phương pháp Patient-Aware Split

Chia theo **bệnh nhân**, đảm bảo **không có dữ liệu nào** của bệnh nhân test xuất hiện trong train:

| Tập | Bệnh nhân (19 train / 5 test) | Tỷ lệ |
|-----|------|---|
| **Train** | chb01, chb02, chb03, chb04, chb05, chb06, chb09, chb10, chb12, chb13, chb14, chb16, chb17, chb18, chb19, chb21, chb22, chb23, chb24 | ~80% |
| **Test** | chb07, chb08, chb11, chb15, chb20 | ~20% |

Nhóm test được chọn sao cho bao gồm đa dạng loại seizure và tỷ lệ seizure/normal tương đương tập train.

### 3.5. Chuẩn hóa dữ liệu (Normalization)

Mỗi cửa sổ được chuẩn hóa **Z-score theo từng kênh, từng window**:

```
x_normalized = (x − mean) / max(std, 1e-6)
```

- `mean`: trung bình tín hiệu trên kênh đó trong window
- `std`: độ lệch chuẩn (clamp tối thiểu 1e-6 tránh chia cho 0)

**Tại sao per-channel, per-window?**
- EEG có biên độ khác nhau giữa các kênh (do vị trí điện cực)
- Biên độ thay đổi theo thời gian (do trạng thái tỉnh/ngủ)
- Chuẩn hóa cục bộ giúp model tập trung vào **hình dạng sóng** thay vì biên độ tuyệt đối

---

## Chương 4: Kiến trúc mô hình học sâu

### 4.1. Tổng quan kiến trúc: 1D-CNN + BiGRU + Temporal Attention

Kiến trúc kết hợp 3 thành phần, mỗi thành phần đóng vai trò riêng biệt:

```
Input (23, 1024) → [CNN Feature Extractor] → [BiGRU Temporal Encoder] → [Attention Pooling] → [Classifier] → Output (0 hoặc 1)
```

| Thành phần | Input | Output | Vai trò |
|---|---|---|---|
| CNN (3 blocks) | (23, 1024) | (128, 64) | Trích xuất đặc trưng cục bộ từ tín hiệu sóng |
| BiGRU (2 layers) | (64, 128) | (64, 128) | Nắm bắt ngữ cảnh thời gian hai chiều |
| Attention | (64, 128) | (128,) | Tập trung vào time steps quan trọng |
| Classifier | (128,) | (1,) | Phân loại Seizure/Normal |

### 4.2. Chi tiết từng lớp

#### 4.2.1. CNN Feature Extractor

```
BLOCK 1:  Conv1d(23→32, kernel=15, stride=2, padding=7)
          → BatchNorm1d(32) → ReLU → MaxPool1d(2) → Dropout(0.1)
          Input: (B, 23, 1024) → Output: (B, 32, 256)

BLOCK 2:  Conv1d(32→64, kernel=7, stride=2, padding=3)
          → BatchNorm1d(64) → ReLU → MaxPool1d(2) → Dropout(0.2)
          Input: (B, 32, 256) → Output: (B, 64, 64)

BLOCK 3:  Conv1d(64→128, kernel=3, padding=1)
          → BatchNorm1d(128) → ReLU → Dropout(0.2)
          Input: (B, 64, 64) → Output: (B, 128, 64)
```

Tín hiệu 1024 mẫu ban đầu được nén xuống 64 time steps với 128 features mỗi step — giảm 16× chiều thời gian trong khi tăng 5.6× chiều đặc trưng.

#### 4.2.2. BiGRU Temporal Encoder

```
Permute: (B, 128, 64) → (B, 64, 128)     [đổi chiều cho GRU]

GRU(input_size=128, hidden_size=64, num_layers=2,
    bidirectional=True, batch_first=True, dropout=0.3)

Output: (B, 64, 128)    [64 time steps × 128 = 64×2 hidden units]
```

- 2 tầng GRU xếp chồng tăng khả năng biểu diễn
- Dropout 0.3 giữa 2 tầng chống overfitting
- Bidirectional: output_size = 2 × hidden_size = 128

#### 4.2.3. Temporal Attention

```
Attention network:
    Linear(128 → 64) → Tanh → Linear(64 → 1)

Weights:  Softmax(scores) ∈ ℝ^{B × 64}
Context:  weighted_sum(gru_output × weights) ∈ ℝ^{B × 128}
```

#### 4.2.4. Classifier

```
Dropout(0.5) → Linear(128 → 64) → ReLU → Dropout(0.3) → Linear(64 → 1)
```

Output qua hàm Sigmoid cho xác suất ∈ [0, 1]:
- < 0.4 (threshold): dự đoán Normal
- ≥ 0.4: dự đoán Seizure

### 4.3. Tổng kết tham số

| Thành phần | Số tham số |
|---|---|
| CNN Block 1 (Conv+BN) | ~11,100 |
| CNN Block 2 (Conv+BN) | ~14,500 |
| CNN Block 3 (Conv+BN) | ~24,800 |
| BiGRU (2 layers, bidirectional) | ~148,000 |
| Temporal Attention | ~4,200 |
| Classifier (FC layers) | ~13,600 |
| **Tổng cộng** | **~216,200** |

Model rất **nhẹ** (~879 KB trên ổ đĩa), phù hợp triển khai trên edge devices và thiết bị di động.

---

## Chương 5: Huấn luyện và tối ưu hóa

### 5.1. Môi trường huấn luyện

| Thông số | Giá trị |
|---|---|
| Platform | Kaggle Notebooks |
| GPU | 2× NVIDIA Tesla T4 (16 GB VRAM mỗi card) |
| RAM | 13 GB |
| Framework | PyTorch 2.x |
| Mixed Precision | AMP (Automatic Mixed Precision) — FP16 |
| Data Parallel | nn.DataParallel (2 GPUs) |

### 5.2. Cấu hình Hyperparameters (v3 Anti-Overfitting)

| Hyperparameter | Giá trị | Lý do chọn |
|---|---|---|
| **Optimizer** | AdamW | Tách biệt weight decay khỏi gradient update, hiệu quả hơn Adam chuẩn cho regularization (Loshchilov & Hutter, 2019) |
| **Learning Rate** | 3×10⁻⁴ | Giảm từ 1×10⁻³ ở v2 — LR thấp hơn giảm dao động val loss, hội tụ ổn định hơn |
| **Weight Decay** | 1×10⁻² | Tăng 100× từ v2 (1×10⁻⁴) — L2 regularization mạnh, phạt weights lớn, chống overfitting |
| **LR Scheduler** | CosineAnnealingWarmRestarts | LR giảm theo cosine rồi restart theo chu kỳ T₀=10, T_mult=2 — linh hoạt hơn OneCycleLR |
| **Epochs** | 50 (max) | Tăng từ 30 ở v2, vì LR thấp hơn cần nhiều epoch hơn để hội tụ |
| **Early Stopping** | patience=10 | Dừng train nếu Val F1 không cải thiện sau 10 epoch → tiết kiệm thời gian, tránh overfit |
| **Batch Size** | 256 | Tối ưu cho dual T4 GPU (vừa đủ VRAM, throughput cao) |
| **Gradient Clipping** | max_norm=1.0 | Ổn định gradient cho các lớp RNN (BiGRU dễ bị exploding gradient) |
| **Focal Loss α** | 0.75 | Trọng số lớp Seizure (thiểu số), bù đắp imbalance |
| **Focal Loss γ** | 2.0 | Focusing parameter — giảm loss cho mẫu dễ, tập trung mẫu khó |
| **Label Smoothing** | 0.05 | Ngăn model overconfident, cải thiện calibration xác suất |
| **Mixed Precision** | FP16 (AMP) | Giảm 50% memory GPU, tăng tốc 20-30% throughput |

### 5.3. So sánh các phiên bản mô hình

| Thuộc tính | v1 (Baseline) | v2 (Improved) | v3 (Anti-Overfitting) |
|---|---|---|---|
| Kiến trúc | 1D-CNN + GRU | 1D-CNN + BiGRU + Attention | 1D-CNN + BiGRU + Attention |
| GRU layers | 1 | 1 | **2** |
| Dropout (CNN) | 0.25 | 0.25 | **0.1/0.2/0.2 (graduated)** |
| GRU dropout | — | — | **0.3** |
| Learning Rate | 1e-3 | 1e-3 | **3e-4** |
| Weight Decay | 1e-4 | 1e-4 | **1e-2** |
| Label Smoothing | — | — | **0.05** |
| Early Stopping | — | — | **patience=10** |
| Scheduler | OneCycleLR | OneCycleLR | **CosineAnnealingWarmRestarts** |
| Time Masking | — | — | **30% (10% thời gian)** |

### 5.4. Quá trình huấn luyện

- Model bắt đầu train từ epoch 1, val F1 tăng dần
- **Best epoch: 14** — Val F1 đạt đỉnh **0.481**
- Sau epoch 14, val F1 dao động và không cải thiện
- **Early stopping kích hoạt tại epoch 24** (patience=10 tính từ epoch 14)
- Tổng thời gian train: ~15 phút trên 2× T4 GPU

---

## Chương 6: Kết quả thực nghiệm

### 6.1. Metrics đánh giá chính

| Metric | Giá trị | Ý nghĩa |
|---|---|---|
| **ROC-AUC** | **0.839** | Khả năng phân biệt Seizure/Normal tổng thể — 0.5 = random, 1.0 = hoàn hảo |
| **PR-AUC (AP)** | **0.529** | Average Precision trên dữ liệu imbalanced — đánh giá chất lượng trên lớp thiểu số |
| **Best Val F1** | **0.481** | F1-Score — trung bình điều hòa giữa Precision và Recall |
| **Best Threshold** | **0.40** | Ngưỡng quyết định tối ưu (được tìm bằng grid search trên validation set) |

### 6.2. Confusion Matrix (threshold = 0.40)

```
                    Predicted Normal    Predicted Seizure
Actual Normal           22,927 (TN)         2,416 (FP)
Actual Seizure           2,059 (FN)         2,238 (TP)
```

Từ confusion matrix, tính các chỉ số lâm sàng:

| Chỉ số | Công thức | Giá trị | Ý nghĩa lâm sàng |
|---|---|---|---|
| **Sensitivity (Recall)** | TP/(TP+FN) | **52.1%** | Trong 100 đoạn seizure thật, model phát hiện được 52 đoạn |
| **Specificity** | TN/(TN+FP) | **90.5%** | Trong 100 đoạn normal thật, model xác nhận đúng 90 đoạn |
| **Precision** | TP/(TP+FP) | **48.1%** | Trong 100 đoạn model dự đoán seizure, 48 đoạn thực sự là seizure |
| **NPV** | TN/(TN+FN) | **91.8%** | Trong 100 đoạn model dự đoán normal, 92 đoạn thực sự là normal |
| **False Positive Rate** | FP/(FP+TN) | **9.5%** | Tỷ lệ báo động giả |

### 6.3. Phân tích kết quả

#### 6.3.1. Điểm mạnh

1. **ROC-AUC = 0.839** cho thấy mô hình có khả năng phân biệt tốt giữa Seizure và Normal ở mức tổng thể. Đây là kết quả **tốt** trên dataset CHB-MIT với patient-aware split (nhiều nghiên cứu đạt 0.85-0.95 nhưng với random split — gây data leakage).

2. **Specificity cao (90.5%)** — ít báo động giả, quan trọng để hệ thống không làm bác sĩ mất niềm tin do quá nhiều false alarm.

3. **Model cực nhẹ** (879 KB, 216K params) — phù hợp triển khai trên thiết bị edge, mobile, hoặc IoT y tế. So sánh: các model Transformer cho EEG thường > 10MB.

4. **Patient-aware evaluation** đảm bảo kết quả phản ánh **khả năng tổng quát hóa thực tế** — model phải phân loại đúng trên bệnh nhân chưa từng thấy.

#### 6.3.2. Hạn chế và hướng cải thiện

1. **Sensitivity 52.1%** còn thấp — bỏ sót ~48% seizure. Trong y tế, sensitivity cao (>80%) là mong muốn. Hướng cải thiện:
   - Thử kiến trúc **Transformer** (Self-Attention thay GRU)
   - **Multi-scale CNN** (song song nhiều kernel size)
   - **Synthetic data augmentation** (dùng GAN tạo thêm dữ liệu seizure)

2. **PR-AUC = 0.529** — phản ánh đúng thách thức của bài toán imbalanced trên patient-aware split.

### 6.4. Hệ thống phân loại 4 mức lâm sàng

Thay vì phân loại nhị phân (Bình thường/Bất thường), hệ thống áp dụng **4 mức phân loại** dựa trên tỷ lệ cửa sổ bất thường và xác suất trung bình:

#### 6.4.1. Phân loại tổng thể (Overall)

| Mức | Tỷ lệ bất thường | HOẶC avg prob | Severity | Khuyến nghị lâm sàng |
|-----|---|---|---|---|
| **BÌNH THƯỜNG** | ≤ 5% | ≤ 20% | normal | Theo dõi định kỳ |
| **CẢNH BÁO NHẸ** | 5% – 15% | 20% – 35% | mild | Tái khám, theo dõi EEG |
| **BẤT THƯỜNG** | 15% – 30% | 35% – 50% | moderate | Khám chuyên khoa thần kinh sớm |
| **NGHIÊM TRỌNG** | > 30% | > 50% | severe | Cần xử lý y khoa ngay |

**Cơ sở thiết kế ngưỡng:**
- **5%**: baseline noise — model có FPR ~9.5%, nhưng EEG bình thường thực tế thường flag 3-5%. Ngưỡng 5% lọc hầu hết false positive.
- **15%**: gấp 3× baseline → gần như chắc chắn có hoạt động bất thường thật.
- **30%**: gần 1/3 recording có seizure → nghi ngờ status epilepticus hoặc seizure cluster.

#### 6.4.2. Phân loại từng cửa sổ (Per-window)

| Probability | Nhãn | Ý nghĩa |
|---|---|---|
| < 20% | Bình thường | Noise hoặc false positive — bỏ qua |
| 20% – 40% | Nghi ngờ | Có đặc trưng lạ nhưng chưa chắc chắn |
| 40% – 70% | Bất thường | Model tự tin phát hiện hoạt động bất thường |
| ≥ 70% | Nguy hiểm | Xác suất cao — gần như chắc chắn seizure |

---

## Chương 7: Kiến trúc hệ thống phần mềm

### 7.1. Kiến trúc tổng thể (System Architecture)

Hệ thống gồm 3 thành phần chính:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Browser)                          │
│  index.html + app.js + styles.css + Chart.js + Three.js            │
│  ─ Dashboard glassmorphism dark theme                              │
│  ─ Upload .edf → hiển thị kết quả                                 │
│  ─ Biểu đồ 2D (23 kênh EEG, anomaly timeline, band power)        │
│  ─ Trực quan hóa 3D (particle system, glow spheres)               │
└─────────────────────────────────────────────────────────────────────┘
                              │ HTTP
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    NODE.JS SERVER (Port 3000)                       │
│  server.js — Express                                               │
│  ─ Serve static files (public/)                                    │
│  ─ Proxy /api/* requests → Flask                                  │
│  ─ File upload handling                                            │
│  ─ MCP Server integration                                         │
└─────────────────────────────────────────────────────────────────────┘
                              │ HTTP (port 5000)
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PYTHON API (Port 5000)                           │
│  python_api.py — Flask + PyTorch                                   │
│  ─ Load CNN model (best_cnn_model.pth) lên GPU                    │
│  ─ Nhận file .edf → tiền xử lý → batch inference                 │
│  ─ Phân loại 4 mức (per-window + overall)                         │
│  ─ Trả JSON: predictions, waveform, band powers                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2. Luồng xử lý khi upload file EDF

```
1. User upload file .edf trên browser
2. Frontend gửi POST /api/predict-edf (multipart/form-data)
3. Node.js proxy forward request → Flask (port 5000)
4. Flask nhận file, lưu tạm, bắt đầu xử lý:
   a. Đọc EDF bằng pyedflib
   b. Lọc kênh EEG (loại ECG, VNS, blank)
   c. Padding/truncate → 23 kênh cố định
   d. Bandpass filter 0.5-40Hz (Butterworth bậc 4, zero-phase)
   e. Cắt sliding windows (4 giây, non-overlapping)
   f. Z-score normalize per-channel per-window
   g. Batch inference (64 windows/batch) trên GPU
   h. Sigmoid → probability → classify_window (4 mức)
   i. Tính overall severity (4 mức)
   j. Compute band powers (Welch PSD) cho display
   k. Downsample waveform (800 pts/channel) cho chart
5. Flask trả JSON response (~50-100KB)
6. Frontend nhận JSON → render:
   - Prediction card (icon, màu, severity label)
   - Summary (tổng windows, % abnormal)
   - Findings list (top 10 flagged windows, 4-tier labels)
   - Abnormalities section (severity breakdown)
   - Recommendations (severity-adapted)
   - EEG waveform chart (23 kênh, abnormal highlights)
   - Anomaly timeline (4-color bar chart)
   - Band power charts (bar, radar, doughnut)
   - 3D visualization (Three.js)
```

### 7.3. API Endpoints

| Endpoint | Method | Mô tả | Response |
|---|---|---|---|
| `/api/predict-edf` | POST | Upload .edf → CNN inference → kết quả phân tích | JSON: overall, severity, windowResults, waveform, bandPowers |
| `/api/model-info` | GET | Thông model (architecture, AUC, threshold, params) | JSON |
| `/health` | GET | Health check | `{ status: 'ok' }` |
| MCP Protocol | stdio | AI chatbot — giải thích kết quả, tư vấn | Text |

### 7.4. Công nghệ sử dụng

| Tầng | Công nghệ | Phiên bản | Vai trò |
|---|---|---|---|
| AI/ML | PyTorch | 2.x | Deep learning framework |
| AI/ML | SciPy | 1.x | Bandpass filter (Butterworth), Welch PSD |
| Backend | Flask | 3.x | REST API server |
| Backend | Flask-CORS | - | Cross-Origin Resource Sharing |
| Backend | pyedflib | 0.1.x | Đọc file EDF |
| Backend | Node.js + Express | 18+ | Web server, proxy, file upload |
| Frontend | Chart.js | 4.x | Biểu đồ 2D (line, bar, radar, doughnut) |
| Frontend | Three.js | r128 | 3D visualization (local, không CDN) |
| Frontend | chartjs-plugin-annotation | 2.x | Annotation boxes cho abnormal regions |

---

## Chương 8: Giao diện người dùng và trực quan hóa

### 8.1. Thiết kế giao diện (UI/UX Design)

Giao diện được thiết kế theo phong cách **Glassmorphism + Dark Theme**, tối ưu cho môi trường y tế:

- **Dark theme**: nền tối giảm mỏi mắt khi bác sĩ xem EEG trong thời gian dài
- **Glassmorphism**: hiệu ứng kính mờ (frosted glass) tạo chiều sâu thị giác sang trọng
- **Color coding**: hệ thống màu nhất quán — xanh lá (normal), vàng (cảnh báo), cam (bất thường), đỏ (nghiêm trọng)
- **Typography**: font Inter (Google Fonts) — clean, modern, dễ đọc

Giao diện chia thành **4 tab chính**:

| Tab | Nội dung |
|---|---|
| **Upload & Phân Tích** | Upload file .edf, hiển thị kết quả prediction, severity, findings, recommendations |
| **Biểu Đồ** | EEG waveform 23 kênh, anomaly timeline, band power charts |
| **Chatbot Tư Vấn** | AI chatbot giải thích kết quả, trả lời câu hỏi y khoa |
| **MCP Server** | Giao diện quản lý MCP protocol cho tích hợp AI |

### 8.2. Biểu đồ 2D (Chart.js)

#### 8.2.1. EEG Waveform Chart

Biểu đồ chính hiển thị **tất cả 23 kênh EEG** trên cùng một canvas:

- **23 màu riêng biệt** cho mỗi kênh (palette thiết kế chuyên dụng)
- **Vertical offset**: mỗi kênh cách nhau 4 đơn vị z-score, tạo hiệu ứng multi-channel montage
- **Tên kênh** hiển thị bên trái chart (custom plugin)
- **Abnormal regions** highlight bằng annotation box:
  - Cam nhạt cho moderate (prob 40-70%)
  - Đỏ đậm cho severe (prob ≥ 70%)
  - Chỉ hiển thị % cho vùng severe
- **800 điểm/kênh** — downsample tối ưu cho performance

#### 8.2.2. Anomaly Probability Timeline

Bar chart hiển thị xác suất bất thường theo thời gian (mỗi bar = 1 window 4 giây):
- **4 màu** phân bậc: xanh (< 20%), vàng (20-40%), cam (40-70%), đỏ (≥ 70%)
- Đường threshold ngang tại 40%

#### 8.2.3. Band Power Charts

Hiển thị phân bố công suất các dải tần số EEG:
- **Bar chart**: so sánh relative power (%) giữa 5 dải tần
- **Radar chart**: biểu đồ mạng nhện cho cái nhìn tổng quan
- **Doughnut chart**: tỷ lệ phần trăm từng dải tần

### 8.3. Trực quan hóa 3D (Three.js)

Sử dụng Three.js r128 (tải local, không CDN) để render EEG trong không gian 3D:

| Tính năng | Mô tả |
|---|---|
| **Star Field** | 2000 ngôi sao nền tạo cảm giác immersive |
| **Particle System** | 300 hạt phát sáng bay lơ lửng (bật/tắt) |
| **Gradient Color** | Màu sóng EEG thay đổi theo biên độ (low→high = dim→bright) |
| **Glow Spheres** | Vùng bất thường: quả cầu phát sáng + nhấp nháy |
| **Accent Lights** | 3 đèn màu (tím, xanh, cyan) tạo dramatic effect |
| **Camera Fly-in** | Camera bay vào mượt mà khi data load |
| **5 View Presets** | Phối cảnh, Trên, Bên, Trước, Cinematic |
| **Surface Mesh** | Bề mặt kết nối giữa các kênh (toggle) |
| **OrbitControls** | Xoay, zoom, pan bằng chuột |

**Hệ trục tọa độ 3D:**
- X: Thời gian (0 → duration)
- Y: Biên độ tín hiệu (z-score normalized)
- Z: Kênh EEG (spacing 6 units)

---

## Chương 9: Kết luận và hướng phát triển

### 9.1. Kết quả đạt được

Đồ án đã xây dựng thành công một **hệ thống end-to-end** phân tích EEG tự động, bao gồm:

1. **Mô hình Học sâu** (1D-CNN + BiGRU + Temporal Attention) với 216,258 tham số, đạt **ROC-AUC = 0.839** trên dataset CHB-MIT với patient-aware evaluation — đảm bảo kết quả phản ánh khả năng tổng quát hóa thực tế.

2. **Hệ thống phân loại 4 mức lâm sàng** (Bình thường → Cảnh báo nhẹ → Bất thường → Nghiêm trọng), áp dụng cho cả per-window (dựa trên probability) và overall (dựa trên tỷ lệ windows bất thường), phù hợp với quy trình chẩn đoán y khoa.

3. **Ứng dụng web hoàn chỉnh** cho phép:
   - Upload file EDF và nhận kết quả phân tích tự động trong vài giây
   - Trực quan hóa tất cả 23 kênh EEG bằng Chart.js với abnormal highlights
   - Khám phá dữ liệu EEG trong không gian 3D tương tác (Three.js)
   - Phân tích band power (delta/theta/alpha/beta/gamma)
   - AI chatbot hỗ trợ giải thích kết quả

4. **Pipeline chống overfitting toàn diện**: Focal Loss + Label Smoothing + Weight Decay + Spatial Dropout + 2-layer BiGRU dropout + Early Stopping + Online Augmentation (4 loại).

### 9.2. Hạn chế

1. **Sensitivity 52.1%** — cần cải thiện để đảm bảo ít bỏ sót seizure hơn.
2. **Chưa hỗ trợ real-time streaming** — hiện tại chỉ phân tích file EDF đã ghi, chưa xử lý EEG trực tuyến.
3. **Chưa được validate trên dataset Việt Nam** — CHB-MIT là dataset nhi khoa phương Tây, có thể có sự khác biệt với đặc điểm EEG dân số Việt Nam.

### 9.3. Hướng phát triển tương lai

| Hướng phát triển | Chi tiết | Độ ưu tiên |
|---|---|---|
| **Cải thiện sensitivity** | Thử Transformer (EEGNet-Transformer), Multi-scale CNN, Self-supervised pretraining | Cao |
| **Tăng dữ liệu** | Thu thập EEG từ bệnh viện Việt Nam, Synthetic augmentation bằng DCGAN/WGAN | Cao |
| **Real-time streaming** | WebSocket-based EEG streaming từ thiết bị → inference liên tục | Trung bình |
| **Mobile app** | React Native / Flutter app cho bác sĩ | Trung bình |
| **Multi-class** | Phân loại thêm: focal vs generalized seizure, interictal vs ictal | Trung bình |
| **Explainable AI** | Trực quan hóa attention weights, GradCAM cho CNN layers | Trung bình |
| **Clinical trial** | Thử nghiệm lâm sàng tại bệnh viện, so sánh với bác sĩ chuyên khoa | Thấp (dài hạn) |
| **FDA/CE certification** | Chứng nhận y tế cho ứng dụng chẩn đoán hỗ trợ | Thấp (dài hạn) |

### 9.4. Kết luận

Đồ án đã chứng minh **tính khả thi** của việc ứng dụng Học sâu vào hỗ trợ chẩn đoán điện não đồ — một bài toán y tế quan trọng và đầy thách thức. Mặc dù mô hình còn hạn chế về sensitivity, kiến trúc 1D-CNN + BiGRU + Temporal Attention đã cho thấy khả năng trích xuất đặc trưng và phát hiện bất thường EEG ở mức có ý nghĩa lâm sàng (ROC-AUC 0.839), đặc biệt khi đánh giá trên bệnh nhân hoàn toàn mới (patient-aware split) — điều khắt khe hơn nhiều so với random split thường thấy trong các nghiên cứu.

Hệ thống web đã hoàn thiện với giao diện chuyên nghiệp, trực quan hóa đa chiều (2D + 3D), và phân loại 4 mức lâm sàng — sẵn sàng cho demo và phát triển tiếp dựa trên phản hồi từ các chuyên gia y tế.

---

## Tài liệu tham khảo

[1] **Shoeb, A.H.** (2009). "Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment." PhD Thesis, Massachusetts Institute of Technology.

[2] **Lin, T.Y., Goyal, P., Girshick, R., He, K., Dollar, P.** (2017). "Focal Loss for Dense Object Detection." IEEE International Conference on Computer Vision (ICCV).

[3] **Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., Bengio, Y.** (2014). "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." Conference on Empirical Methods in Natural Language Processing (EMNLP).

[4] **Bahdanau, D., Cho, K., Bengio, Y.** (2015). "Neural Machine Translation by Jointly Learning to Align and Translate." International Conference on Learning Representations (ICLR).

[5] **Loshchilov, I. & Hutter, F.** (2019). "Decoupled Weight Decay Regularization." International Conference on Learning Representations (ICLR).

[6] **Ioffe, S. & Szegedy, C.** (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." International Conference on Machine Learning (ICML).

[7] **Micikevicius, P., Narang, S., Alben, J., Diamos, G., et al.** (2018). "Mixed Precision Training." International Conference on Learning Representations (ICLR).

[8] **Goldberger, A., Amaral, L., Glass, L., et al.** (2000). "PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals." Circulation, 101(23), e215-e220.

[9] **Gramfort, A., Luessi, M., Larson, E., et al.** (2013). "MEG and EEG data analysis with MNE-Python." Frontiers in Neuroscience, 7:267.

[10] **World Health Organization (WHO)** (2023). "Epilepsy Fact Sheet." https://www.who.int/news-room/fact-sheets/detail/epilepsy

[11] **Three.js Contributors** (2021). Three.js — JavaScript 3D Library. https://threejs.org/

[12] **Chart.js Contributors** (2023). Chart.js — Simple yet flexible JavaScript charting. https://www.chartjs.org/

---

**Nhóm thực hiện:** Nhóm dự án EEG Brain Analysis
**Công nghệ chính:** Python (PyTorch, Flask) · Node.js (Express) · Three.js · Chart.js
**Dataset:** CHB-MIT Scalp EEG Database (PhysioNet)
**Ngày hoàn thành:** Tháng 4/2026
]]>
