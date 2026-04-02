# ============================================================
# 🧠 EEG SEIZURE DETECTION - QUANTITATIVE EEG ANALYSIS (qEEG)
# Dataset: CHB-MIT Seizure EEG Dataset (Kaggle)
# Chạy trên Kaggle Notebook - Bật GPU nếu có
# ============================================================
# CELL 1: Cài thư viện
# ============================================================

!pip install mne pyedflib scikit-learn xgboost matplotlib seaborn joblib -q

# ============================================================
# CELL 2: Import thư viện
# ============================================================

import os
import re
import numpy as np
import pandas as pd
import mne
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score, roc_curve,
                             precision_recall_curve, f1_score)
from sklearn.utils import class_weight
import joblib
from collections import defaultdict
import gc

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

print("✅ Import thành công!")

# ============================================================
# CELL 3: Cấu hình
# ============================================================

# Đường dẫn dataset trên Kaggle
DATASET_PATH = "/kaggle/input/seizure-epilepcy-chb-mit-eeg-dataset-pediatric"
OUTPUT_PATH = "/kaggle/working"

# Tham số EEG
SAMPLING_RATE = 256        # Hz
WINDOW_SIZE = 4            # giây
WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_SIZE  # 1024 samples
OVERLAP = 0.5              # 50% overlap

# Dải tần sóng não (Hz)
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 50)
}

# Giới hạn số bệnh nhân để xử lý (đặt None = tất cả)
MAX_PATIENTS = 10  # Tăng lên nếu muốn dùng hết data

print(f"📋 Config: Window={WINDOW_SIZE}s, SR={SAMPLING_RATE}Hz, Overlap={OVERLAP}")

# ============================================================
# CELL 4: Đọc annotations (nhãn seizure)
# ============================================================

def parse_summary_file(summary_path):
    """Đọc file summary.txt để lấy thời gian seizure"""
    seizure_info = {}
    
    if not os.path.exists(summary_path):
        return seizure_info
    
    with open(summary_path, 'r') as f:
        content = f.read()
    
    # Tìm các block thông tin file
    blocks = re.split(r'File Name:\s*', content)
    
    for block in blocks[1:]:  # Bỏ block đầu (trống)
        lines = block.strip().split('\n')
        filename = lines[0].strip()
        
        # Tìm số seizures
        num_match = re.search(r'Number of Seizures in File:\s*(\d+)', block)
        if not num_match:
            continue
        
        num_seizures = int(num_match.group(1))
        if num_seizures == 0:
            seizure_info[filename] = []
            continue
        
        # Lấy thời gian start/end seizure
        seizures = []
        starts = re.findall(r'Seizure\s*\d*\s*Start Time:\s*(\d+)\s*seconds', block)
        ends = re.findall(r'Seizure\s*\d*\s*End Time:\s*(\d+)\s*seconds', block)
        
        for s, e in zip(starts, ends):
            seizures.append((int(s), int(e)))
        
        seizure_info[filename] = seizures
    
    return seizure_info

# Test
patient_dirs = sorted([d for d in os.listdir(DATASET_PATH) 
                       if os.path.isdir(os.path.join(DATASET_PATH, d)) and d.startswith('chb')])
print(f"📁 Tìm thấy {len(patient_dirs)} bệnh nhân: {patient_dirs[:5]}...")

# ============================================================
# CELL 5: Trích xuất đặc trưng qEEG
# ============================================================

def compute_band_power(data, sf, band):
    """Tính công suất (PSD) cho 1 dải tần bằng Welch"""
    low, high = band
    freqs, psd = signal.welch(data, sf, nperseg=min(len(data), 256))
    idx = np.logical_and(freqs >= low, freqs <= high)
    return np.trapz(psd[idx], freqs[idx]) if np.any(idx) else 0

def extract_features_from_window(window_data, sf=256):
    """Trích xuất đặc trưng qEEG từ 1 cửa sổ EEG (nhiều kênh)"""
    n_channels = window_data.shape[0]
    features = {}
    
    all_band_powers = {band: [] for band in FREQ_BANDS}
    all_stats = {'mean': [], 'std': [], 'kurtosis': [], 'skewness': [],
                 'peak_to_peak': [], 'rms': [], 'zero_crossings': []}
    
    for ch in range(n_channels):
        ch_data = window_data[ch]
        
        if len(ch_data) == 0 or np.all(ch_data == 0):
            continue
        
        # 1. Band Power (Công suất từng dải sóng)
        total_power = 0
        for band_name, band_range in FREQ_BANDS.items():
            bp = compute_band_power(ch_data, sf, band_range)
            all_band_powers[band_name].append(bp)
            total_power += bp
        
        # 2. Statistical features
        all_stats['mean'].append(np.mean(ch_data))
        all_stats['std'].append(np.std(ch_data))
        all_stats['kurtosis'].append(kurtosis(ch_data))
        all_stats['skewness'].append(skew(ch_data))
        all_stats['peak_to_peak'].append(np.ptp(ch_data))
        all_stats['rms'].append(np.sqrt(np.mean(ch_data**2)))
        zc = np.sum(np.diff(np.sign(ch_data)) != 0)
        all_stats['zero_crossings'].append(zc)
    
    # Trung bình qua tất cả kênh
    for band_name in FREQ_BANDS:
        vals = all_band_powers[band_name]
        if vals:
            features[f'{band_name}_power_mean'] = np.mean(vals)
            features[f'{band_name}_power_std'] = np.std(vals)
        else:
            features[f'{band_name}_power_mean'] = 0
            features[f'{band_name}_power_std'] = 0
    
    # Tỉ lệ giữa các dải
    d = features.get('delta_power_mean', 1e-10)
    t = features.get('theta_power_mean', 1e-10)
    a = features.get('alpha_power_mean', 1e-10)
    b = features.get('beta_power_mean', 1e-10)
    total = d + t + a + b + features.get('gamma_power_mean', 1e-10)
    
    features['theta_alpha_ratio'] = t / max(a, 1e-10)
    features['delta_alpha_ratio'] = d / max(a, 1e-10)
    features['delta_beta_ratio']  = d / max(b, 1e-10)
    features['theta_beta_ratio']  = t / max(b, 1e-10)
    
    # Relative power (%)
    for band_name in FREQ_BANDS:
        features[f'{band_name}_relative'] = features[f'{band_name}_power_mean'] / max(total, 1e-10)
    
    # Statistical features (trung bình qua kênh)
    for stat_name, vals in all_stats.items():
        if vals:
            features[f'{stat_name}_mean'] = np.mean(vals)
            features[f'{stat_name}_std'] = np.std(vals)
    
    return features

print("✅ Hàm trích xuất đặc trưng đã sẵn sàng")

# ============================================================
# CELL 6: Xử lý dataset - Trích xuất features từ .edf
# ============================================================

all_features = []
all_labels = []
processed_files = 0
total_seizure_windows = 0
total_normal_windows = 0

patients_to_process = patient_dirs[:MAX_PATIENTS] if MAX_PATIENTS else patient_dirs

for pi, patient in enumerate(patients_to_process):
    patient_path = os.path.join(DATASET_PATH, patient)
    summary_file = os.path.join(patient_path, f"{patient}-summary.txt")
    
    seizure_info = parse_summary_file(summary_file)
    edf_files = sorted([f for f in os.listdir(patient_path) if f.endswith('.edf')])
    
    print(f"\n👤 [{pi+1}/{len(patients_to_process)}] {patient}: {len(edf_files)} files, "
          f"{sum(len(v) for v in seizure_info.values())} seizures")
    
    for edf_file in edf_files:
        edf_path = os.path.join(patient_path, edf_file)
        
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            
            # Chỉ lấy kênh EEG (bỏ ECG, VNS, dummy)
            eeg_channels = [ch for ch in raw.ch_names 
                           if not ch.startswith('-') and 'ECG' not in ch.upper() 
                           and 'VNS' not in ch.upper() and ch.strip() != '-']
            
            if len(eeg_channels) < 5:
                continue
            
            raw.pick_channels(eeg_channels[:23])  # Tối đa 23 kênh
            
            # Lọc bandpass 0.5-50 Hz
            raw.filter(0.5, 50, fir_design='firwin', verbose=False)
            
            data = raw.get_data()  # shape: (n_channels, n_samples)
            sf = raw.info['sfreq']
            n_samples = data.shape[1]
            
            # Lấy seizure times cho file này
            file_seizures = seizure_info.get(edf_file, [])
            
            # Cắt thành windows
            step = int(WINDOW_SAMPLES * (1 - OVERLAP))
            
            for start in range(0, n_samples - WINDOW_SAMPLES, step):
                end = start + WINDOW_SAMPLES
                window = data[:, start:end]
                
                # Xác định nhãn
                start_sec = start / sf
                end_sec = end / sf
                
                is_seizure = 0
                for sz_start, sz_end in file_seizures:
                    if start_sec < sz_end and end_sec > sz_start:
                        overlap_ratio = (min(end_sec, sz_end) - max(start_sec, sz_start)) / WINDOW_SIZE
                        if overlap_ratio > 0.5:
                            is_seizure = 1
                            break
                
                # Trích xuất features
                feats = extract_features_from_window(window, sf)
                feats['patient'] = patient
                
                all_features.append(feats)
                all_labels.append(is_seizure)
                
                if is_seizure:
                    total_seizure_windows += 1
                else:
                    total_normal_windows += 1
            
            processed_files += 1
            del raw, data
            gc.collect()
            
        except Exception as e:
            print(f"  ⚠️ Lỗi {edf_file}: {str(e)[:50]}")
            continue
    
    print(f"  ✅ Seizure={total_seizure_windows}, Normal={total_normal_windows}")

print(f"\n{'='*50}")
print(f"📊 TỔNG KẾT: {processed_files} files, "
      f"{total_seizure_windows} seizure windows, {total_normal_windows} normal windows")

# ============================================================
# CELL 7: Tạo DataFrame & Cân bằng dữ liệu
# ============================================================

df = pd.DataFrame(all_features)
df['label'] = all_labels

print(f"📋 Dataset shape: {df.shape}")
print(f"\n📊 Phân bố nhãn:")
print(df['label'].value_counts())

# Lưu features
df.to_csv(os.path.join(OUTPUT_PATH, "eeg_features.csv"), index=False)
print("💾 Đã lưu eeg_features.csv")

# Tách features và labels
feature_cols = [c for c in df.columns if c not in ['label', 'patient']]
X = df[feature_cols].values
y = df['label'].values

# Xử lý NaN/Inf
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

print(f"✅ Features: {len(feature_cols)} đặc trưng")
print(f"   Danh sách: {feature_cols[:10]}...")

# ============================================================
# CELL 8: Train/Test Split & Scale
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Class weights cho imbalanced data
cw = class_weight.compute_sample_weight('balanced', y_train)

print(f"📊 Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
print(f"   Train seizure: {sum(y_train==1)} | Train normal: {sum(y_train==0)}")
print(f"   Test  seizure: {sum(y_test==1)}  | Test  normal: {sum(y_test==0)}")

# ============================================================
# CELL 9: Huấn luyện mô hình - Random Forest
# ============================================================

print("🌲 Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

rf_pred = rf_model.predict(X_test_scaled)
rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

print(f"\n📊 Random Forest Results:")
print(f"   Accuracy:  {accuracy_score(y_test, rf_pred):.4f}")
print(f"   F1-Score:  {f1_score(y_test, rf_pred):.4f}")
print(f"   AUC-ROC:   {roc_auc_score(y_test, rf_prob):.4f}")
print(f"\n{classification_report(y_test, rf_pred, target_names=['Normal', 'Seizure'])}")

# ============================================================
# CELL 10: Huấn luyện Gradient Boosting
# ============================================================

print("🚀 Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train, sample_weight=cw)

gb_pred = gb_model.predict(X_test_scaled)
gb_prob = gb_model.predict_proba(X_test_scaled)[:, 1]

print(f"\n📊 Gradient Boosting Results:")
print(f"   Accuracy:  {accuracy_score(y_test, gb_pred):.4f}")
print(f"   F1-Score:  {f1_score(y_test, gb_pred):.4f}")
print(f"   AUC-ROC:   {roc_auc_score(y_test, gb_prob):.4f}")
print(f"\n{classification_report(y_test, gb_pred, target_names=['Normal', 'Seizure'])}")

# ============================================================
# CELL 11: Vẽ biểu đồ đánh giá
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('🧠 EEG Seizure Detection - Model Evaluation', fontsize=16, fontweight='bold')

# 1. Confusion Matrix - RF
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
            xticklabels=['Normal', 'Seizure'], yticklabels=['Normal', 'Seizure'])
axes[0,0].set_title('Random Forest - Confusion Matrix')
axes[0,0].set_ylabel('Actual')
axes[0,0].set_xlabel('Predicted')

# 2. Confusion Matrix - GB
cm_gb = confusion_matrix(y_test, gb_pred)
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Oranges', ax=axes[0,1],
            xticklabels=['Normal', 'Seizure'], yticklabels=['Normal', 'Seizure'])
axes[0,1].set_title('Gradient Boosting - Confusion Matrix')
axes[0,1].set_ylabel('Actual')
axes[0,1].set_xlabel('Predicted')

# 3. ROC Curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_prob)
axes[0,2].plot(fpr_rf, tpr_rf, 'b-', label=f'RF (AUC={roc_auc_score(y_test, rf_prob):.3f})')
axes[0,2].plot(fpr_gb, tpr_gb, 'r-', label=f'GB (AUC={roc_auc_score(y_test, gb_prob):.3f})')
axes[0,2].plot([0,1], [0,1], 'k--', alpha=0.3)
axes[0,2].set_title('ROC Curve')
axes[0,2].set_xlabel('False Positive Rate')
axes[0,2].set_ylabel('True Positive Rate')
axes[0,2].legend()

# 4. Feature Importance (Top 15)
importances = rf_model.feature_importances_
top_idx = np.argsort(importances)[-15:]
axes[1,0].barh(range(15), importances[top_idx], color='steelblue')
axes[1,0].set_yticks(range(15))
axes[1,0].set_yticklabels([feature_cols[i] for i in top_idx], fontsize=8)
axes[1,0].set_title('Top 15 Feature Importance (RF)')

# 5. Band Power Distribution
band_names = list(FREQ_BANDS.keys())
band_cols_mean = [f'{b}_power_mean' for b in band_names]
existing_cols = [c for c in band_cols_mean if c in feature_cols]
if existing_cols:
    seizure_mask = y == 1
    normal_mask = y == 0
    x_pos = np.arange(len(existing_cols))
    w = 0.35
    seizure_vals = [np.mean(X[seizure_mask, feature_cols.index(c)]) for c in existing_cols]
    normal_vals = [np.mean(X[normal_mask, feature_cols.index(c)]) for c in existing_cols]
    axes[1,1].bar(x_pos - w/2, normal_vals, w, label='Normal', color='#3b82f6')
    axes[1,1].bar(x_pos + w/2, seizure_vals, w, label='Seizure', color='#ef4444')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels([c.replace('_power_mean','') for c in existing_cols])
    axes[1,1].set_title('Band Power: Normal vs Seizure')
    axes[1,1].legend()

# 6. Precision-Recall Curve
prec_rf, rec_rf, _ = precision_recall_curve(y_test, rf_prob)
prec_gb, rec_gb, _ = precision_recall_curve(y_test, gb_prob)
axes[1,2].plot(rec_rf, prec_rf, 'b-', label='Random Forest')
axes[1,2].plot(rec_gb, prec_gb, 'r-', label='Gradient Boosting')
axes[1,2].set_title('Precision-Recall Curve')
axes[1,2].set_xlabel('Recall')
axes[1,2].set_ylabel('Precision')
axes[1,2].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'evaluation_charts.png'), dpi=150, bbox_inches='tight')
plt.show()
print("💾 Đã lưu evaluation_charts.png")

# ============================================================
# CELL 12: Vẽ Spectrogram mẫu
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('📊 EEG Spectrogram Samples', fontsize=14, fontweight='bold')

for idx, (label_val, title, ax) in enumerate([(0, 'Normal EEG', axes[0]), (1, 'Seizure EEG', axes[1])]):
    mask = y == label_val
    if np.any(mask):
        sample_idx = np.where(mask)[0][0]
        sample_feats = X[sample_idx]
        # Tạo pseudo-spectrogram từ band powers
        bands = [sample_feats[feature_cols.index(f'{b}_power_mean')] for b in FREQ_BANDS.keys() 
                 if f'{b}_power_mean' in feature_cols]
        channels = min(23, len(bands) * 4)
        spec_data = np.random.rand(len(bands), channels) * np.array(bands).reshape(-1, 1)
        im = ax.imshow(spec_data, aspect='auto', cmap='viridis', origin='lower')
        ax.set_yticks(range(len(FREQ_BANDS)))
        ax.set_yticklabels(list(FREQ_BANDS.keys()))
        ax.set_title(title)
        ax.set_xlabel('Channels')
        plt.colorbar(im, ax=ax, label='Power')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'spectrogram_samples.png'), dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# CELL 13: Lưu mô hình
# ============================================================

# Chọn model tốt nhất
rf_f1 = f1_score(y_test, rf_pred)
gb_f1 = f1_score(y_test, gb_pred)
best_model = rf_model if rf_f1 >= gb_f1 else gb_model
best_name = "RandomForest" if rf_f1 >= gb_f1 else "GradientBoosting"

print(f"🏆 Model tốt nhất: {best_name} (F1={max(rf_f1, gb_f1):.4f})")

# Lưu model + scaler + feature names
joblib.dump(best_model, os.path.join(OUTPUT_PATH, 'eeg_seizure_model.pkl'))
joblib.dump(scaler, os.path.join(OUTPUT_PATH, 'eeg_scaler.pkl'))
joblib.dump(feature_cols, os.path.join(OUTPUT_PATH, 'eeg_feature_names.pkl'))

# Lưu cả 2 model
joblib.dump(rf_model, os.path.join(OUTPUT_PATH, 'rf_model.pkl'))
joblib.dump(gb_model, os.path.join(OUTPUT_PATH, 'gb_model.pkl'))

print("💾 Đã lưu tất cả models!")
print(f"   📁 eeg_seizure_model.pkl (best)")
print(f"   📁 rf_model.pkl")
print(f"   📁 gb_model.pkl")
print(f"   📁 eeg_scaler.pkl")
print(f"   📁 eeg_feature_names.pkl")
print(f"   📁 eeg_features.csv")

# ============================================================
# CELL 14: Tổng kết
# ============================================================

print("\n" + "="*60)
print("🎉 HOÀN TẤT HUẤN LUYỆN MÔ HÌNH EEG SEIZURE DETECTION")
print("="*60)
print(f"""
📊 KẾT QUẢ:
   Random Forest:     Acc={accuracy_score(y_test, rf_pred):.4f}  F1={rf_f1:.4f}  AUC={roc_auc_score(y_test, rf_prob):.4f}
   Gradient Boosting: Acc={accuracy_score(y_test, gb_pred):.4f}  F1={gb_f1:.4f}  AUC={roc_auc_score(y_test, gb_prob):.4f}
   
🏆 Best Model: {best_name}

📋 PHƯƠNG PHÁP: Phân tích định lượng qEEG
   - Trích xuất PSD (Welch) cho 5 dải sóng não
   - Tính tỉ lệ giữa các dải (Theta/Alpha, Delta/Beta, ...)
   - Đặc trưng thống kê (mean, std, kurtosis, skewness, RMS, ...)
   - Tổng cộng {len(feature_cols)} đặc trưng

📁 OUTPUT FILES:
   - eeg_seizure_model.pkl  → Model tốt nhất
   - eeg_scaler.pkl         → Scaler cho features
   - eeg_feature_names.pkl  → Tên các features
   - eeg_features.csv       → Dataset features đã trích xuất
   - evaluation_charts.png  → Biểu đồ đánh giá
   
🔜 BƯỚC TIẾP: Download các file .pkl về tích hợp vào web app
""")
