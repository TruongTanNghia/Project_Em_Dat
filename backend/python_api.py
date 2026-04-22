"""
🧠 EEG Seizure Detection - Python API Service
Model: 1D-CNN + BiGRU + Temporal Attention (PyTorch)
Dataset: CHB-MIT Scalp EEG Database
Chạy song song với Node.js server trên port 5000
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import json
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as sp_signal

app = Flask(__name__)
CORS(app)

# ===== CONFIG =====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
SAMPLING_RATE = 256
WINDOW_SIZE = 4  # seconds
WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_SIZE  # 1024
MAX_CHANNELS = 23
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 40.0
FILTER_ORDER = 4

FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 50)
}

# ===== MODEL ARCHITECTURE (must match training) =====
class TemporalAttention(nn.Module):
    """Soft attention trên chiều thời gian — học cách tập trung vào vùng sóng bất thường."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, gru_output):
        scores = self.attention(gru_output).squeeze(-1)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        context = (gru_output * weights).sum(dim=1)
        return context, weights.squeeze(-1)

class EEG_CNN_BiGRU_Attention(nn.Module):
    """
    Hybrid Architecture: 1D-CNN + BiGRU + Temporal Attention
    - CNN blocks + Spatial Dropout: trích xuất mẫu sóng cục bộ
    - 2-layer BiGRU + dropout: nắm bắt ngữ cảnh thời gian
    - Temporal Attention: tập trung vào vùng bất thường
    - Classifier + Dropout: phân loại với regularization
    """
    def __init__(self, in_channels=23, gru_hidden=64, num_classes=1):
        super().__init__()
        
        # CNN Block 1: (23, 1024) -> (32, 256)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        self.drop1 = nn.Dropout(0.1)
        
        # CNN Block 2: (32, 256) -> (64, 64)
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        self.drop2 = nn.Dropout(0.2)
        
        # CNN Block 3: (64, 64) -> (128, 64)
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.drop3 = nn.Dropout(0.2)
        
        # BiGRU: 2 layers
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Temporal Attention
        self.attention = TemporalAttention(gru_hidden * 2)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(gru_hidden * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.drop1(self.conv_block1(x))
        x = self.drop2(self.conv_block2(x))
        x = self.drop3(self.conv_block3(x))
        x = x.permute(0, 2, 1)  # (B, seq_len, features)
        gru_out, _ = self.gru(x)
        context, attn_weights = self.attention(gru_out)
        out = self.classifier(context)
        return out

# ===== LOAD MODEL =====
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load metadata
metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

THRESHOLD = metadata.get('best_threshold', 0.4)
print(f"[INFO] Model metadata loaded: {metadata['architecture']}")
print(f"   ROC-AUC: {metadata.get('roc_auc', 'N/A'):.4f}")
print(f"   PR-AUC:  {metadata.get('pr_auc', 'N/A'):.4f}")
print(f"   Threshold: {THRESHOLD}")

# Load model weights
model = EEG_CNN_BiGRU_Attention(in_channels=MAX_CHANNELS).to(DEVICE)
model_path = os.path.join(MODEL_DIR, 'best_cnn_model.pth')
state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)

# Handle DataParallel prefix
first_key = next(iter(state_dict))
if first_key.startswith('module.'):
    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()
print(f"[OK] CNN model loaded on {DEVICE} ({sum(p.numel() for p in model.parameters()):,} params)")

# ===== BANDPASS FILTER =====
def create_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    sos = sp_signal.butter(order, [lowcut/nyq, highcut/nyq], btype='band', output='sos')
    return sos

BANDPASS_SOS = create_bandpass(BANDPASS_LOW, BANDPASS_HIGH, SAMPLING_RATE, FILTER_ORDER)
print(f"[OK] Bandpass filter: {BANDPASS_LOW}-{BANDPASS_HIGH} Hz")

# ===== HELPER: BAND POWER (for frequency analysis display) =====
def compute_band_power(data, sf, band):
    low, high = band
    freqs, psd = sp_signal.welch(data, sf, nperseg=min(len(data), 256))
    idx = np.logical_and(freqs >= low, freqs <= high)
    return np.trapz(psd[idx], freqs[idx]) if np.any(idx) else 0

# ===== HELPER: PER-WINDOW 4-TIER CLASSIFICATION =====
def classify_window(prob):
    """
    Phan loai tung window dua tren xac suat model:
    - < 20%  : Binh thuong (noise / FP)
    - 20-40% : Nghi ngo (can theo doi)
    - 40-70% : Bat thuong (co hoat dong bat thuong)
    - >= 70% : Nguy hiem (co giat ro rang)
    """
    p = float(prob)
    if p >= 0.70:
        return {'label': 'Nguy hiem', 'severity': 'severe', 'color': '#ef4444'}
    elif p >= 0.40:
        return {'label': 'Bat thuong', 'severity': 'moderate', 'color': '#f97316'}
    elif p >= 0.20:
        return {'label': 'Nghi ngo', 'severity': 'mild', 'color': '#f59e0b'}
    else:
        return {'label': 'Binh thuong', 'severity': 'normal', 'color': '#10b981'}

# ===== API ENDPOINTS =====
@app.route('/api/predict-edf', methods=['POST'])
def predict_edf():
    """Nhận file .edf → CNN inference → trả về kết quả phân tích"""
    if 'edfFile' not in request.files:
        return jsonify({'error': 'Không có file .edf'}), 400
    
    file = request.files['edfFile']
    if not file.filename.lower().endswith('.edf'):
        return jsonify({'error': 'Chỉ hỗ trợ file .edf'}), 400
    
    tmp = tempfile.NamedTemporaryFile(suffix='.edf', delete=False)
    file.save(tmp.name)
    tmp.close()
    
    try:
        import pyedflib
        
        reader = pyedflib.EdfReader(tmp.name)
        n_channels_file = reader.signals_in_file
        labels = reader.getSignalLabels()
        sf = reader.getSampleFrequency(0)
        n_samples = reader.getNSamples()[0]
        
        # Filter EEG channels only
        eeg_idx = [i for i, lb in enumerate(labels)
                   if lb.strip() not in ['-', '']
                   and 'ECG' not in lb.upper()
                   and 'VNS' not in lb.upper()][:MAX_CHANNELS]
        
        if len(eeg_idx) < 3:
            reader.close()
            return jsonify({'error': 'File EDF không đủ kênh EEG (cần ≥ 3)'}), 400
        
        # Read raw EEG data
        data = np.array([reader.readSignal(i).astype(np.float32) for i in eeg_idx])
        channel_names = [labels[i] for i in eeg_idx]
        reader.close()
        
        # Pad to MAX_CHANNELS if needed
        if data.shape[0] < MAX_CHANNELS:
            pad = np.zeros((MAX_CHANNELS - data.shape[0], data.shape[1]), dtype=np.float32)
            data = np.vstack([data, pad])
        
        # Apply bandpass filter (vectorized)
        data_filtered = sp_signal.sosfiltfilt(BANDPASS_SOS, data, axis=1).astype(np.float32)

        # ===== SLIDING WINDOW INFERENCE =====
        step = WINDOW_SAMPLES  # non-overlapping for inference
        predictions = []
        probabilities = []
        window_results = []
        
        # Batch inference for speed
        windows = []
        window_starts = []
        
        for start in range(0, n_samples - WINDOW_SAMPLES, step):
            end = start + WINDOW_SAMPLES
            window = data_filtered[:, start:end]
            
            # Z-score normalization (per-channel, per-window — matches training)
            mean_w = window.mean(axis=1, keepdims=True)
            std_w = window.std(axis=1, keepdims=True)
            std_w[std_w < 1e-6] = 1e-6
            window = (window - mean_w) / std_w
            
            windows.append(window)
            window_starts.append(start)
            
            # Process in batches of 64
            if len(windows) >= 64:
                batch = np.stack(windows)
                batch_tensor = torch.from_numpy(batch).float().to(DEVICE)
                
                with torch.inference_mode():
                    logits = model(batch_tensor).squeeze(1)
                    probs = torch.sigmoid(logits).cpu().numpy()
                
                for i, (s, prob) in enumerate(zip(window_starts, probs)):
                    pred = 1 if prob >= THRESHOLD else 0
                    wclass = classify_window(prob)
                    predictions.append(pred)
                    probabilities.append(float(prob))
                    
                    # Compute band powers for this window (on original unfiltered for display)
                    w_raw = data[:len(eeg_idx), s:s+WINDOW_SAMPLES]
                    band_powers = {}
                    for band_name, band_range in FREQ_BANDS.items():
                        bp_vals = [compute_band_power(w_raw[ch], sf, band_range)
                                   for ch in range(min(3, len(eeg_idx)))]  # Only first 3 channels for speed
                        band_powers[band_name] = round(float(np.mean(bp_vals)), 6)
                    
                    window_results.append({
                        'startSec': round(s / sf, 1),
                        'endSec': round((s + WINDOW_SAMPLES) / sf, 1),
                        'prediction': wclass['label'],
                        'severity': wclass['severity'],
                        'color': wclass['color'],
                        'probability': round(float(prob) * 100, 2),
                        'bandPowers': band_powers
                    })
                
                windows = []
                window_starts = []
        
        # Process remaining windows
        if windows:
            batch = np.stack(windows)
            batch_tensor = torch.from_numpy(batch).float().to(DEVICE)
            
            with torch.inference_mode():
                logits = model(batch_tensor).squeeze(1)
                probs = torch.sigmoid(logits).cpu().numpy()
            
            for i, (s, prob) in enumerate(zip(window_starts, probs)):
                pred = 1 if prob >= THRESHOLD else 0
                wclass = classify_window(prob)
                predictions.append(pred)
                probabilities.append(float(prob))
                
                w_raw = data[:len(eeg_idx), s:s+WINDOW_SAMPLES]
                band_powers = {}
                for band_name, band_range in FREQ_BANDS.items():
                    bp_vals = [compute_band_power(w_raw[ch], sf, band_range)
                               for ch in range(min(3, len(eeg_idx)))]
                    band_powers[band_name] = round(float(np.mean(bp_vals)), 6)
                
                window_results.append({
                    'startSec': round(s / sf, 1),
                    'endSec': round((s + WINDOW_SAMPLES) / sf, 1),
                    'prediction': wclass['label'],
                    'severity': wclass['severity'],
                    'color': wclass['color'],
                    'probability': round(float(prob) * 100, 2),
                    'bandPowers': band_powers
                })
        
        # ===== OVERALL RESULTS =====
        total_windows = len(predictions)
        abnormal_windows = sum(predictions)
        abnormal_ratio = abnormal_windows / max(total_windows, 1)
        avg_prob = np.mean(probabilities) if probabilities else 0
        
        # Severity distribution (per-window breakdown)
        sev_counts = {'normal': 0, 'mild': 0, 'moderate': 0, 'severe': 0}
        for w in window_results:
            sev_counts[w['severity']] = sev_counts.get(w['severity'], 0) + 1
        
        # ===== 4-TIER CLINICAL SEVERITY CLASSIFICATION =====
        #
        # Model FPR ~9.5%: EEG hoan toan binh thuong co the bi flag ~5-10% windows
        # => Can dat nguong du cao de tranh bao dong gia
        #
        # | Muc          | abnormal_ratio | avg_prob  | severity |
        # |--------------|----------------|-----------|----------|
        # | BINH THUONG  | <= 5%          | <= 0.20   | normal   |
        # | CANH BAO NHE | 5% - 15%       | 0.20-0.35 | mild     |
        # | BAT THUONG   | 15% - 30%      | 0.35-0.50 | moderate |
        # | NGHIEM TRONG | > 30%          | > 0.50    | severe   |
        
        if abnormal_ratio > 0.30 or avg_prob > 0.50:
            overall = 'NGHIEM TRONG'
            severity = 'severe'
            severity_vn = 'Nghiem trong - Nghi ngo co giat lien tuc'
        elif abnormal_ratio > 0.15 or avg_prob > 0.35:
            overall = 'BAT THUONG'
            severity = 'moderate'
            severity_vn = 'Bat thuong - Phat hien hoat dong bat thuong ro rang'
        elif abnormal_ratio > 0.05 or avg_prob > 0.20:
            overall = 'CANH BAO NHE'
            severity = 'mild'
            severity_vn = 'Canh bao nhe - Mot so doan nghi ngo, can theo doi'
        else:
            overall = 'BINH THUONG'
            severity = 'normal'
            severity_vn = 'Binh thuong - Khong phat hien bat thuong dang ke'
        
        # ===== FREQUENCY BAND ANALYSIS =====
        sample_windows = min(10, total_windows)
        avg_bands = {}
        for band_name in FREQ_BANDS:
            vals = [w['bandPowers'].get(band_name, 0) for w in window_results[:sample_windows]]
            total_power = sum(
                sum(w['bandPowers'].get(b, 0) for b in FREQ_BANDS)
                for w in window_results[:sample_windows]
            ) / max(sample_windows, 1)
            avg_bands[band_name] = {
                'power': round(float(np.mean(vals)), 6),
                'relative': round(float(np.mean(vals)) / max(total_power, 1e-10) * 100, 2)
            }
        
        # ===== WAVEFORM DATA FOR VISUALIZATION =====
        # Show ALL channels, but limit points per channel for performance
        max_viz_points = 800  # per channel (23 ch x 800 pts = ~18K points total)
        viz_channels = len(eeg_idx)  # ALL channels
        downsample_factor = max(1, n_samples // max_viz_points)
        
        waveform_data = {}
        for ch_i in range(viz_channels):
            ch_signal = data[ch_i]
            n_blocks = len(ch_signal) // downsample_factor
            if n_blocks > 0:
                truncated = ch_signal[:n_blocks * downsample_factor]
                downsampled = truncated.reshape(n_blocks, downsample_factor).mean(axis=1)
                waveform_data[channel_names[ch_i]] = [round(float(v), 2) for v in downsampled]
        
        waveform_time = [round(i * downsample_factor / sf, 2) 
                         for i in range(len(list(waveform_data.values())[0]))]
        
        # Only highlight moderate (>=40%) and severe (>=70%) → cleaner chart
        abnormal_regions = [
            {'start': w['startSec'], 'end': w['endSec'], 'prob': w['probability'], 
             'severity': w['severity'], 'label': w['prediction']}
            for w in window_results if w['severity'] in ('moderate', 'severe')
        ]
        
        result = {
            'success': True,
            'overall': overall,
            'severity': severity,
            'severityDescription': severity_vn,
            'confidence': round(avg_prob * 100, 2),
            'totalWindows': total_windows,
            'abnormalWindows': abnormal_windows,
            'abnormalRatio': round(abnormal_ratio * 100, 2),
            'severityCounts': sev_counts,
            'duration': round(n_samples / sf, 1),
            'channels': len(eeg_idx),
            'channelNames': channel_names,
            'samplingRate': sf,
            'threshold': THRESHOLD,
            'frequencyBands': avg_bands,
            'windowResults': window_results,
            'abnormalRegions': abnormal_regions,
            'waveform': {
                'time': waveform_time,
                'channels': waveform_data
            },
            'modelInfo': {
                'name': metadata['architecture'],
                'type': 'Deep Learning (PyTorch)',
                'roc_auc': metadata.get('roc_auc', 0),
                'pr_auc': metadata.get('pr_auc', 0),
                'threshold': THRESHOLD,
                'bandpass': f'{BANDPASS_LOW}-{BANDPASS_HIGH} Hz'
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500
    finally:
        os.unlink(tmp.name)

@app.route('/api/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'model': metadata['architecture'],
        'type': 'Deep Learning (1D-CNN + BiGRU + Attention)',
        'device': str(DEVICE),
        'threshold': THRESHOLD,
        'roc_auc': metadata.get('roc_auc', 0),
        'pr_auc': metadata.get('pr_auc', 0),
        'bandpass': f'{BANDPASS_LOW}-{BANDPASS_HIGH} Hz',
        'window_size': f'{WINDOW_SIZE}s ({WINDOW_SAMPLES} samples)',
        'params': sum(p.numel() for p in model.parameters()),
        'status': 'active'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'loaded', 'type': 'CNN+BiGRU+Attention'})

if __name__ == '__main__':
    print("[START] EEG Python API (CNN) starting on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
