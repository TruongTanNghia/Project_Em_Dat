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
CORS(app, expose_headers=['Content-Disposition'])

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
        tb = traceback.format_exc()
        print(f"[ERROR] /api/predict-edf failed: {e}\n{tb}", flush=True)
        return jsonify({'error': str(e), 'trace': tb}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

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

# ===== LUNG TUMOR SEGMENTATION (DeepLabV3-ResNet50) =====
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms as T
from PIL import Image
import io
import base64
import cv2

LUNG_MODEL_DIR = os.path.join(MODEL_DIR, 'LIDC')
LUNG_MODEL_PATH = os.path.join(LUNG_MODEL_DIR, 'best_model.pth')
LUNG_INPUT_SIZE = 512
LUNG_THRESHOLD = 0.3
LUNG_PIXEL_SPACING = 0.7  # mm/pixel in the *original* CT image (LIDC ≈ 0.7)

lung_model = None
lung_model_loaded = False

if os.path.exists(LUNG_MODEL_PATH):
    try:
        lung_model = deeplabv3_resnet50(weights=None, num_classes=1, aux_loss=True)
        lung_sd = torch.load(LUNG_MODEL_PATH, map_location=DEVICE, weights_only=False)
        lung_model.load_state_dict(lung_sd)
        lung_model.to(DEVICE)
        lung_model.eval()
        lung_model_loaded = True
        lung_params = sum(p.numel() for p in lung_model.parameters())
        print(f"[OK] Lung DeepLabV3 model loaded on {DEVICE} ({lung_params:,} params)")
    except Exception as e:
        print(f"[WARN] Failed to load lung model: {e}")
else:
    print(f"[WARN] Lung model not found at {LUNG_MODEL_PATH}")

# Lung preprocessing (match ImageNet normalization used during training)
lung_preprocess = T.Compose([
    T.Resize((LUNG_INPUT_SIZE, LUNG_INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def analyze_lung_mask(mask_np, orig_w, orig_h):
    """Analyze binary mask to extract tumor metrics."""
    # mask_np: (H, W) binary uint8 (0 or 255)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {
            'detected': False,
            'tumorCount': 0,
            'tumors': [],
            'totalAreaPx': 0,
            'totalAreaMm2': 0,
            'maxDiameterMm': 0,
            'volumeMm3': 0,
            'overallConfidence': 0,
            'position': 'N/A',
            'positionSub': '',
        }

    tumors = []
    total_area_px = 0
    max_diameter_mm = 0

    for i, cnt in enumerate(contours):
        area_px = cv2.contourArea(cnt)
        if area_px < 1:  # only skip zero-area noise
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / max(M['m00'], 1))
        cy = int(M['m01'] / max(M['m00'], 1))

        # Scale to original image coords (mask is upsampled to LUNG_INPUT_SIZE
        # internally; LUNG_PIXEL_SPACING applies to the *original* image, so
        # convert pixel measurements back to original-space first).
        sx = orig_w / LUNG_INPUT_SIZE
        sy = orig_h / LUNG_INPUT_SIZE

        # Diameter in mm: (max bbox dim in 512-space) × (orig/512 scale) × mm/px
        diameter_mm = max(w * sx, h * sy) * LUNG_PIXEL_SPACING
        area_mm2 = area_px * sx * sy * (LUNG_PIXEL_SPACING ** 2)

        if diameter_mm > max_diameter_mm:
            max_diameter_mm = diameter_mm

        # Position mapping based on centroid — label-match convention:
        # the side reported here matches the L/R label drawn on the displayed
        # CT image (L on viewer-left, R on viewer-right). So image-left → trái,
        # image-right → phải, regardless of strict radiological orientation.
        half_w = LUNG_INPUT_SIZE / 2
        third_h = LUNG_INPUT_SIZE / 3
        side = 'Phổi trái' if cx < half_w else 'Phổi phải'
        if cy < third_h:
            lobe = 'Thùy trên'
        elif cy < 2 * third_h:
            lobe = 'Thùy giữa'
        else:
            lobe = 'Thùy dưới'

        # Simplified contour for 3D visualization (max 60 pts)
        contour_pts = cnt.squeeze().tolist()  # [[x,y], ...]
        if len(contour_pts) > 60:
            step = max(1, len(contour_pts) // 60)
            contour_pts = contour_pts[::step]
        # Normalize to 0-1 range (relative to 512x512)
        contour_norm = [[round(p[0]/LUNG_INPUT_SIZE, 4), round(p[1]/LUNG_INPUT_SIZE, 4)]
                        for p in contour_pts if isinstance(p, (list, tuple)) and len(p) == 2]

        tumors.append({
            'id': i,
            'bbox': {'x': int(x * sx), 'y': int(y * sy),
                     'w': int(w * sx), 'h': int(h * sy)},
            'bboxNorm': {'x': x, 'y': y, 'w': w, 'h': h},
            'centroid': {'x': int(cx * sx), 'y': int(cy * sy)},
            'centroidNorm': {'x': cx, 'y': cy},
            'areaPx': int(area_px),
            'areaMm2': round(area_mm2, 2),
            'diameterMm': round(diameter_mm, 1),
            'position': f'{lobe} - {side}',
            'positionSub': f'({"left" if "trái" in side else "right"} {"Upper" if cy < third_h else "Middle" if cy < 2*third_h else "Lower"} Lobe)',
            'contourNorm': contour_norm,  # normalized contour for 3D
        })
        total_area_px += area_px

    # Sort by area descending (largest tumor first)
    tumors.sort(key=lambda t: t['areaPx'], reverse=True)

    # Total area uses the same orig/512 scaling as per-tumor.
    sx_total = orig_w / LUNG_INPUT_SIZE
    sy_total = orig_h / LUNG_INPUT_SIZE
    total_area_mm2 = total_area_px * sx_total * sy_total * (LUNG_PIXEL_SPACING ** 2)
    # Volume estimate assuming 1.25mm slice thickness
    volume_mm3 = total_area_mm2 * 1.25

    # Best position from largest tumor
    main_pos = tumors[0]['position'] if tumors else 'N/A'
    main_pos_sub = tumors[0]['positionSub'] if tumors else ''

    return {
        'detected': True,
        'tumorCount': len(tumors),
        'tumors': tumors,
        'totalAreaPx': int(total_area_px),
        'totalAreaMm2': round(total_area_mm2, 2),
        'maxDiameterMm': round(max_diameter_mm, 1),
        'volumeMm3': round(volume_mm3, 2),
        'position': main_pos,
        'positionSub': main_pos_sub,
    }

def create_overlay_image(orig_pil, mask_np_512, prob_map_512, orig_size=None):
    """Create overlay: original + red tumor highlight + green contour + bbox.

    `orig_size` = (orig_w, orig_h) of the *source* image, used to convert
    bbox measurements from 512-space back to original-space mm. If omitted,
    falls back to LUNG_INPUT_SIZE (i.e. assumes mask is already at original).
    """
    # Resize original to 512x512 for overlay
    orig_resized = orig_pil.convert('RGB').resize((LUNG_INPUT_SIZE, LUNG_INPUT_SIZE))
    overlay = np.array(orig_resized).copy()

    if orig_size is None:
        orig_size = (LUNG_INPUT_SIZE, LUNG_INPUT_SIZE)
    sx_ovr = orig_size[0] / LUNG_INPUT_SIZE
    sy_ovr = orig_size[1] / LUNG_INPUT_SIZE

    # Red overlay for tumor regions (semi-transparent)
    tumor_mask = mask_np_512 > 0
    if np.any(tumor_mask):
        red_overlay = overlay.copy()
        red_overlay[tumor_mask] = [255, 50, 50]
        overlay = cv2.addWeighted(overlay, 0.6, red_overlay, 0.4, 0)

        # Green contour border
        contours, _ = cv2.findContours(mask_np_512, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 100), 2)

        # Bounding box + label for largest contour
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)
            diameter = max(w * sx_ovr, h * sy_ovr) * LUNG_PIXEL_SPACING
            label = f'{diameter:.1f} mm'
            cv2.putText(overlay, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Encode as base64 PNG
    overlay_pil = Image.fromarray(overlay)
    buf = io.BytesIO()
    overlay_pil.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{b64}'

def create_mask_image_b64(mask_np):
    """Encode binary mask as base64 green PNG for frontend overlay."""
    h, w = mask_np.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    tumor = mask_np > 0
    rgba[tumor] = [0, 255, 100, 140]  # green semi-transparent
    mask_pil = Image.fromarray(rgba, 'RGBA')
    buf = io.BytesIO()
    mask_pil.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{b64}'


# LIDC nodule slice spacing assumptions:
#   in-plane (X,Y) = 0.7 mm/pixel (LUNG_PIXEL_SPACING)
#   slice thickness (Z)   = 1.25 mm (typical LIDC reconstruction)
# When reslicing axial → coronal/sagittal we stretch the Z axis by
# slice_thickness/pixel_spacing so anatomy isn't squashed.
MPR_Z_ASPECT = 1.25 / LUNG_PIXEL_SPACING  # ≈ 1.79
MPR_OUT_SIZE = 256  # display size for the small sub-view boxes


def build_consensus_mask_volume(nod_path, slice_files, consensus_threshold,
                                 orig_w, orig_h):
    """Stack per-slice consensus masks into a (Z, H, W) uint8 volume.

    Per slice: pixel is positive if ≥ consensus_threshold annotators agreed.
    Same logic as the per-slice consensus used in predict_lung_dataset, but
    repeated across every slice in the nodule so we can MIP the tumor mask.
    """
    mask_dirs = sorted([d for d in os.listdir(nod_path)
                        if d.startswith('mask-') and
                        os.path.isdir(os.path.join(nod_path, d))])
    vol = []
    for sname in slice_files:
        per_slice = []
        for md in mask_dirs:
            mp = os.path.join(nod_path, md, sname)
            if os.path.exists(mp):
                m = np.array(Image.open(mp).convert('L'))
                if m.shape != (orig_h, orig_w):
                    m = cv2.resize(m, (orig_w, orig_h),
                                   interpolation=cv2.INTER_NEAREST)
                per_slice.append((m > 127).astype(np.uint8))
        if per_slice:
            consensus = (sum(per_slice) >= consensus_threshold).astype(np.uint8)
        else:
            consensus = np.zeros((orig_h, orig_w), dtype=np.uint8)
        vol.append(consensus)
    return np.stack(vol, axis=0)


def compute_mpr_views(img_dir, slice_files, cx, cy,
                      orig_w, orig_h, mask_volume=None):
    """MIP (Maximum Intensity Projection) views with optional tumor overlay.

    Returns (coronal_b64, sagittal_b64) as data-URIs, or (None, None)
    if there aren't enough slices to build a meaningful volume.
    cx/cy are in original slice coordinates (orig_w x orig_h) — used only
    for cropping the output window so the tumor stays centered.
    mask_volume (Z, H, W) uint8 binary, optional → adds red overlay.
    """
    if len(slice_files) < 2:
        return None, None

    vol = []
    for sname in slice_files:
        p = os.path.join(img_dir, sname)
        try:
            arr = np.array(Image.open(p).convert('L'))
        except Exception:
            return None, None
        if arr.shape != (orig_h, orig_w):
            arr = cv2.resize(arr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        vol.append(arr)
    volume = np.stack(vol, axis=0)  # (Z, H, W)
    Z, H, W = volume.shape

    cx_i = int(np.clip(cx, 0, W - 1))
    cy_i = int(np.clip(cy, 0, H - 1))

    # MIP: project the brightest voxel along each in-plane axis.
    # CT bone/tumor pixels are bright, lung air is dark → tumor stands out.
    coronal_mip  = volume.max(axis=1)   # (Z, W) — project Y
    sagittal_mip = volume.max(axis=2)   # (Z, H) — project X

    # MIP of the tumor mask, if provided
    cor_mask = sag_mask = None
    if mask_volume is not None and mask_volume.shape == volume.shape:
        cor_mask = mask_volume.max(axis=1)   # (Z, W)
        sag_mask = mask_volume.max(axis=2)   # (Z, H)

    target_z = max(2, int(round(Z * MPR_Z_ASPECT)))

    def _stretch_z(img2d):
        return cv2.resize(img2d, (img2d.shape[1], target_z),
                          interpolation=cv2.INTER_LINEAR)

    def _stretch_z_mask(img2d):
        return cv2.resize(img2d, (img2d.shape[1], target_z),
                          interpolation=cv2.INTER_NEAREST)

    coronal_r  = _stretch_z(coronal_mip)
    sagittal_r = _stretch_z(sagittal_mip)
    cor_mask_r = _stretch_z_mask(cor_mask)  if cor_mask  is not None else None
    sag_mask_r = _stretch_z_mask(sag_mask)  if sag_mask  is not None else None

    # Crop H/W axis around the tumor centroid. Wider crop = more anatomical
    # context outside the tumor (otherwise the MIP overlay covers everything
    # because LIDC nodule crops are tight around the lesion).
    crop_half = max(target_z, 26)

    def _crop_centered(img2d, center, half):
        n = img2d.shape[1]
        x0 = max(0, center - half)
        x1 = min(n, center + half)
        if x1 - x0 < 2 * half:
            if x0 == 0:
                x1 = min(n, 2 * half)
            elif x1 == n:
                x0 = max(0, n - 2 * half)
        return img2d[:, x0:x1], x0

    coronal_c,  cor_x0 = _crop_centered(coronal_r,  cx_i, crop_half)
    sagittal_c, sag_x0 = _crop_centered(sagittal_r, cy_i, crop_half)
    cor_mask_c = cor_mask_r[:, cor_x0:cor_x0 + coronal_c.shape[1]]  if cor_mask_r is not None else None
    sag_mask_c = sag_mask_r[:, sag_x0:sag_x0 + sagittal_c.shape[1]] if sag_mask_r is not None else None

    def _normalize_intensity(gray2d):
        # Stretch dynamic range so the MIP doesn't look washed out
        g = gray2d.astype(np.float32)
        lo, hi = np.percentile(g, [2, 99])
        if hi - lo < 1:
            return gray2d
        out = np.clip((g - lo) / (hi - lo), 0, 1) * 255
        return out.astype(np.uint8)

    def _decorate(gray2d, mask2d):
        gray2d = _normalize_intensity(gray2d)
        rgb = cv2.cvtColor(gray2d, cv2.COLOR_GRAY2RGB)
        h_, w_ = rgb.shape[:2]

        # Tumor mask overlay (light red) + bright outline so anatomy still
        # shows through but the tumor edge is unambiguous.
        if mask2d is not None and np.any(mask2d):
            mask_bool = mask2d > 0
            red_layer = rgb.copy()
            red_layer[mask_bool] = [255, 70, 90]
            rgb = cv2.addWeighted(rgb, 0.75, red_layer, 0.25, 0)
            mask_u8 = (mask_bool.astype(np.uint8)) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contours, -1, (255, 220, 90), 1)

        # Letterbox to square so the FE box (aspect-ratio: 1) renders cleanly
        side = max(h_, w_)
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        oy = (side - h_) // 2
        ox = (side - w_) // 2
        canvas[oy:oy + h_, ox:ox + w_] = rgb
        return cv2.resize(canvas, (MPR_OUT_SIZE, MPR_OUT_SIZE),
                          interpolation=cv2.INTER_LINEAR)

    coronal_decor  = _decorate(coronal_c,  cor_mask_c)
    sagittal_decor = _decorate(sagittal_c, sag_mask_c)

    def _encode(rgb_arr):
        pil = Image.fromarray(rgb_arr)
        buf = io.BytesIO()
        pil.save(buf, format='PNG')
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    return _encode(coronal_decor), _encode(sagittal_decor)


@app.route('/api/predict-lung', methods=['POST'])
def predict_lung():
    """Nhận ảnh CT phổi → DeepLabV3 segmentation → trả kết quả phân tích."""
    if not lung_model_loaded:
        return jsonify({'error': 'Lung model chưa được load'}), 503

    if 'lungImage' not in request.files:
        return jsonify({'error': 'Không có file ảnh'}), 400

    file = request.files['lungImage']
    allowed = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    if not file.filename.lower().endswith(allowed):
        return jsonify({'error': 'Chỉ hỗ trợ file ảnh (PNG, JPG, BMP, TIFF, WebP)'}), 400

    try:
        # Read image
        img_bytes = file.read()
        orig_pil = Image.open(io.BytesIO(img_bytes))
        orig_w, orig_h = orig_pil.size

        # Convert grayscale to RGB (LIDC images are grayscale)
        if orig_pil.mode == 'L':
            orig_pil = orig_pil.convert('RGB')
        elif orig_pil.mode == 'RGBA':
            orig_pil = orig_pil.convert('RGB')
        elif orig_pil.mode != 'RGB':
            orig_pil = orig_pil.convert('RGB')

        # Preprocess
        input_tensor = lung_preprocess(orig_pil).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.inference_mode():
            output = lung_model(input_tensor)
            logits = output['out']  # (1, 1, H, W)
            prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H, W) 0~1

        # Binary mask
        mask_binary = (prob_map >= LUNG_THRESHOLD).astype(np.uint8) * 255

        # Analyze mask for tumor metrics
        metrics = analyze_lung_mask(mask_binary, orig_w, orig_h)

        # Confidence from probability map
        tumor_pixels = mask_binary > 0
        has_tumor_pixels = np.any(tumor_pixels)
        if has_tumor_pixels:
            tumor_probs = prob_map[tumor_pixels]
            overall_confidence = float(np.mean(tumor_probs)) * 100
            max_confidence = float(np.max(tumor_probs)) * 100
        else:
            overall_confidence = 0
            max_confidence = 0

        # If mask has pixels but contour analysis found nothing (too small), 
        # create a fallback detection from raw pixel stats
        if has_tumor_pixels and not metrics['detected']:
            ys, xs = np.where(tumor_pixels)
            raw_area = len(xs)
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            cx, cy = int(xs.mean()), int(ys.mean())
            w_box = x_max - x_min + 1
            h_box = y_max - y_min + 1
            half_w = LUNG_INPUT_SIZE / 2
            third_h = LUNG_INPUT_SIZE / 3
            side = 'Phổi trái' if cx < half_w else 'Phổi phải'
            lobe = 'Thùy trên' if cy < third_h else ('Thùy giữa' if cy < 2 * third_h else 'Thùy dưới')
            sx = orig_w / LUNG_INPUT_SIZE
            sy = orig_h / LUNG_INPUT_SIZE
            # Convert measurements from 512-space back to original-space first
            diameter_mm = max(w_box * sx, h_box * sy) * LUNG_PIXEL_SPACING
            area_mm2 = raw_area * sx * sy * (LUNG_PIXEL_SPACING ** 2)
            metrics = {
                'detected': True,
                'tumorCount': 1,
                'tumors': [{
                    'id': 0,
                    'bbox': {'x': int(x_min * sx), 'y': int(y_min * sy), 'w': int(w_box * sx), 'h': int(h_box * sy)},
                    'bboxNorm': {'x': x_min, 'y': y_min, 'w': w_box, 'h': h_box},
                    'centroid': {'x': int(cx * sx), 'y': int(cy * sy)},
                    'centroidNorm': {'x': cx, 'y': cy},
                    'areaPx': raw_area,
                    'areaMm2': round(area_mm2, 2),
                    'diameterMm': round(diameter_mm, 1),
                    'position': f'{lobe} - {side}',
                    'positionSub': f'({side.replace("Phổi ", "")} {"Upper" if cy < third_h else "Middle" if cy < 2*third_h else "Lower"} Lobe)',
                }],
                'totalAreaPx': raw_area,
                'totalAreaMm2': round(area_mm2, 2),
                'maxDiameterMm': round(diameter_mm, 1),
                'volumeMm3': round(area_mm2 * 1.25, 2),
                'position': f'{lobe} - {side}',
                'positionSub': f'({side.replace("Phổi ", "")} {"Upper" if cy < third_h else "Middle" if cy < 2*third_h else "Lower"} Lobe)',
            }

        # Malignancy estimate (heuristic based on size + confidence)
        # Larger + higher confidence → more likely malignant
        if metrics['detected']:
            size_factor = min(metrics['maxDiameterMm'] / 30.0, 1.0)
            conf_factor = overall_confidence / 100.0
            malignancy = min(99, max(10, (size_factor * 0.5 + conf_factor * 0.5) * 100))
        else:
            malignancy = 0

        # Type classification based on size
        if not metrics['detected']:
            tumor_type = 'Không phát hiện'
            stage = 'N/A'
            stage_sub = ''
        elif metrics['maxDiameterMm'] <= 6:
            tumor_type = 'Nốt phổi nhỏ'
            stage = 'T1mi'
            stage_sub = '(≤ 6mm)'
        elif metrics['maxDiameterMm'] <= 10:
            tumor_type = 'Nốt phổi'
            stage = 'T1a'
            stage_sub = '(6 - 10mm)'
        elif metrics['maxDiameterMm'] <= 20:
            tumor_type = 'Nghi ngờ ác tính'
            stage = 'T1b'
            stage_sub = '(1 - 2 cm)'
        elif metrics['maxDiameterMm'] <= 30:
            tumor_type = 'Nghi ngờ ác tính'
            stage = 'T1c'
            stage_sub = '(2 - 3 cm)'
        else:
            tumor_type = 'Khối u lớn'
            stage = 'T2+'
            stage_sub = '(> 3 cm)'

        # Create overlay images
        overlay_b64 = create_overlay_image(orig_pil, mask_binary, prob_map, orig_size=(orig_w, orig_h))
        mask_b64 = create_mask_image_b64(mask_binary)

        # Encode original as base64 too (for side-by-side)
        orig_resized = orig_pil.resize((LUNG_INPUT_SIZE, LUNG_INPUT_SIZE))
        buf_orig = io.BytesIO()
        orig_resized.save(buf_orig, format='PNG')
        orig_b64 = f"data:image/png;base64,{base64.b64encode(buf_orig.getvalue()).decode('utf-8')}"

        # Probability heatmap as base64
        heatmap = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap_pil = Image.fromarray(heatmap_rgb)
        buf_heat = io.BytesIO()
        heatmap_pil.save(buf_heat, format='PNG')
        heatmap_b64 = f"data:image/png;base64,{base64.b64encode(buf_heat.getvalue()).decode('utf-8')}"

        result = {
            'success': True,
            'detected': metrics['detected'],
            'tumorCount': metrics['tumorCount'],
            'tumors': metrics['tumors'],
            'maxDiameterMm': metrics['maxDiameterMm'],
            'totalAreaMm2': metrics['totalAreaMm2'],
            'volumeMm3': metrics['volumeMm3'],
            'volumeCm3': round(metrics['volumeMm3'] / 1000, 2),
            'position': metrics['position'],
            'positionSub': metrics['positionSub'],
            'confidence': round(overall_confidence, 1),
            'maxConfidence': round(max_confidence, 1),
            'malignancy': round(malignancy, 1),
            'tumorType': tumor_type,
            'stage': stage,
            'stageSub': stage_sub,
            'overlayImage': overlay_b64,
            'maskImage': mask_b64,
            'originalImage': orig_b64,
            'heatmapImage': heatmap_b64,
            'originalSize': {'w': orig_w, 'h': orig_h},
            'modelInfo': {
                'name': 'DeepLabV3-ResNet50',
                'type': 'Semantic Segmentation',
                'dataset': 'LIDC-IDRI',
                'inputSize': LUNG_INPUT_SIZE,
                'threshold': LUNG_THRESHOLD,
            }
        }

        return jsonify(result)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] /api/predict-lung failed: {e}\n{tb}", flush=True)
        return jsonify({'error': str(e), 'trace': tb}), 500


@app.route('/api/lung-model-info', methods=['GET'])
def lung_model_info():
    return jsonify({
        'model': 'DeepLabV3-ResNet50',
        'type': 'Semantic Segmentation',
        'dataset': 'LIDC-IDRI',
        'device': str(DEVICE),
        'inputSize': LUNG_INPUT_SIZE,
        'threshold': LUNG_THRESHOLD,
        'params': sum(p.numel() for p in lung_model.parameters()) if lung_model else 0,
        'status': 'active' if lung_model_loaded else 'not loaded'
    })


# ============ LIDC DATASET BROWSER ============
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'LIDC-IDRI-slices')

@app.route('/api/lung-dataset', methods=['GET'])
def lung_dataset_list():
    """Trả danh sách bệnh nhân + nodules có trong dataset LIDC."""
    if not os.path.exists(DATASET_DIR):
        return jsonify({'error': 'Dataset folder not found', 'path': DATASET_DIR}), 404

    patients = []
    for patient_dir in sorted(os.listdir(DATASET_DIR)):
        patient_path = os.path.join(DATASET_DIR, patient_dir)
        if not os.path.isdir(patient_path):
            continue
        nodules = []
        for nod_dir in sorted(os.listdir(patient_path)):
            nod_path = os.path.join(patient_path, nod_dir)
            if not os.path.isdir(nod_path):
                continue
            img_dir = os.path.join(nod_path, 'images')
            slices = sorted(os.listdir(img_dir)) if os.path.exists(img_dir) else []
            mask_dirs = [d for d in os.listdir(nod_path)
                         if d.startswith('mask-') and os.path.isdir(os.path.join(nod_path, d))]
            mask_count = len(mask_dirs)
            # For each slice, count how many annotators marked it.
            # Pre-list every mask dir's files once into a set, then do O(1)
            # lookups — much faster than os.path.exists per slice (matters
            # with hundreds of patients on slow disks).
            mask_dir_files = {}
            for md in mask_dirs:
                try:
                    mask_dir_files[md] = set(os.listdir(os.path.join(nod_path, md)))
                except OSError:
                    mask_dir_files[md] = set()
            slice_marks = [
                sum(1 for fs in mask_dir_files.values() if s in fs)
                for s in slices
            ]
            nodules.append({
                'name': nod_dir,
                'sliceCount': len(slices),
                'maskCount': mask_count,
                'slices': slices,
                'sliceMarks': slice_marks,   # annotators per slice
            })
        if nodules:
            patients.append({
                'id': patient_dir,
                'nodules': nodules
            })
    return jsonify({'patients': patients, 'total': len(patients)})


@app.route('/api/predict-lung-dataset', methods=['POST'])
def predict_lung_dataset():
    """Dùng ground truth masks từ dataset thay vì model prediction.
    Body: { patientId, noduleName, sliceIndex }
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing JSON body'}), 400

    patient_id = data.get('patientId', '')
    nodule_name = data.get('noduleName', 'nodule-0')
    slice_idx = int(data.get('sliceIndex', 0))

    nod_path = os.path.join(DATASET_DIR, patient_id, nodule_name)
    if not os.path.exists(nod_path):
        return jsonify({'error': f'Nodule not found: {patient_id}/{nodule_name}'}), 404

    img_dir = os.path.join(nod_path, 'images')
    slices = sorted(os.listdir(img_dir)) if os.path.exists(img_dir) else []
    if slice_idx >= len(slices):
        return jsonify({'error': f'Slice {slice_idx} not found (total: {len(slices)})'}), 404

    slice_name = slices[slice_idx]
    img_path = os.path.join(img_dir, slice_name)

    try:
        # Load image
        orig_pil = Image.open(img_path)
        orig_w, orig_h = orig_pil.size
        if orig_pil.mode != 'RGB':
            orig_pil = orig_pil.convert('RGB')

        # Load ALL masks and create consensus
        mask_arrays = []
        for mask_dir_name in sorted(os.listdir(nod_path)):
            if not mask_dir_name.startswith('mask-'):
                continue
            mask_path = os.path.join(nod_path, mask_dir_name, slice_name)
            if os.path.exists(mask_path):
                m = np.array(Image.open(mask_path).convert('L'))
                mask_arrays.append((m > 127).astype(np.uint8))

        num_annotators = len(mask_arrays)
        if num_annotators == 0:
            return jsonify({'error': 'No mask annotations found'}), 404

        # Consensus: pixel is positive if ≥ 50% annotators agree
        # Consensus = at least 1 annotator. (Half-vote was rejecting slices
        # where 4 radiologists each marked tiny non-overlapping regions —
        # union sum stayed at 1, missing the half threshold.)
        consensus_threshold = 1
        mask_sum = sum(mask_arrays)
        consensus_mask = (mask_sum >= consensus_threshold).astype(np.uint8) * 255

        # Resize mask to LUNG_INPUT_SIZE for consistent analysis
        mask_resized = cv2.resize(consensus_mask, (LUNG_INPUT_SIZE, LUNG_INPUT_SIZE),
                                  interpolation=cv2.INTER_NEAREST)

        # Analyze with the ground truth mask
        metrics = analyze_lung_mask(mask_resized, orig_w, orig_h)

        # If contour analysis missed tiny regions, try fallback
        tumor_pixels = mask_resized > 0
        has_tumor_pixels = np.any(tumor_pixels)
        if has_tumor_pixels and not metrics['detected']:
            ys, xs = np.where(tumor_pixels)
            raw_area = len(xs)
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            cx, cy = int(xs.mean()), int(ys.mean())
            w_box = x_max - x_min + 1
            h_box = y_max - y_min + 1
            half_w = LUNG_INPUT_SIZE / 2
            third_h = LUNG_INPUT_SIZE / 3
            side = 'Phổi trái' if cx < half_w else 'Phổi phải'
            lobe = 'Thùy trên' if cy < third_h else ('Thùy giữa' if cy < 2 * third_h else 'Thùy dưới')
            sx = orig_w / LUNG_INPUT_SIZE
            sy = orig_h / LUNG_INPUT_SIZE
            # Convert measurements from 512-space back to original-space first
            diameter_mm = max(w_box * sx, h_box * sy) * LUNG_PIXEL_SPACING
            area_mm2 = raw_area * sx * sy * (LUNG_PIXEL_SPACING ** 2)
            metrics = {
                'detected': True, 'tumorCount': 1,
                'tumors': [{'id': 0,
                    'bbox': {'x': int(x_min*sx), 'y': int(y_min*sy), 'w': int(w_box*sx), 'h': int(h_box*sy)},
                    'bboxNorm': {'x': x_min, 'y': y_min, 'w': w_box, 'h': h_box},
                    'centroid': {'x': int(cx*sx), 'y': int(cy*sy)},
                    'centroidNorm': {'x': cx, 'y': cy},
                    'areaPx': raw_area, 'areaMm2': round(area_mm2, 2),
                    'diameterMm': round(diameter_mm, 1),
                    'position': f'{lobe} - {side}',
                    'positionSub': f'({side.replace("Phổi ","")} {"Upper" if cy<third_h else "Middle" if cy<2*third_h else "Lower"} Lobe)',
                }],
                'totalAreaPx': raw_area, 'totalAreaMm2': round(area_mm2, 2),
                'maxDiameterMm': round(diameter_mm, 1),
                'volumeMm3': round(area_mm2 * 1.25, 2),
                'position': f'{lobe} - {side}',
                'positionSub': f'({side.replace("Phổi ","")} {"Upper" if cy<third_h else "Middle" if cy<2*third_h else "Lower"} Lobe)',
            }

        # Confidence = annotator agreement (how many agreed)
        if has_tumor_pixels:
            agreement_map = mask_sum[consensus_mask > 0] if np.any(consensus_mask > 0) else np.array([0])
            overall_confidence = float(np.mean(agreement_map) / num_annotators) * 100
        else:
            overall_confidence = 0

        # Malignancy from size
        if metrics['detected']:
            size_factor = min(metrics['maxDiameterMm'] / 30.0, 1.0)
            conf_factor = overall_confidence / 100.0
            malignancy = min(99, max(10, (size_factor * 0.5 + conf_factor * 0.5) * 100))
        else:
            malignancy = 0

        # Type/stage classification
        d = metrics.get('maxDiameterMm', 0)
        if not metrics['detected']:
            tumor_type, stage, stage_sub = 'Không phát hiện', 'N/A', ''
        elif d <= 6:
            tumor_type, stage, stage_sub = 'Nốt phổi nhỏ', 'T1mi', '(≤ 6mm)'
        elif d <= 10:
            tumor_type, stage, stage_sub = 'Nốt phổi', 'T1a', '(6 - 10mm)'
        elif d <= 20:
            tumor_type, stage, stage_sub = 'Nghi ngờ ác tính', 'T1b', '(1 - 2 cm)'
        elif d <= 30:
            tumor_type, stage, stage_sub = 'Nghi ngờ ác tính', 'T1c', '(2 - 3 cm)'
        else:
            tumor_type, stage, stage_sub = 'Khối u lớn', 'T2+', '(> 3 cm)'

        # Create overlay images
        prob_map = (consensus_mask.astype(np.float32) / 255.0)
        prob_map_resized = cv2.resize(prob_map, (LUNG_INPUT_SIZE, LUNG_INPUT_SIZE))
        orig_resized = orig_pil.resize((LUNG_INPUT_SIZE, LUNG_INPUT_SIZE))
        overlay_b64 = create_overlay_image(orig_resized, mask_resized, prob_map_resized, orig_size=(orig_w, orig_h))
        mask_b64 = create_mask_image_b64(mask_resized)

        # Original as base64
        buf_orig = io.BytesIO()
        orig_resized.save(buf_orig, format='PNG')
        orig_b64 = f"data:image/png;base64,{base64.b64encode(buf_orig.getvalue()).decode('utf-8')}"

        # Heatmap — color by agreement level
        agreement_vis = cv2.resize(mask_sum.astype(np.float32), (LUNG_INPUT_SIZE, LUNG_INPUT_SIZE))
        agreement_norm = (agreement_vis / max(num_annotators, 1) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(agreement_norm, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        buf_heat = io.BytesIO()
        Image.fromarray(heatmap_rgb).save(buf_heat, format='PNG')
        heatmap_b64 = f"data:image/png;base64,{base64.b64encode(buf_heat.getvalue()).decode('utf-8')}"

        # MIP (Maximum Intensity Projection) views with consensus-mask overlay.
        # Tumor centroid is only used to crop the window — projection itself is
        # over the full volume axis so the tumor stands out as a bright blob.
        if metrics['detected'] and metrics.get('tumors'):
            t0 = metrics['tumors'][0]
            mpr_cx = int(t0['centroid']['x'])
            mpr_cy = int(t0['centroid']['y'])
        else:
            mpr_cx, mpr_cy = orig_w // 2, orig_h // 2
        try:
            mask_vol = build_consensus_mask_volume(
                nod_path, slices, consensus_threshold, orig_w, orig_h)
            coronal_b64, sagittal_b64 = compute_mpr_views(
                img_dir, slices, mpr_cx, mpr_cy, orig_w, orig_h,
                mask_volume=mask_vol)
        except Exception as mpr_err:
            print(f"[WARN] MPR compute failed: {mpr_err}")
            coronal_b64, sagittal_b64 = None, None

        # NB: An earlier version of this function rewrote position text and
        # centroid based on a hash of patient_id. That made every patient
        # show a different lobe/side, but it diverged from where the user
        # actually saw the tumor on the displayed slice. Position is now
        # derived purely from the real mask centroid in `analyze_lung_mask`,
        # so the text always matches the visual location.

        result = {
            'success': True,
            'detected': metrics['detected'],
            'tumorCount': metrics.get('tumorCount', 0),
            'tumors': metrics.get('tumors', []),
            'maxDiameterMm': metrics.get('maxDiameterMm', 0),
            'totalAreaMm2': metrics.get('totalAreaMm2', 0),
            'volumeMm3': metrics.get('volumeMm3', 0),
            'volumeCm3': round(metrics.get('volumeMm3', 0) / 1000, 2),
            'position': metrics.get('position', 'N/A'),
            'positionSub': metrics.get('positionSub', ''),
            'confidence': round(overall_confidence, 1),
            'maxConfidence': round(overall_confidence, 1),
            'malignancy': round(malignancy, 1),
            'tumorType': tumor_type,
            'stage': stage,
            'stageSub': stage_sub,
            'overlayImage': overlay_b64,
            'maskImage': mask_b64,
            'originalImage': orig_b64,
            'heatmapImage': heatmap_b64,
            'coronalImage': coronal_b64,
            'sagittalImage': sagittal_b64,
            'originalSize': {'w': orig_w, 'h': orig_h},
            'datasetInfo': {
                'patientId': patient_id,
                'nodule': nodule_name,
                'slice': slice_name,
                'sliceIndex': slice_idx,
                'totalSlices': len(slices),
                'numAnnotators': num_annotators,
                'consensusThreshold': consensus_threshold,
                'source': 'LIDC-IDRI Ground Truth (Consensus Mask)',
            },
            'modelInfo': {
                'name': 'Ground Truth (Radiologist Consensus)',
                'type': f'{num_annotators}-Annotator Majority Vote',
                'dataset': 'LIDC-IDRI',
                'inputSize': LUNG_INPUT_SIZE,
            }
        }

        print(f"[DATASET] {patient_id}/{nodule_name}/{slice_name} -> "
              f"detected={metrics['detected']}, d={d:.1f}mm, "
              f"annotators={num_annotators}, conf={overall_confidence:.1f}%")

        return jsonify(result)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            print(f"[ERROR] /api/predict-lung-dataset: {e}")
            print(tb)
        except UnicodeEncodeError:
            print("[ERROR] /api/predict-lung-dataset (encoding error in traceback)")
        return jsonify({'error': str(e), 'trace': tb}), 500


@app.route('/api/lung-dataset-image', methods=['GET'])
def lung_dataset_image():
    """Serve a slice image from the dataset for preview."""
    patient = request.args.get('patient', '')
    nodule = request.args.get('nodule', 'nodule-0')
    slice_idx = int(request.args.get('slice', 0))
    nod_path = os.path.join(DATASET_DIR, patient, nodule, 'images')
    if not os.path.exists(nod_path):
        return 'Not found', 404
    slices = sorted(os.listdir(nod_path))
    if slice_idx >= len(slices):
        return 'Slice not found', 404
    from flask import send_file
    return send_file(os.path.join(nod_path, slices[slice_idx]), mimetype='image/png')


# ===== BRAIN TUMOR SEGMENTATION (3D U-Net, Keras BraTS) ===========
# Lazy-load TF + the .keras model on first request to avoid slowing
# Flask startup when the brain endpoint isn't being used.
# Brain segmentation model. Set env var BRAIN_MODEL_FILE to switch:
#   BRAIN_MODEL_FILE=brats_3d_unet_kaggle_best.keras                      (default, OncoSense, 68MB)
#   BRAIN_MODEL_FILE=model.keras                                          (the 372MB one)
#   BRAIN_MODEL_FILE=models_3D/3D-UNet-2018-weights-improvement-08-0.994.hdf5  (best epoch of 2018 set)
_brain_model_name = os.environ.get('BRAIN_MODEL_FILE', 'brats_3d_unet_kaggle_best.keras')
BRAIN_MODEL_PATH = os.path.join(MODEL_DIR, 'brain', _brain_model_name)
BRAIN_FALLBACK_PATH = os.path.join(MODEL_DIR, 'brain', 'brats_3d_unet_kaggle_best.keras')
BRAIN_INPUT_SHAPE = (128, 128, 128, 4)
brain_model = None
brain_model_loaded = False
brain_model_error = None


def _build_2018_3d_unet(input_shape=(128, 128, 128, 4), num_classes=4):
    """Recreate the 2018 BraTS 3D U-Net architecture from Bhuvan Aggarwal's
    Kaggle notebook, used for `3D-UNet-2018-weights-improvement-XX-Y.YYY.hdf5`
    checkpoints. Those .hdf5 files are WEIGHTS-ONLY (model.save_weights output)
    so we have to build the same graph before load_weights() can populate it.

    Reference: 4-level encoder/decoder, 32→64→128→256 features, instance norm
    + LeakyReLU, last activation softmax. Matches Bhuvan A. 2021 BraTS21
    notebook architecture.
    """
    import tensorflow as tf
    from tensorflow.keras import layers as L, models as M

    def conv_block(x, n, drop=0.1):
        x = L.Conv3D(n, 3, padding='same', kernel_initializer='he_normal')(x)
        x = L.Activation('relu')(x)
        x = L.Dropout(drop)(x)
        x = L.Conv3D(n, 3, padding='same', kernel_initializer='he_normal')(x)
        x = L.Activation('relu')(x)
        return x

    inp = L.Input(shape=input_shape)
    c1 = conv_block(inp, 32);   p1 = L.MaxPooling3D(2)(c1)
    c2 = conv_block(p1,  64);   p2 = L.MaxPooling3D(2)(c2)
    c3 = conv_block(p2,  128);  p3 = L.MaxPooling3D(2)(c3)
    c4 = conv_block(p3,  256);  p4 = L.MaxPooling3D(2)(c4)
    c5 = conv_block(p4,  512, drop=0.2)
    u6 = L.Conv3DTranspose(256, 2, strides=2, padding='same')(c5)
    u6 = L.concatenate([u6, c4]); c6 = conv_block(u6, 256)
    u7 = L.Conv3DTranspose(128, 2, strides=2, padding='same')(c6)
    u7 = L.concatenate([u7, c3]); c7 = conv_block(u7, 128)
    u8 = L.Conv3DTranspose(64, 2, strides=2, padding='same')(c7)
    u8 = L.concatenate([u8, c2]); c8 = conv_block(u8, 64)
    u9 = L.Conv3DTranspose(32, 2, strides=2, padding='same')(c8)
    u9 = L.concatenate([u9, c1]); c9 = conv_block(u9, 32)
    out = L.Conv3D(num_classes, 1, activation='softmax')(c9)
    return M.Model(inp, out)


def _try_load_one(path):
    """Attempt to load a single model file. Returns the model on success, raises on failure.
    Handles both full-model files and weights-only checkpoints from the 2018 notebook set.
    """
    import tensorflow as tf
    # Try full-model load first (works for .keras files + some .hdf5 with full graph)
    try:
        m = tf.keras.models.load_model(path, compile=False)
        print(f'[Brain] Loaded as full model: {path}')
        return m
    except Exception as e:
        print(f'[Brain] Full-model load failed ({e.__class__.__name__}: {e}); '
              f'trying weights-only path for {os.path.basename(path)}...')
    # Weights-only fallback: rebuild the 2018 3D U-Net architecture + load_weights
    m = _build_2018_3d_unet(BRAIN_INPUT_SHAPE, num_classes=4)
    m.load_weights(path)
    print(f'[Brain] Loaded as weights into rebuilt 2018 3D U-Net: {path}')
    return m


def _load_brain_model():
    global brain_model, brain_model_loaded, brain_model_error
    if brain_model_loaded or brain_model_error:
        return
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

    paths_to_try = [BRAIN_MODEL_PATH]
    # Always fall back to the known-good kaggle model if the env-var path fails
    if BRAIN_MODEL_PATH != BRAIN_FALLBACK_PATH:
        paths_to_try.append(BRAIN_FALLBACK_PATH)

    last_error = None
    for p in paths_to_try:
        if not os.path.exists(p):
            print(f'[Brain] Skipping (not found): {p}')
            continue
        try:
            brain_model = _try_load_one(p)
            brain_model_loaded = True
            print(f'[OK] Brain 3D U-Net loaded from {p}')
            print(f'     input  shape: {brain_model.input_shape}')
            print(f'     output shape: {brain_model.output_shape}')
            print(f'     params:       {brain_model.count_params():,}')
            return
        except Exception as e:
            last_error = e
            print(f'[ERROR] Failed loading {p}: {e}')
            continue
    brain_model_error = str(last_error) if last_error else 'No brain model could be loaded'
    print(f'[ERROR] All brain model load attempts failed: {brain_model_error}')


def _zscore(arr):
    """Per-volume z-score on brain voxels only; background reset to 0.

    This is the standard BraTS preprocessing — applying z-score over the
    whole volume (including the zero background) would shift the brain to
    a weird intensity range. We mask, normalize, then zero out background.
    """
    arr = arr.astype(np.float32)
    mask = arr > 0
    if mask.sum() == 0:
        return arr
    mean = arr[mask].mean()
    std  = arr[mask].std()
    if std < 1e-6:
        std = 1.0
    arr = (arr - mean) / std
    arr[~mask] = 0           # ← critical: reset background AFTER normalize
    return arr


def _crop_or_pad_3d(vol, target=128):
    """Center-crop or zero-pad a 3D volume to (target, target, target).
    Legacy — still used for the GT seg file. For MRI modalities, prefer
    `_brain_bbox_resize` which preserves the entire brain.
    """
    out = np.zeros((target, target, target), dtype=vol.dtype)
    src_slices, dst_slices = [], []
    for i in range(3):
        s_in = vol.shape[i]
        if s_in >= target:
            start = (s_in - target) // 2
            src_slices.append(slice(start, start + target))
            dst_slices.append(slice(0, target))
        else:
            start = (target - s_in) // 2
            src_slices.append(slice(0, s_in))
            dst_slices.append(slice(start, start + s_in))
    out[tuple(dst_slices)] = vol[tuple(src_slices)]
    return out


# Module-level cache so consecutive modality crops use the SAME brain bbox
# (computed from whichever modality is loaded first — typically FLAIR).
# Reset on every new request via _reset_brain_bbox_cache().
_BRAIN_BBOX_CACHE = {'bbox': None, 'orig_shape': None, 'voxel_mm3_resized': None}

def _reset_brain_bbox_cache():
    _BRAIN_BBOX_CACHE['bbox'] = None
    _BRAIN_BBOX_CACHE['orig_shape'] = None
    _BRAIN_BBOX_CACHE['voxel_mm3_resized'] = None


def _gaussian_window_3d(size=128, sigma_scale=0.125):
    """3D Gaussian importance weight for sliding-window inference blending.

    Standard nnU-Net / MONAI technique: when 2 patches overlap, the
    voxels near the patch CENTER (where context is best) should
    contribute more than the voxels near the patch EDGE (where the
    receptive field is one-sided and predictions are less reliable).
    """
    sigma = size * sigma_scale
    coords = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    g1d = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g3d = g1d[:, None, None] * g1d[None, :, None] * g1d[None, None, :]
    g3d = g3d / g3d.max()
    g3d = np.maximum(g3d, 1e-3)        # min weight at corners (avoid 0/0)
    return g3d.astype(np.float32)


def _sliding_window_inference(x_full, model, window=128, overlap=0.5,
                               num_classes=4, use_tta=True,
                               progress_cb=None):
    """Sliding-window inference at native resolution + optional 4-way TTA.

    Pipeline:
      1. Slide a `window`³ patch across the full volume with `overlap`
         stride. Each patch goes through the model (fully convolutional →
         accepts any size if shape matches checkpoint).
      2. For each patch, run 4 forward passes with axis flips, average
         the softmax probabilities (un-flipped). This is Test-Time
         Augmentation — typically +2-3% Dice for free.
      3. Blend overlapping patches with a 3D Gaussian weight so the
         seam between patches is smooth.

    Returns the final softmax probability map at the input's native
    resolution: shape (D, H, W, num_classes).

    Args:
      x_full:    (1, D, H, W, 4) input — z-score normalized 4-channel MRI
      model:     Keras 3D U-Net (input 128³, output 128³ × num_classes)
      window:    Patch side length (must match model input)
      overlap:   Fraction of overlap between adjacent patches (0.5 = 50%)
      use_tta:   If True, do 4 flip augmentations per patch
      progress_cb: Optional callable(done, total) for progress reporting
    """
    _, D, H, W, C = x_full.shape
    stride = max(1, int(window * (1.0 - overlap)))

    # Pad volume up to at least one full window per axis
    pad_d = max(0, window - D)
    pad_h = max(0, window - H)
    pad_w = max(0, window - W)
    if pad_d + pad_h + pad_w > 0:
        x_pad = np.pad(x_full,
                       ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w), (0, 0)),
                       mode='constant', constant_values=0)
    else:
        x_pad = x_full
    Dp, Hp, Wp = x_pad.shape[1], x_pad.shape[2], x_pad.shape[3]

    def _starts(total, win, st):
        if total <= win:
            return [0]
        s = list(range(0, total - win + 1, st))
        if s[-1] + win < total:
            s.append(total - win)
        return s
    zs = _starts(Dp, window, stride)
    ys = _starts(Hp, window, stride)
    xs = _starts(Wp, window, stride)

    n_windows = len(zs) * len(ys) * len(xs)
    flips = [(), (1,), (2,), (3,)] if use_tta else [()]  # 4-way TTA on D/H/W
    total_passes = n_windows * len(flips)
    print(f'[Brain] Sliding window: {n_windows} windows '
          f'({len(zs)}×{len(ys)}×{len(xs)}), window={window}, stride={stride}, '
          f'TTA={"on (4-way)" if use_tta else "off"}, '
          f'total forward passes={total_passes}', flush=True)

    out_probs = np.zeros((Dp, Hp, Wp, num_classes), dtype=np.float32)
    out_count = np.zeros((Dp, Hp, Wp), dtype=np.float32)
    gauss_w = _gaussian_window_3d(window)

    done = 0
    win_idx = 0
    for z in zs:
        for y in ys:
            for x in xs:
                win_idx += 1
                patch = x_pad[0, z:z+window, y:y+window, x:x+window, :]   # (D,H,W,C)
                # Build a single BATCH of flipped patches → 1 predict() call
                # instead of len(flips) calls. Cuts TF overhead drastically.
                # Axes in 4D patch are (D=0, H=1, W=2, C=3) so flip_axes from
                # the original (B,D,H,W,C) need offset -1.
                batch = []
                for fa in flips:
                    p = patch
                    if fa:
                        p = np.flip(p, axis=tuple(a - 1 for a in fa))
                    batch.append(p)
                batch_in = np.stack(batch, axis=0).astype(np.float32)
                # One forward pass for all TTA variants of this window
                batch_logits = model.predict(batch_in, verbose=0,
                                              batch_size=len(flips))
                tta_probs = np.zeros((window, window, window, num_classes),
                                      dtype=np.float32)
                for i, fa in enumerate(flips):
                    logits_i = batch_logits[i]
                    if fa:
                        logits_i = np.flip(logits_i, axis=tuple(a - 1 for a in fa))
                    e = np.exp(logits_i - logits_i.max(axis=-1, keepdims=True))
                    tta_probs += e / e.sum(axis=-1, keepdims=True)
                    done += 1
                tta_probs /= float(len(flips))
                if progress_cb:
                    try: progress_cb(done, total_passes)
                    except Exception: pass
                print(f'[Brain]   window {win_idx}/{n_windows} done '
                      f'({done}/{total_passes} TTA passes)', flush=True)

                # Gaussian-weighted blend into the accumulator
                out_probs[z:z+window, y:y+window, x:x+window] += (
                    tta_probs * gauss_w[..., None]
                )
                out_count[z:z+window, y:y+window, x:x+window] += gauss_w

    # Normalize and crop back to original native shape
    out_count = np.maximum(out_count, 1e-6)
    out_probs = out_probs / out_count[..., None]
    out_probs = out_probs[:D, :H, :W]
    return out_probs


def _predict_with_tta(x, model, num_classes=4):
    """Dispatcher: detect 3D vs 2D model from input shape, route accordingly.

    Returns probability map at our 128³ space, shape (D, H, W, num_classes).
    For 2D models, slices are predicted one-by-one and stacked.
    """
    in_shape = tuple(model.input_shape)
    out_shape = tuple(model.output_shape)

    # 3D segmentation model: (None, D, H, W, C) → (None, D, H, W, K)
    if len(in_shape) == 5 and len(out_shape) == 5:
        return _predict_3d_tta(x, model, num_classes)

    # 2D segmentation model: (None, H, W, C) → (None, H, W, K)
    if len(in_shape) == 4 and len(out_shape) == 4:
        return _predict_2d_per_slice_tta(x, model, num_classes)

    raise ValueError(
        f'Model not compatible with segmentation pipeline. '
        f'Input shape {in_shape}, output shape {out_shape}. '
        f'Expected 3D U-Net (None,D,H,W,C)→(None,D,H,W,K) or '
        f'2D U-Net (None,H,W,C)→(None,H,W,K).'
    )


def _predict_3d_tta(x, model, num_classes=4):
    """3D U-Net inference + 4-way TTA on (D,H,W) spatial axes.
    Single forward pass per flip, batched as 4-element batch."""
    if x.ndim == 5:
        patch = x[0]
    else:
        patch = x
    flips = [(), (0,), (1,), (2,)]
    batch = []
    for fa in flips:
        p = patch
        if fa:
            p = np.flip(p, axis=fa)
        batch.append(p)
    batch_in = np.stack(batch, axis=0).astype(np.float32)
    batch_logits = model.predict(batch_in, verbose=0, batch_size=len(flips))
    K = batch_logits.shape[-1]
    out = np.zeros(patch.shape[:3] + (K,), dtype=np.float32)
    for i, fa in enumerate(flips):
        l = batch_logits[i]
        if fa:
            l = np.flip(l, axis=fa)
        # Some models already softmax; detect by range
        if l.min() < -1e-3 or l.max() > 1.0 + 1e-3:
            e = np.exp(l - l.max(axis=-1, keepdims=True))
            l = e / e.sum(axis=-1, keepdims=True)
        out += l
    out = out / float(len(flips))
    # If output has fewer classes than requested, pad with zeros
    if out.shape[-1] < num_classes:
        pad = np.zeros(out.shape[:3] + (num_classes - out.shape[-1],), dtype=np.float32)
        out = np.concatenate([out, pad], axis=-1)
    return out


def _predict_2d_per_slice_tta(x, model, num_classes=4):
    """2D U-Net inference: process each axial slice independently, build 3D
    probability volume from per-slice predictions. Auto-resizes to the
    model's expected (H, W) and picks the right channels.

    The 2018 BraTS 2D U-Net checkpoints expect (128, 128, 2) input — the
    convention is [FLAIR, T1ce] (the two most discriminative modalities
    for tumor segmentation). Other channel counts are auto-mapped from our
    [FLAIR, T1, T1ce, T2] = channels [0, 1, 2, 3] stack.
    """
    from scipy.ndimage import zoom
    in_shape = tuple(model.input_shape)
    out_shape = tuple(model.output_shape)
    target_h = in_shape[1] or 128
    target_w = in_shape[2] or 128
    target_c = in_shape[3] or 4
    out_classes = out_shape[3] or num_classes

    # Map our 4 channels → model's expected channel count
    # Channel order in our stack: 0=FLAIR, 1=T1, 2=T1ce, 3=T2
    channel_maps = {
        1: [0],            # FLAIR only
        2: [0, 2],         # FLAIR + T1ce (BraTS 2D U-Net convention)
        3: [0, 2, 3],      # FLAIR + T1ce + T2 (skip T1, less discriminative)
        4: [0, 1, 2, 3],   # all
    }
    channels = channel_maps.get(target_c, list(range(min(target_c, 4))))

    if x.ndim == 5:
        vol = x[0]
    else:
        vol = x
    D, H, W, C = vol.shape
    out_3d = np.zeros((D, H, W, max(num_classes, out_classes)), dtype=np.float32)

    print(f'[Brain] 2D per-slice inference: model expects ({target_h},{target_w},{target_c}) '
          f'→ ({out_classes} classes). Mapping channels {channels} from our 4-stack.', flush=True)

    flips_2d = [(), (0,), (1,), (0, 1)]   # H/W flips (axis 0 and 1 in 3D-slice)

    BATCH_SIZE = 32   # batch many slices to amortize TF overhead
    slice_inputs = []
    slice_idx = []

    def _flush_batch():
        if not slice_inputs:
            return
        batch_in = np.stack(slice_inputs, axis=0).astype(np.float32)
        batch_logits = model.predict(batch_in, verbose=0,
                                      batch_size=min(BATCH_SIZE * 4, len(slice_inputs)))
        # Each slice has 4 TTA entries in batch — reduce
        for slot, z in enumerate(slice_idx):
            base = slot * len(flips_2d)
            probs_z = np.zeros((target_h, target_w, batch_logits.shape[-1]),
                                dtype=np.float32)
            for i, fa in enumerate(flips_2d):
                l = batch_logits[base + i]
                if fa:
                    l = np.flip(l, axis=fa)
                if l.min() < -1e-3 or l.max() > 1.0 + 1e-3:
                    e = np.exp(l - l.max(axis=-1, keepdims=True))
                    l = e / e.sum(axis=-1, keepdims=True)
                probs_z += l
            probs_z = probs_z / float(len(flips_2d))
            # Resize back if model H/W != our H/W
            if (target_h, target_w) != (H, W):
                fz = (H / target_h, W / target_w, 1.0)
                probs_z = zoom(probs_z, fz, order=1, mode='constant', cval=0)
                # Trim or pad to exact (H, W)
                trimmed = np.zeros((H, W, probs_z.shape[-1]), dtype=np.float32)
                sh = min(H, probs_z.shape[0])
                sw = min(W, probs_z.shape[1])
                trimmed[:sh, :sw] = probs_z[:sh, :sw]
                probs_z = trimmed
            # Write into output (broadcast across class count diff)
            kc = min(out_3d.shape[-1], probs_z.shape[-1])
            out_3d[z, ..., :kc] = probs_z[..., :kc]
        slice_inputs.clear()
        slice_idx.clear()

    for z in range(D):
        slc = vol[z, :, :, :][:, :, channels]   # (H, W, target_c)
        if (target_h, target_w) != (H, W):
            fz = (target_h / H, target_w / W, 1.0)
            slc = zoom(slc, fz, order=1, mode='constant', cval=0)
            # Trim/pad to exact (target_h, target_w)
            trimmed = np.zeros((target_h, target_w, target_c), dtype=np.float32)
            sh = min(target_h, slc.shape[0])
            sw = min(target_w, slc.shape[1])
            trimmed[:sh, :sw] = slc[:sh, :sw]
            slc = trimmed
        # Generate 4 TTA flips of this slice
        for fa in flips_2d:
            p = slc
            if fa:
                p = np.flip(p, axis=fa)
            slice_inputs.append(p)
        slice_idx.append(z)
        if len(slice_inputs) >= BATCH_SIZE * len(flips_2d):
            _flush_batch()
    _flush_batch()
    print(f'[Brain] 2D per-slice inference complete — output shape {out_3d.shape}', flush=True)

    # If output classes < num_classes, pad
    if out_3d.shape[-1] < num_classes:
        pad = np.zeros(out_3d.shape[:3] + (num_classes - out_3d.shape[-1],), dtype=np.float32)
        out_3d = np.concatenate([out_3d, pad], axis=-1)
    return out_3d[..., :num_classes]


def _build_native_brain_input(files_dict):
    """Z-score normalize the 4 modalities at NATIVE RESOLUTION.

    Unlike `_build_brain_input_from_files` (which downsamples to 128³ via
    bbox-crop), this returns the full native volume so sliding-window
    inference can preserve every voxel of detail.
    Returns (x_native, raw_modalities, affine, orig_shape, voxel_mm3).
    """
    import nibabel as nib
    keys = ['flair', 't1', 't1ce', 't2']
    loaded, loaded_raw = {}, {}
    orig_shape = None
    orig_affine = np.eye(4)
    ref_shape = None
    for k in keys:
        p = files_dict.get(k)
        if not p or not os.path.exists(p):
            continue
        nii = nib.load(p)
        v = np.asarray(nii.get_fdata()).astype(np.float32)
        if v.ndim == 4:
            v = v[..., 0]
        if orig_shape is None:
            orig_shape = v.shape
            try:
                orig_affine = np.asarray(nii.affine, dtype=np.float64)
            except Exception:
                pass
            ref_shape = v.shape
        # Reject seg-label uploads (same defensive check as before)
        peak = float(np.max(v))
        if peak <= 10 and (v > 0).any():
            pos_min = float(np.min(v[v > 0]))
            if pos_min >= 1:
                raise ValueError(
                    f'File assigned to "{k}" looks like a segmentation label '
                    f'(intensity range 0-{peak:.0f}), not an MRI modality.'
                )
        # Spatial pad/crop to ref_shape (handle mismatched modality sizes)
        if v.shape != ref_shape:
            v = _crop_or_pad_3d(v, target=max(ref_shape))[:ref_shape[0], :ref_shape[1], :ref_shape[2]]
        loaded_raw[k] = v.copy()
        loaded[k] = _zscore(v)
        print(f'[Brain] {k} (native): shape {v.shape}, intensity {v.min():.1f}-{v.max():.1f}, '
              f'norm {loaded[k].min():.2f}-{loaded[k].max():.2f}', flush=True)
    if not loaded:
        raise ValueError('No NIfTI files loaded')
    # Replicate missing modalities from the first available one
    fallback = next(iter(loaded.values()))
    channels = [loaded.get(k, fallback) for k in keys]
    vol = np.stack(channels, axis=-1)          # (D, H, W, 4)
    # Per-voxel volume from affine det (BraTS native = 1mm³)
    try:
        voxel_mm3 = float(abs(np.linalg.det(orig_affine[:3, :3])))
        if not (1e-6 < voxel_mm3 < 50.0):
            voxel_mm3 = 1.0
    except Exception:
        voxel_mm3 = 1.0
    print(f'[Brain] Native input stacked: shape {vol.shape}, '
          f'voxel volume {voxel_mm3:.3f} mm³', flush=True)
    return vol[None, ...].astype(np.float32), loaded_raw, orig_affine, orig_shape, voxel_mm3


def _brain_bbox_resize(vol, target=128, pad=4, brain_thresh=0.01,
                       affine=None, use_cache=True):
    """Crop to brain bounding box → resize to (target,target,target).

    This is the correct BraTS preprocessing for our 128³ U-Net. Why:

      - Native BraTS shape is 240×240×155 but the brain typically fills
        only ~180×180×130. Center-cropping 240→128 removes 56 px each
        side of XY and 14/13 of Z, SLICING INTO the brain (occipital,
        frontal cortex get cut off).
      - Naive zoom 240→128 wastes 1/3 of resolution on background air.

    The bbox-crop approach finds the brain mask (signal > 1% peak),
    pads by `pad` voxels, then resizes that brain-only volume to 128³.
    No brain tissue is cut. Resolution per voxel becomes
        new_voxel = (brain_extent_mm / 128) per axis
    typically ~1.5 mm³/voxel for a 190mm brain → much better than the
    ~2.8 mm³/voxel of a naive full-volume zoom.

    Returns (resized_vol, bbox_tuple, voxel_mm3_resized).
    """
    from scipy.ndimage import zoom

    # Reuse the first modality's bbox so all 4 channels align spatially
    cached = _BRAIN_BBOX_CACHE['bbox'] if use_cache else None

    if vol.max() <= 0:
        out = np.zeros((target, target, target), dtype=np.float32)
        return out, (0, 1, 0, 1, 0, 1), 1.0

    if cached is None:
        brain_mask = vol > (brain_thresh * float(vol.max()))
        if not brain_mask.any():
            # All-background or weird volume — fallback to center crop
            return _crop_or_pad_3d(vol, target), (0, vol.shape[0], 0, vol.shape[1], 0, vol.shape[2]), 1.0
        zs, ys, xs = np.where(brain_mask)
        z_min = max(0, int(zs.min()) - pad)
        z_max = min(vol.shape[0], int(zs.max()) + pad + 1)
        y_min = max(0, int(ys.min()) - pad)
        y_max = min(vol.shape[1], int(ys.max()) + pad + 1)
        x_min = max(0, int(xs.min()) - pad)
        x_max = min(vol.shape[2], int(xs.max()) + pad + 1)
        bbox = (z_min, z_max, y_min, y_max, x_min, x_max)
        if use_cache:
            _BRAIN_BBOX_CACHE['bbox'] = bbox
            _BRAIN_BBOX_CACHE['orig_shape'] = vol.shape
    else:
        bbox = cached
    z_min, z_max, y_min, y_max, x_min, x_max = bbox

    cropped = vol[z_min:z_max, y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        return _crop_or_pad_3d(vol, target), bbox, 1.0

    factors = (target / float(cropped.shape[0]),
               target / float(cropped.shape[1]),
               target / float(cropped.shape[2]))
    resized = zoom(cropped, factors, order=1, mode='constant', cval=0)
    out = np.zeros((target, target, target), dtype=np.float32)
    s = [min(target, resized.shape[i]) for i in range(3)]
    out[:s[0], :s[1], :s[2]] = resized[:s[0], :s[1], :s[2]].astype(np.float32)

    # Compute per-voxel volume in the RESIZED space (mm³). Original BraTS
    # is 1mm³ per native voxel; each resized voxel covers
    #   (extent_z * extent_y * extent_x) / target³ native voxels
    orig_voxel_mm3 = 1.0
    if affine is not None:
        try:
            orig_voxel_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
            if orig_voxel_mm3 <= 0 or orig_voxel_mm3 > 50:
                orig_voxel_mm3 = 1.0
        except Exception:
            pass
    bbox_voxels = (z_max - z_min) * (y_max - y_min) * (x_max - x_min)
    voxel_mm3_resized = (bbox_voxels * orig_voxel_mm3) / float(target ** 3)
    if use_cache:
        _BRAIN_BBOX_CACHE['voxel_mm3_resized'] = voxel_mm3_resized

    return out, bbox, voxel_mm3_resized


def _build_brain_input_from_files(files_dict):
    """Build the model's (1, 128, 128, 128, 4) input tensor.

    `files_dict`: { 'flair': path, 't1': path, 't1ce': path, 't2': path }
    Any subset is accepted; missing modalities are replicated from the
    first one provided so the model still gets a 4-channel input.
    """
    import nibabel as nib
    keys = ['flair', 't1', 't1ce', 't2']
    loaded = {}                # z-score normalized (for model)
    loaded_raw = {}            # raw intensity (for visualization)
    orig_shape = None
    orig_affine = np.eye(4)
    # Reset bbox cache so each request computes its own brain bbox
    _reset_brain_bbox_cache()
    voxel_mm3_resized = 1.0
    bbox_used = None
    for k in keys:
        p = files_dict.get(k)
        if not p or not os.path.exists(p):
            continue
        nii = nib.load(p)
        v = np.asarray(nii.get_fdata()).astype(np.float32)
        if v.ndim == 4:
            v = v[..., 0]
        if orig_shape is None:
            orig_shape = v.shape
            try:
                orig_affine = np.asarray(nii.affine, dtype=np.float64)
            except Exception:
                pass
        # Brain-bbox crop + resize to 128³. First call (typically FLAIR)
        # computes the bbox; subsequent modalities reuse it so all 4
        # channels are spatially aligned.
        v_raw_crop, bbox_used, voxel_mm3_resized = _brain_bbox_resize(
            v, target=128, affine=orig_affine, use_cache=True,
        )
        loaded_raw[k] = v_raw_crop
        v_norm = _zscore(v)
        v_crop, _, _ = _brain_bbox_resize(
            v_norm, target=128, affine=orig_affine, use_cache=True,
        )
        print(f'[Brain] {k}: orig {v.shape}, intensity {v.min():.1f}-{v.max():.1f}, '
              f'norm {v_norm.min():.2f}-{v_norm.max():.2f}, '
              f'after bbox-resize {v_crop.shape}, non-zero {(v_crop != 0).sum()}', flush=True)
        loaded[k] = v_crop
    if bbox_used is not None:
        z0, z1, y0, y1, x0, x1 = bbox_used
        print(f'[Brain] Brain bbox: z[{z0}:{z1}] y[{y0}:{y1}] x[{x0}:{x1}] '
              f'→ extent {z1-z0}×{y1-y0}×{x1-x0} voxels '
              f'→ resized voxel = {voxel_mm3_resized:.3f} mm³', flush=True)

    if not loaded:
        raise ValueError('No NIfTI files loaded')

    # Reject obvious mis-uploads: BraTS *_seg.nii.gz has intensity range 0-4
    # (class labels). Use a defensive multi-step check — combining the
    # conditions into one boolean expression can confuse numpy's __bool__
    # if any sub-expression unexpectedly evaluates to an array.
    for k, v in list(loaded_raw.items()):
        peak = float(np.max(v))
        if peak > 10:
            continue            # real MRI intensity, not seg labels
        pos_vals = v[v > 0]
        if int(pos_vals.size) == 0:
            continue            # all-zero volume, skip
        min_pos = float(np.min(pos_vals))
        if min_pos < 1:
            continue            # has fractional values → not seg labels
        # Reaching here: peak <= 10 AND min positive value >= 1 → integer labels
        raise ValueError(
            f'File assigned to "{k}" looks like a segmentation label '
            f'(intensity range 0-{peak:.0f}), not an MRI modality. '
            f'Upload the 4 modality files: *_flair, *_t1, *_t1ce, *_t2 '
            f'(NOT *_seg.nii.gz which is the ground-truth mask).'
        )

    # Fill missing channels by replicating the first available modality
    fallback = next(iter(loaded.values()))
    channels = [loaded.get(k, fallback) for k in keys]
    vol = np.stack(channels, axis=-1)         # (128,128,128,4)
    print(f'[Brain] Stacked input: {vol.shape}, dtype {vol.dtype}, '
          f'range {vol.min():.2f}-{vol.max():.2f}', flush=True)
    # Return ALL raw modalities dict so we can render the pipeline preview
    # (FLAIR / T1 / T1ce / T2 grid) — callers that only need FLAIR can do
    # raw_modalities.get('flair') ?? first.
    return vol[None, ...].astype(np.float32), orig_shape, loaded_raw, orig_affine


def _build_brain_input_from_image(img_pil):
    """Demo fallback: 1 image -> replicate to 4 channels + fake 3D volume."""
    img = img_pil.convert('L').resize((128, 128))
    slice_raw_2d = np.array(img).astype(np.float32)
    slice_2d = _zscore(slice_raw_2d)
    vol_3d = np.stack([slice_2d] * 128, axis=0)
    vol = np.stack([vol_3d] * 4, axis=-1)
    raw_3d = np.stack([slice_raw_2d] * 128, axis=0)
    # Wrap into the same dict shape as the NIfTI branch so callers don't care
    raw_modalities = {'flair': raw_3d, 't1': raw_3d, 't1ce': raw_3d, 't2': raw_3d}
    return vol[None, ...].astype(np.float32), (128, 128), raw_modalities, np.eye(4)


# Discrete BraTS palette — uses matplotlib's "Paired" tab10-style colors that
# the Kaggle BraTS preprocessing notebooks use, so users can visually
# cross-reference with the reference implementations they're already
# familiar with. Index 0 = label 1 (NCR), etc.
def _brats_listed_cmap():
    from matplotlib.colors import ListedColormap
    return ListedColormap([
        '#1f77b4',   # NCR — Paired blue
        '#a6cee3',   # ED  — Paired light blue
        '#ffbf00',   # ET  — Paired amber
    ])


def _render_brats_multiview(raw_flair_3d, seg_3d, affine, centroid_xyz=None):
    """Render the classic BraTS plot_roi style — 3 orthogonal views
    (sagittal / coronal / axial) of the FLAIR with the multi-class tumor mask
    overlaid in DISTINCT colors per class (NCR red / ED yellow / ET magenta).

    Falls back to a single axial slice if nilearn / matplotlib aren't
    available. Returns a base64 PNG data-URI.
    """
    try:
        import nibabel as nib
        from nilearn import plotting
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
    except Exception as e:
        print('[Brain] nilearn unavailable, falling back:', e, flush=True)
        return _render_axial_only(raw_flair_3d, seg_3d > 0,
                                  centroid_xyz[2] if centroid_xyz is not None else seg_3d.shape[2] // 2)

    flair_img = nib.Nifti1Image(raw_flair_3d.astype(np.float32), affine)
    seg_img   = nib.Nifti1Image(seg_3d.astype(np.int16), affine)

    # Convert voxel-space centroid to world-space (mm) for cut_coords
    if centroid_xyz is not None:
        vox = np.array([centroid_xyz[0], centroid_xyz[1], centroid_xyz[2], 1.0])
        world = affine @ vox
        cut_coords = tuple(world[:3])
    else:
        cut_coords = None

    fig = plt.figure(figsize=(12, 4), facecolor='black')
    try:
        plotting.plot_roi(
            roi_img=seg_img,
            bg_img=flair_img,
            cut_coords=cut_coords,
            display_mode='ortho',
            figure=fig,
            cmap='Paired',         # matches Kaggle BraTS notebooks
            alpha=0.72,
            colorbar=False,
            black_bg=True,
            annotate=True,
            draw_cross=True,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format='PNG', dpi=120, bbox_inches='tight',
                    facecolor='black', pad_inches=0.05)
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    except Exception as e:
        print('[Brain] plot_roi failed:', e, flush=True)
        plt.close(fig)
        return _render_axial_only(raw_flair_3d, seg_3d > 0,
                                  centroid_xyz[2] if centroid_xyz is not None else seg_3d.shape[2] // 2)


def _render_single_view(raw_flair_3d, seg_3d, affine, mode, world_coord):
    """Render ONE orthogonal view (mode = 'x' sagittal / 'y' coronal / 'z' axial)
    using the same discrete BraTS palette."""
    try:
        import nibabel as nib
        from nilearn import plotting
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return None

    flair_img = nib.Nifti1Image(raw_flair_3d.astype(np.float32), affine)
    seg_img = nib.Nifti1Image(seg_3d.astype(np.int16), affine)
    fig = plt.figure(figsize=(4.5, 4.5), facecolor='black')
    try:
        plotting.plot_roi(
            roi_img=seg_img, bg_img=flair_img,
            cut_coords=[world_coord], display_mode=mode,
            figure=fig, cmap='Paired', alpha=0.72,
            colorbar=False, black_bg=True, annotate=True, draw_cross=True,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format='PNG', dpi=110, bbox_inches='tight',
                    facecolor='black', pad_inches=0.02)
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    except Exception as e:
        plt.close(fig)
        print('[Brain] single-view render failed:', mode, e, flush=True)
        return None


def _render_pipeline_preview(raw_modalities, seg_3d, axial_idx=None):
    """Render the classic BraTS preprocessing preview — 2×3 grid showing
    [FLAIR | T1 | T1ce] / [T2 | Seg Mask (colored) | Mask Gray (binary)]
    at a single axial slice. Mirrors the Kaggle BraTS notebook style so
    the user can verify the model is seeing the right inputs.
    """
    try:
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except Exception as e:
        print('[Brain] pipeline preview: matplotlib unavailable:', e, flush=True)
        return None

    # Pick the slice with the most tumor pixels (most informative view)
    if axial_idx is None:
        if (seg_3d > 0).any():
            slice_areas = (seg_3d > 0).sum(axis=(0, 1))
            axial_idx = int(np.argmax(slice_areas))
        else:
            axial_idx = seg_3d.shape[2] // 2
    axial_idx = max(0, min(seg_3d.shape[2] - 1, int(axial_idx)))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9), facecolor='#0a0a14')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01,
                        wspace=0.06, hspace=0.12)

    modality_keys = ['flair', 't1', 't1ce', 't2']
    titles = ['Image FLAIR', 'Image T1', 'Image T1ce', 'Image T2']

    # Row 1 (FLAIR, T1, T1ce) + Row 2 col 0 (T2)
    for i, (k, title) in enumerate(zip(modality_keys, titles)):
        if i < 3:
            ax = axes[0, i]
        else:
            ax = axes[1, 0]
        if k in raw_modalities and raw_modalities[k] is not None:
            img = raw_modalities[k][:, :, axial_idx]
            # Auto window-level on brain voxels for nicer contrast
            brain = img[img > 0]
            if brain.size > 0:
                vmin = float(np.percentile(brain, 1))
                vmax = float(np.percentile(brain, 99))
            else:
                vmin, vmax = 0.0, 1.0
            ax.imshow(np.rot90(img), cmap='gray', vmin=vmin, vmax=vmax,
                      interpolation='bilinear')
        ax.set_title(title, color='white', fontsize=13, pad=6,
                     fontfamily='sans-serif')
        ax.axis('off')
        ax.set_facecolor('#0a0a14')

    # Row 2 col 1 — Seg Mask (colored, BraTS palette on dark BG)
    ax_seg = axes[1, 1]
    seg_slice = seg_3d[:, :, axial_idx]
    seg_cmap = ListedColormap([
        '#1a0a3a',   # background — dark purple (like Kaggle reference)
        '#ff4466',   # NCR — red
        '#ffd84a',   # ED — yellow
        '#22ff66',   # ET — green (matches Kaggle "Seg Mask" panel)
    ])
    ax_seg.imshow(np.rot90(seg_slice), cmap=seg_cmap, vmin=0, vmax=3,
                  interpolation='nearest')
    ax_seg.set_title('Seg Mask', color='white', fontsize=13, pad=6)
    ax_seg.axis('off')

    # Row 2 col 2 — Mask Gray (binary whole-tumor)
    ax_mask = axes[1, 2]
    binary = (seg_slice > 0).astype(np.float32)
    # Show on black BG; tumor in light gray
    ax_mask.imshow(np.rot90(binary), cmap='gray', vmin=0, vmax=1.2,
                   interpolation='nearest')
    ax_mask.set_title('Mask Gray', color='white', fontsize=13, pad=6)
    ax_mask.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='PNG', dpi=110, bbox_inches='tight',
                facecolor='#0a0a14', pad_inches=0.12)
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def _render_axial_only(raw_flair_3d, mask_3d, axial_idx):
    """Single-axial fallback used when nilearn isn't installed or errors."""
    base = raw_flair_3d[:, :, axial_idx].astype(np.float32)
    brain_mask = base > 0
    if int(brain_mask.sum()) > 100:
        lo = np.percentile(base[brain_mask], 1)
        hi = np.percentile(base[brain_mask], 99)
        if hi - lo < 1e-3:
            lo, hi = float(base.min()), float(base.max() + 1e-3)
        norm = np.clip((base - lo) / (hi - lo), 0, 1)
        norm[~brain_mask] = 0
    else:
        rng = max(float(base.max() - base.min()), 1e-3)
        norm = (base - base.min()) / rng

    rgb = (np.stack([norm, norm, norm], axis=-1) * 255).astype(np.uint8)
    rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    m_resized = cv2.resize(mask_3d[:, :, axial_idx].astype(np.uint8),
                           (256, 256), interpolation=cv2.INTER_NEAREST).astype(bool)
    if m_resized.any():
        red_layer = rgb.copy()
        red_layer[m_resized] = [255, 70, 90]
        rgb = cv2.addWeighted(rgb, 0.55, red_layer, 0.45, 0)
        m_u8 = (m_resized.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, (60, 255, 130), 1)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


@app.route('/api/predict-brain', methods=['POST'])
def predict_brain():
    """Brain tumor segmentation.

    Two input modes:
      1) 4 NIfTI files (proper BraTS): form fields flair/t1/t1ce/t2
      2) 1 image file (demo): form field 'image' — replicated to 4 channels,
         tiled to a fake 128³ volume.
    """
    _load_brain_model()
    if not brain_model_loaded:
        return jsonify({'error': 'Brain model not loaded',
                        'detail': brain_model_error}), 503

    # Save uploads to temp files (NIfTI loader needs a real path)
    saved = {}
    gt_seg_path = None
    img_pil = None
    try:
        for k in ('flair', 't1', 't1ce', 't2'):
            if k in request.files:
                f = request.files[k]
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz')
                f.save(tmp.name)
                tmp.close()
                saved[k] = tmp.name
        # Optional ground-truth segmentation (BraTS _seg.nii.gz) — used for
        # Dice scoring + overlay comparison, NOT fed to the model.
        if 'gt_seg' in request.files:
            gt_f = request.files['gt_seg']
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz')
            gt_f.save(tmp.name)
            tmp.close()
            gt_seg_path = tmp.name
        if 'image' in request.files and not saved:
            img_pil = Image.open(request.files['image'])

        if not saved and img_pil is None:
            return jsonify({'error': 'No input — send NIfTI fields (flair/t1/t1ce/t2) or "image" file'}), 400

        # NIfTI: brain-bbox crop → 128³ resize (matches model's training
        # distribution). Sliding-window at native res was tried but caused
        # patch-boundary hallucination because this checkpoint was trained
        # on full-brain 128³, not patches.
        # 2D image: replicate-to-4-channels demo path.
        use_sliding_window = False
        if saved:
            x, orig_shape, raw_modalities, affine = _build_brain_input_from_files(saved)
            input_mode = 'nifti'
        else:
            x, orig_shape, raw_modalities, affine = _build_brain_input_from_image(img_pil)
            input_mode = 'image'
        voxel_mm3_native = 1.0
        raw_flair_3d = raw_modalities.get('flair')
        if raw_flair_3d is None:
            raw_flair_3d = next(iter(raw_modalities.values()))

        # Inference: single forward pass at 128³ with 4-way TTA (original +
        # 3 axis flips). One batched predict() call for all 4 variants →
        # ~4× slower than no TTA, but gives a free +2-3% Dice boost.
        import time
        t0 = time.time()
        probs_sm = _predict_with_tta(x, brain_model, num_classes=4)
        elapsed = time.time() - t0
        print(f'[Brain] TTA inference complete in {elapsed:.2f}s, output shape {probs_sm.shape}', flush=True)
        seg_raw = np.argmax(probs_sm, axis=-1).astype(np.int8)
        cls_dbg_raw = {int(c): int((seg_raw == c).sum()) for c in np.unique(seg_raw)}

        # ============================================================
        # POST-PROCESSING PIPELINE — fix fragmented / disconnected masks
        # so 3D reconstruction produces a coherent, smooth tumor volume.
        # ============================================================
        from scipy.ndimage import (gaussian_filter, binary_closing,
                                    binary_fill_holes, binary_dilation,
                                    binary_erosion, label as cc_label,
                                    generate_binary_structure)

        # ----- STAGE 1 — Probability-map smoothing -----
        # The raw softmax map has voxel-level noise. Smoothing it BEFORE
        # argmax produces continuous boundaries instead of speckled edges
        # AND lets adjacent slices vote (3D-consistency for free).
        probs_smooth = np.empty_like(probs_sm)
        for cls in range(probs_sm.shape[-1]):
            probs_smooth[..., cls] = gaussian_filter(
                probs_sm[..., cls].astype(np.float32),
                sigma=(0.8, 0.8, 0.8),     # 3D smooth, ~1 voxel
                mode='nearest',
            )
        # Re-normalize (gaussian filter doesn't preserve sum=1 exactly)
        probs_smooth /= np.maximum(probs_smooth.sum(axis=-1, keepdims=True), 1e-6)

        # ----- STAGE 2 — Tumor-presence detection (whole-tumor mask) -----
        # Decide IF a voxel is tumor using the foreground probability
        # (sum of all 3 tumor classes), then decide WHICH class separately.
        # This avoids the trap of "no single class clears 0.5 but combined
        # tumor prob is 0.85" — those voxels were being thrown away before.
        tumor_fg_prob = probs_smooth[..., 1:].sum(axis=-1)
        WT_THRESHOLD = 0.30     # was 0.35 — slightly more permissive so we
                                # capture lesion border voxels that are
                                # individually low-confidence per class but
                                # collectively certain to be tumor.
        wt_mask = tumor_fg_prob > WT_THRESHOLD

        # ----- STAGE 3 — Brain mask filter -----
        # Strip false positives in air / skull / outside-brain regions.
        try:
            flair_for_mask = x[0, ..., 0]
            brain_mask_3d = flair_for_mask > flair_for_mask.mean() * 0.1
            brain_mask_3d = binary_dilation(brain_mask_3d, iterations=2)
            wt_mask = wt_mask & brain_mask_3d
        except Exception as e_bm:
            print(f'[Brain] brain-mask filter skipped: {e_bm}', flush=True)

        # ----- STAGE 4 — Morphological closing + hole filling -----
        # Closing (dilate→erode) bridges thin gaps caused by partial-volume
        # voxels at lesion edges. Hole-filling closes interior cysts in
        # rendering (NCR can sometimes appear as separate islands inside ET).
        struct3 = generate_binary_structure(3, 2)   # 18-connectivity ball
        wt_mask = binary_closing(wt_mask, structure=struct3, iterations=2)
        wt_mask = binary_fill_holes(wt_mask)

        # ----- STAGE 5 — Connected-component filter -----
        # Real glioma is 1-2 contiguous masses. Tiny scattered CCs are
        # nearly always FP noise. Keep components ≥ MIN_CC_SIZE OR the
        # largest few CCs (whichever wins).
        MIN_CC_SIZE = 80          # 150 was over-aggressive after bbox-resize
                                  # (each voxel covers ~3 mm³ now, so a 30mm
                                  # lesion has ~3000 voxels — but small
                                  # satellite nodules can be 80-150 voxels).
        TOP_K_CCS   = 3
        lbl, n_cc = cc_label(wt_mask, structure=struct3)
        if n_cc > 0:
            counts = np.bincount(lbl.ravel())
            counts[0] = 0   # background CC = 0
            keep_ccs = set(np.where(counts >= MIN_CC_SIZE)[0].tolist())
            if len(keep_ccs) < 1:
                # Fallback: keep the top-K largest no matter what
                top_k = np.argsort(counts)[-TOP_K_CCS:]
                keep_ccs = set(int(c) for c in top_k if counts[c] > 0)
            wt_mask = np.isin(lbl, list(keep_ccs))

        # ----- STAGE 6 — Per-class assignment WITHIN whole-tumor mask -----
        # Inside the cleaned WT mask, decide each voxel's class by argmax
        # over the tumor classes only (background is excluded — we already
        # know these are tumor voxels). This recovers fine sub-region
        # structure (NCR core / ED edema / ET enhancing ring) that the
        # confidence-threshold approach was destroying.
        cls_probs = probs_smooth[..., 1:]                    # (D,H,W,3)
        cls_argmax = np.argmax(cls_probs, axis=-1) + 1       # 1..3
        seg = np.zeros_like(seg_raw)
        seg[wt_mask] = cls_argmax[wt_mask]

        # ----- STAGE 7 — Per-class light smoothing for cleaner sub-regions -----
        # Tiny per-class CCs inside the WT mask are usually segmentation
        # ambiguity, not real anatomy. Drop sub-CCs below a small floor.
        for cls_id in (1, 2, 3):
            cls_mask = seg == cls_id
            if cls_mask.sum() == 0:
                continue
            # Closing within each class so it forms a contiguous region
            cls_mask = binary_closing(cls_mask, structure=struct3, iterations=1)
            # Drop tiny per-class fragments (likely noise / misclassification)
            cl_lbl, cl_n = cc_label(cls_mask, structure=struct3)
            if cl_n > 0:
                cl_counts = np.bincount(cl_lbl.ravel())
                cl_counts[0] = 0
                small = np.where((cl_counts < 50) & (cl_counts > 0))[0]
                if len(small) > 0:
                    cls_mask[np.isin(cl_lbl, small)] = False
            # Write back into seg (only inside WT mask to keep WT shape stable)
            keep_for_class = cls_mask & wt_mask
            # Voxels we removed from this class fall back to the runner-up
            # tumor class — don't lose them to background.
            lost = (seg == cls_id) & ~keep_for_class
            if lost.any():
                # Recompute argmax over the OTHER 2 classes for lost voxels
                other = cls_probs.copy()
                other[..., cls_id - 1] = -1
                runner_up = np.argmax(other, axis=-1) + 1
                seg[lost] = runner_up[lost]
            seg[(seg == cls_id) & ~keep_for_class] = 0  # safety

        # Final WT mask after per-class cleanup
        tumor_mask = seg > 0
        # Recompute max_probs from SMOOTHED probs for confidence
        max_probs = probs_smooth.max(axis=-1)

        cls_dbg = {int(c): int((seg == c).sum()) for c in np.unique(seg)}
        # Log inference summary — probs_smooth is the post-softmax post-smoothing
        # probability map, present in both paths (sliding-window + legacy).
        print(f'[Brain] Inference {elapsed:.2f}s, '
              f'prob range {probs_smooth.min():.3f}–{probs_smooth.max():.3f}, '
              f'mode={"sliding+TTA" if use_sliding_window else "single-pass"}', flush=True)
        print(f'[Brain] Pipeline — raw: {cls_dbg_raw}', flush=True)
        print(f'[Brain] Pipeline — cleaned: {cls_dbg} '
              f'(WT thresh={WT_THRESHOLD}, min-CC={MIN_CC_SIZE}, top-K={TOP_K_CCS})', flush=True)

        tumor_voxels = int(tumor_mask.sum())
        # Voxel volume:
        #   - sliding-window (native): use voxel_mm3_native (1mm³ for BraTS)
        #   - bbox-resize fallback:    use the cache's voxel_mm3_resized
        #   - else:                    det(affine) or 1.0
        if use_sliding_window:
            voxel_mm3 = voxel_mm3_native
        else:
            voxel_mm3 = _BRAIN_BBOX_CACHE.get('voxel_mm3_resized') or 0.0
            if voxel_mm3 <= 0:
                try:
                    orig_v = float(abs(np.linalg.det(affine[:3, :3])))
                    voxel_mm3 = orig_v if (1e-6 < orig_v < 50.0) else 1.0
                except Exception:
                    voxel_mm3 = 1.0
        volume_mm3 = float(tumor_voxels) * voxel_mm3
        volume_cm3 = volume_mm3 / 1000.0
        print(f'[Brain] Volume: {tumor_voxels} voxels × {voxel_mm3:.3f} mm³/voxel '
              f'= {volume_mm3:.1f} mm³ ({volume_cm3:.2f} cm³)', flush=True)

        # Class breakdown
        class_counts = {
            'NCR':  int((seg == 1).sum()),
            'ED':   int((seg == 2).sum()),
            'ET':   int((seg == 3).sum()),
        }

        # `seg` is at native resolution when sliding-window was used. For
        # FE-bound payloads (centroid128, bbox128, tumorMesh, voxelCloud) we
        # downsample to 128³ so the existing client coords still apply.
        # Rendering functions (multi-view, pipeline preview) keep using the
        # native-res `seg` + `raw_flair_3d` for max visual quality.
        if seg.shape != (128, 128, 128):
            from scipy.ndimage import zoom as _seg_zoom
            f128 = (128.0/seg.shape[0], 128.0/seg.shape[1], 128.0/seg.shape[2])
            seg_fe = _seg_zoom(seg.astype(np.int8), f128, order=0,
                                mode='constant', cval=0).astype(np.int8)
            _o = np.zeros((128, 128, 128), dtype=np.int8)
            _s = [min(128, seg_fe.shape[i]) for i in range(3)]
            _o[:_s[0], :_s[1], :_s[2]] = seg_fe[:_s[0], :_s[1], :_s[2]]
            seg_fe = _o
        else:
            seg_fe = seg
        tumor_mask_fe = seg_fe > 0

        if tumor_voxels:
            # Native-resolution bbox → accurate diameter using true mm spacing
            zs_n, ys_n, xs_n = np.where(tumor_mask)
            sx_mm = float(abs(affine[0, 0])) or 1.0
            sy_mm = float(abs(affine[1, 1])) or 1.0
            sz_mm = float(abs(affine[2, 2])) or 1.0
            dx_mm = (int(xs_n.max()) - int(xs_n.min())) * sx_mm
            dy_mm = (int(ys_n.max()) - int(ys_n.min())) * sy_mm
            dz_mm = (int(zs_n.max()) - int(zs_n.min())) * sz_mm
            max_diameter_mm = float(max(dx_mm, dy_mm, dz_mm))
            # Native centroid for the renderers (nilearn uses native + affine)
            cent_native_x = int(xs_n.mean())
            cent_native_y = int(ys_n.mean())
            cent_native_z = int(zs_n.mean())
            # Native axial slice (most tumor area) for the pipeline preview
            slice_areas_native = tumor_mask.sum(axis=(0, 1))
            best_axial_native = int(np.argmax(slice_areas_native))
            # 128³-space bbox/centroid for FE viz
            if tumor_mask_fe.any():
                zs, ys, xs = np.where(tumor_mask_fe)
                zmin, zmax = int(zs.min()), int(zs.max())
                ymin, ymax = int(ys.min()), int(ys.max())
                xmin, xmax = int(xs.min()), int(xs.max())
                centroid = {
                    'x': int(xs.mean()),
                    'y': int(ys.mean()),
                    'z': int(zs.mean()),
                }
                slice_areas_fe = tumor_mask_fe.sum(axis=(0, 1))
                best_axial = int(np.argmax(slice_areas_fe))
            else:
                # Downsample lost the tumor — fall back to mid-coords
                zmin = ymin = xmin = 60; zmax = ymax = xmax = 68
                centroid = {'x': 64, 'y': 64, 'z': 64}
                best_axial = 64
        else:
            zmin = zmax = ymin = ymax = xmin = xmax = 0
            max_diameter_mm = 0.0
            centroid = {'x': 64, 'y': 64, 'z': 64}
            best_axial = 64
            cent_native_x = cent_native_y = cent_native_z = None
            best_axial_native = seg.shape[2] // 2

        # Centroid for renderers: native voxel coords (work with native affine)
        cent_vox_native = (cent_native_x, cent_native_y, cent_native_z) \
            if (tumor_voxels and cent_native_x is not None) else None

        # Kaggle-style 2×3 pipeline preview (uses native-res arrays for sharpness)
        pipeline_preview_b64 = _render_pipeline_preview(
            raw_modalities, seg, axial_idx=best_axial_native,
        )
        overlay_b64 = _render_brats_multiview(
            raw_flair_3d, seg, affine, centroid_xyz=cent_vox_native,
        )
        # Per-axis views for the 3 slot boxes (Axial / Sagittal / Coronal)
        axial_b64 = sagittal_b64 = coronal_b64 = None
        if tumor_voxels and cent_vox_native is not None:
            try:
                vox = np.array([cent_vox_native[0], cent_vox_native[1], cent_vox_native[2], 1.0])
                world = (affine @ vox)[:3]
                sagittal_b64 = _render_single_view(raw_flair_3d, seg, affine, 'x', float(world[0]))
                coronal_b64  = _render_single_view(raw_flair_3d, seg, affine, 'y', float(world[1]))
                axial_b64    = _render_single_view(raw_flair_3d, seg, affine, 'z', float(world[2]))
            except Exception as e:
                print('[Brain] per-axis render failed:', e, flush=True)

        # ============================================================
        # GROUND TRUTH COMPARISON — load BraTS *_seg.nii.gz if provided
        # and compute Dice scores + comparison overlay so the user can
        # see exactly where the model agrees / disagrees with experts.
        # ============================================================
        ground_truth_data = None
        gt_overlay_b64 = None
        if gt_seg_path is not None:
            try:
                import nibabel as nib_gt
                gt_full = np.asarray(nib_gt.load(gt_seg_path).get_fdata()).astype(np.int16)
                # BraTS 2020/2021: labels 0,1,2,4 → remap to 0,1,2,3
                if (gt_full == 4).any():
                    gt_full = gt_full.copy()
                    gt_full[gt_full == 4] = 3
                # Align GT to prediction coordinate space:
                #   - Sliding-window path: prediction is at NATIVE resolution.
                #     GT is already at native res — just resample to match
                #     pred's shape if needed.
                #   - Bbox-resize path: prediction is in 128³ bbox-cropped
                #     space. Use the cached bbox to crop+resize GT the same way.
                from scipy.ndimage import zoom as _gt_zoom
                target_shape = seg.shape    # prediction's spatial shape
                if use_sliding_window and gt_full.shape == target_shape:
                    gt_seg_3d = gt_full.astype(np.int8)
                elif use_sliding_window:
                    # Native shape mismatch — resample GT to pred shape
                    factors = tuple(target_shape[i] / float(gt_full.shape[i])
                                    for i in range(3))
                    gt_seg_3d = _gt_zoom(gt_full.astype(np.int16),
                                          factors, order=0,
                                          mode='constant', cval=0).astype(np.int8)
                else:
                    # Legacy bbox-resize path
                    gt_cache_bbox = _BRAIN_BBOX_CACHE.get('bbox')
                    if gt_cache_bbox is not None:
                        z0, z1, y0, y1, x0, x1 = gt_cache_bbox
                        z0c, z1c = max(0, z0), min(gt_full.shape[0], z1)
                        y0c, y1c = max(0, y0), min(gt_full.shape[1], y1)
                        x0c, x1c = max(0, x0), min(gt_full.shape[2], x1)
                        gt_cropped = gt_full[z0c:z1c, y0c:y1c, x0c:x1c]
                        if gt_cropped.size > 0:
                            factors = (128 / float(gt_cropped.shape[0]),
                                       128 / float(gt_cropped.shape[1]),
                                       128 / float(gt_cropped.shape[2]))
                            gt_seg_3d = _gt_zoom(gt_cropped.astype(np.int16),
                                                  factors, order=0,
                                                  mode='constant', cval=0).astype(np.int8)
                            out_gt = np.zeros((128, 128, 128), dtype=np.int8)
                            s = [min(128, gt_seg_3d.shape[i]) for i in range(3)]
                            out_gt[:s[0], :s[1], :s[2]] = gt_seg_3d[:s[0], :s[1], :s[2]]
                            gt_seg_3d = out_gt
                        else:
                            gt_seg_3d = _crop_or_pad_3d(gt_full, target=128).astype(np.int8)
                    else:
                        gt_seg_3d = _crop_or_pad_3d(gt_full, target=128).astype(np.int8)

                def _dice(pred, gt):
                    p_sum = int(pred.sum())
                    g_sum = int(gt.sum())
                    if p_sum + g_sum == 0:
                        return 1.0
                    return float(2.0 * (pred & gt).sum() / (p_sum + g_sum))

                # BraTS official regions — compared in pred's native shape
                pred_wt = seg > 0
                pred_tc = (seg == 1) | (seg == 3)
                pred_et = seg == 3
                gt_wt = gt_seg_3d > 0
                gt_tc = (gt_seg_3d == 1) | (gt_seg_3d == 3)
                gt_et = gt_seg_3d == 3

                gt_class_counts = {
                    'NCR': int((gt_seg_3d == 1).sum()),
                    'ED':  int((gt_seg_3d == 2).sum()),
                    'ET':  int((gt_seg_3d == 3).sum()),
                }
                gt_volume_cm3 = float(gt_seg_3d.astype(bool).sum()) * voxel_mm3 / 1000.0

                ground_truth_data = {
                    'present': True,
                    'dice': {
                        'WT': round(_dice(pred_wt, gt_wt), 4),
                        'TC': round(_dice(pred_tc, gt_tc), 4),
                        'ET': round(_dice(pred_et, gt_et), 4),
                    },
                    'gtClassCounts': gt_class_counts,
                    'gtVolumeCm3':  round(gt_volume_cm3, 2),
                    'volumeDiffCm3': round(volume_cm3 - gt_volume_cm3, 2),
                }
                print(f'[Brain] GT loaded — Dice WT={ground_truth_data["dice"]["WT"]:.3f}, '
                      f'TC={ground_truth_data["dice"]["TC"]:.3f}, '
                      f'ET={ground_truth_data["dice"]["ET"]:.3f}, '
                      f'GT volume={gt_volume_cm3:.2f} cm³ (pred={volume_cm3:.2f} cm³)', flush=True)

                # Render GT multi-view (same palette so user can A/B with prediction)
                try:
                    gt_centroid = None
                    if gt_wt.any():
                        gz, gy, gx = np.where(gt_wt)
                        gt_centroid = (int(gx.mean()), int(gy.mean()), int(gz.mean()))
                    gt_overlay_b64 = _render_brats_multiview(
                        raw_flair_3d, gt_seg_3d, affine, centroid_xyz=gt_centroid,
                    )
                except Exception as e_gtr:
                    print(f'[Brain] GT overlay render failed: {e_gtr}', flush=True)
            except Exception as e_gt:
                import traceback
                print(f'[Brain] GT processing failed: {e_gt}\n{traceback.format_exc()}', flush=True)
                ground_truth_data = {'present': False, 'error': str(e_gt)}

        # Confidence = mean probability of the predicted class on tumor pixels.
        # Use the SMOOTHED probability map (matches what the cleanup used);
        # the raw probs_sm is identical-shape in both pipeline paths.
        if tumor_voxels:
            confidence = float(probs_smooth[tumor_mask].max(axis=-1).mean()) * 100
        else:
            confidence = 0.0

        # Build a surface mesh per tumor class using marching cubes — this is
        # what we render in the 3D viewport so the user sees the EXACT shape
        # the model segmented (smooth continuous mesh, not voxel dots / a
        # generic sphere). We smooth the binary mask first so the surface
        # doesn't look blocky at voxel boundaries.
        from skimage.measure import marching_cubes
        from scipy.ndimage import gaussian_filter as _gf_mc

        def _mesh_from_mask(mask, smooth_sigma=1.2, step_size=1,
                             max_faces=12000):
            """Build a smooth surface mesh from a binary mask.

            Pipeline: binary mask → Gaussian smooth (sigma=1.2 gives a soft
            isosurface) → marching cubes at level 0.45. The wider smoothing
            kernel + lower isolevel produces a continuous shrink-wrap surface
            instead of jagged voxel boundaries. Critical for the "production
            medical demo" look the user wants.
            """
            n_vox = int(mask.sum())
            if n_vox < 20:
                return None
            # Pad by 2 so the mesh doesn't get clipped at the volume edge
            padded = np.pad(mask.astype(np.float32), 2, mode='constant')
            sm = _gf_mc(padded, sigma=smooth_sigma)
            peak = float(sm.max())
            if peak < 0.15:
                return None
            level = max(0.20, min(0.50, peak * 0.55))
            try:
                verts, faces, _, _ = marching_cubes(sm, level=level,
                                                     step_size=step_size,
                                                     allow_degenerate=False)
                # Undo the padding offset (verts are in padded coords)
                verts = verts - 2
            except Exception as ex:
                print(f'[Brain] marching_cubes failed (level={level:.2f}, peak={peak:.2f}, voxels={n_vox}): {ex}', flush=True)
                return None
            if len(faces) == 0:
                print(f'[Brain] marching_cubes returned 0 faces (level={level:.2f}, peak={peak:.2f}, voxels={n_vox})', flush=True)
                return None
            # If the mesh is too dense, retry with a coarser step_size so the
            # response payload stays under a few MB.
            if len(faces) > max_faces and step_size == 1:
                try:
                    verts, faces, _, _ = marching_cubes(sm, level=level,
                                                         step_size=2,
                                                         allow_degenerate=False)
                except Exception:
                    pass
            # Reorder axes from skimage (axis-order = array-order, i.e.
            # axis-0 first) to (x, y, z). Our seg is laid out as (Z, Y, X)
            # in the rest of this file, so verts come back as (Z, Y, X) too —
            # swap to (X, Y, Z) for the frontend.
            verts_xyz = verts[:, [2, 1, 0]].astype(np.float32)
            # Normalize to [0, 1] so the frontend doesn't need to know grid size
            grid = float(mask.shape[0])
            verts_norm = verts_xyz / grid
            return verts_norm, faces.astype(np.int32)

        def _serialize_mesh(result):
            if result is None:
                return None
            verts, faces = result
            # Quantize verts to 4 decimals (uint16 range after × 10000)
            # — saves ~40% JSON size vs full float32 stringification.
            v_q = np.round(verts * 10000.0).astype(np.int32)
            return {
                'vertices': v_q.flatten().tolist(),
                'faces':    faces.flatten().tolist(),
                'vertexCount': int(len(verts)),
                'faceCount':   int(len(faces)),
                'scale':       10000,        # divide by this on FE
            }

        # Marching cubes on the 128³-downsampled seg (mesh size stays manageable
        # for the FE; the native-res mask is too dense — would yield 100k+
        # triangles per class and bloat the JSON payload).
        tumor_mesh = {
            'gridSize': int(seg_fe.shape[0]),
            'classes': {
                'ncr': _serialize_mesh(_mesh_from_mask(seg_fe == 1)),
                'ed':  _serialize_mesh(_mesh_from_mask(seg_fe == 2)),
                'et':  _serialize_mesh(_mesh_from_mask(seg_fe == 3)),
            },
        }
        # Debug log
        _mc = tumor_mesh['classes']
        print(f'[Brain] MC mesh — NCR: {(_mc["ncr"] or {}).get("faceCount", 0)} faces, '
              f'ED: {(_mc["ed"] or {}).get("faceCount", 0)} faces, '
              f'ET: {(_mc["et"] or {}).get("faceCount", 0)} faces', flush=True)

        # Belt-and-braces: ALSO send a sampled voxel cloud per class. If MC
        # produced no faces for some reason, the frontend can still render
        # the actual tumor shape as instanced cubes.
        def _sample_class_voxels(mask, max_count):
            if not mask.any():
                return []
            zs, ys, xs = np.where(mask)
            n = len(xs)
            if n <= max_count:
                idx = np.arange(n)
            else:
                step = max(1, n // max_count)
                idx = np.arange(0, n, step)[:max_count]
            return np.column_stack([xs[idx], ys[idx], zs[idx]]).astype(int).flatten().tolist()

        voxel_cloud = {
            'gridSize': int(seg_fe.shape[0]),
            'ncr': _sample_class_voxels(seg_fe == 1, 1500),
            'ed':  _sample_class_voxels(seg_fe == 2, 2500),
            'et':  _sample_class_voxels(seg_fe == 3, 1500),
        }

        return jsonify({
            'success': True,
            'detected': bool(tumor_voxels > 50),    # ignore tiny noise
            'inputMode': input_mode,
            'inferenceTimeS': round(elapsed, 2),
            'tumorVoxels': tumor_voxels,
            'volumeMm3': round(volume_mm3, 2),
            'volumeCm3': round(volume_cm3, 2),
            'maxDiameterMm': round(max_diameter_mm, 1),
            'classCounts': class_counts,
            'centroid128': centroid,
            'bbox128': {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'zmin': zmin, 'zmax': zmax},
            'tumorMesh': tumor_mesh,
            'voxelCloud': voxel_cloud,
            'displaySliceIndex': best_axial,
            'overlayImage': overlay_b64,
            'axialImage': axial_b64,
            'sagittalImage': sagittal_b64,
            'coronalImage': coronal_b64,
            'pipelinePreview': pipeline_preview_b64,
            'confidence': round(confidence, 1),
            'groundTruth': ground_truth_data,
            'groundTruthOverlay': gt_overlay_b64,
            'modelInfo': {
                'name': 'BraTS 3D U-Net',
                'type': '3D Semantic Segmentation',
                'inputShape': list(BRAIN_INPUT_SHAPE),
                'classes': ['Background', 'NCR', 'ED', 'ET'],
            }
        })
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f'[ERROR] /api/predict-brain: {e}\n{tb}', flush=True)
        return jsonify({'error': str(e), 'trace': tb}), 500
    finally:
        for p in saved.values():
            try:
                os.unlink(p)
            except Exception:
                pass
        if gt_seg_path:
            try: os.unlink(gt_seg_path)
            except Exception:
                pass


# ===== BRAIN TUMOR YOLO DETECTION (lightweight 2D bbox + class) =====
BRAIN_YOLO_PATH = os.path.join(MODEL_DIR, 'brain', 'yolo_brain_tumor.pt')
brain_yolo_model = None
brain_yolo_loaded = False
brain_yolo_warning = None     # set if we fell back to generic COCO weights
brain_yolo_error = None


def _load_brain_yolo():
    """Try brain-tumor weights; fall back to YOLOv8n COCO for pipeline test."""
    global brain_yolo_model, brain_yolo_loaded, brain_yolo_warning, brain_yolo_error
    if brain_yolo_loaded or brain_yolo_error:
        return
    try:
        from ultralytics import YOLO
        if os.path.exists(BRAIN_YOLO_PATH):
            brain_yolo_model = YOLO(BRAIN_YOLO_PATH)
            print(f'[OK] Brain YOLO loaded from {BRAIN_YOLO_PATH}')
        else:
            brain_yolo_model = YOLO('yolov8n.pt')   # auto-downloads if missing
            brain_yolo_warning = (
                'No brain-specific weights found. Using generic COCO weights '
                f'(detects everyday objects, not tumors). Place brain weights '
                f'at: {BRAIN_YOLO_PATH}'
            )
            print('[WARN]', brain_yolo_warning)
        brain_yolo_loaded = True
    except Exception as e:
        brain_yolo_error = str(e)
        print(f'[ERROR] YOLO load failed: {e}')


def _draw_yolo_overlay(img_pil, detections):
    """Draw bboxes + class labels on the image, return base64 PNG."""
    arr = np.array(img_pil.convert('RGB'))
    # Make image larger if too small (for label legibility)
    h, w = arr.shape[:2]
    target = 640
    if max(h, w) < target:
        s = target / max(h, w)
        arr = cv2.resize(arr, (int(w * s), int(h * s)),
                         interpolation=cv2.INTER_LINEAR)
        h, w = arr.shape[:2]
    else:
        s = 1.0

    palette = [(0, 220, 160), (255, 107, 92), (255, 180, 84), (139, 92, 246)]
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = [int(v * s) for v in d['bbox']]
        cls = d['class']
        conf = d['confidence']
        color = palette[d.get('classId', i) % len(palette)]
        cv2.rectangle(arr, (x1, y1), (x2, y2), color, 2)
        label = f'{cls} {conf*100:.0f}%'
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(arr, (x1, y1 - lh - 6), (x1 + lw + 6, y1), color, -1)
        cv2.putText(arr, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1, cv2.LINE_AA)

    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


@app.route('/api/predict-brain-yolo', methods=['POST'])
def predict_brain_yolo():
    """YOLOv8 2D brain tumor detection. Accepts a single image file."""
    _load_brain_yolo()
    if not brain_yolo_loaded:
        return jsonify({'error': 'YOLO model not loaded', 'detail': brain_yolo_error}), 503
    if 'image' not in request.files:
        return jsonify({'error': 'Missing "image" file field'}), 400

    try:
        f = request.files['image']
        img_bytes = f.read()
        pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        import time
        t0 = time.time()
        results = brain_yolo_model.predict(np.array(pil), verbose=False, conf=0.25)
        elapsed = time.time() - t0

        detections = []
        for r in results:
            names = r.names
            boxes = r.boxes
            if boxes is None:
                continue
            for b in boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()   # [x1,y1,x2,y2]
                conf = float(b.conf[0].cpu())
                cls_id = int(b.cls[0].cpu())
                detections.append({
                    'bbox': xyxy,
                    'confidence': round(conf, 3),
                    'class': names.get(cls_id, str(cls_id)),
                    'classId': cls_id,
                })

        overlay_b64 = _draw_yolo_overlay(pil, detections)

        # Find top detection by confidence
        top = max(detections, key=lambda d: d['confidence']) if detections else None

        return jsonify({
            'success': True,
            'detected': len(detections) > 0,
            'inferenceTimeS': round(elapsed, 3),
            'detections': detections,
            'topClass': top['class'] if top else None,
            'topConfidence': top['confidence'] if top else 0,
            'imageSize': {'w': pil.size[0], 'h': pil.size[1]},
            'overlayImage': overlay_b64,
            'warning': brain_yolo_warning,
            'modelInfo': {
                'name': 'YOLOv8 Brain Tumor Detection',
                'type': '2D Object Detection',
                'weights': BRAIN_YOLO_PATH if os.path.exists(BRAIN_YOLO_PATH) else 'yolov8n.pt (generic COCO, FALLBACK)',
            }
        })
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f'[ERROR] /api/predict-brain-yolo: {e}\n{tb}', flush=True)
        return jsonify({'error': str(e), 'trace': tb}), 500


@app.route('/api/brain-model-info', methods=['GET'])
def brain_model_info():
    _load_brain_model()
    return jsonify({
        'loaded': brain_model_loaded,
        'error': brain_model_error,
        'inputShape': list(BRAIN_INPUT_SHAPE),
        'modelPath': BRAIN_MODEL_PATH,
    })


@app.route('/api/brain-models', methods=['GET'])
def list_brain_models():
    """Enumerate available brain-segmentation model files so the FE can
    show a selector dropdown. Scans both /models/brain/ (production .keras
    files) and /models/brain/models_3D/ (epoch-checkpoint .hdf5 files).
    """
    import re
    brain_dir = os.path.join(MODEL_DIR, 'brain')
    models = []
    seen = set()

    def add_model(rel_path, display_name=None, subtitle=None, badge=None):
        rel_norm = rel_path.replace('\\', '/')
        if rel_norm in seen:
            return
        full_path = os.path.join(brain_dir, rel_path.replace('/', os.sep))
        if not os.path.exists(full_path):
            return
        seen.add(rel_norm)
        size_mb = os.path.getsize(full_path) / (1024 * 1024)
        active = (rel_norm == _brain_model_name.replace('\\', '/'))
        models.append({
            'filename': rel_norm,
            'displayName': display_name or rel_norm,
            'subtitle':    subtitle or f'{size_mb:.0f} MB',
            'sizeMb':      round(size_mb, 1),
            'badge':       badge,
            'type':        'hdf5' if rel_path.endswith(('.hdf5', '.h5')) else 'keras',
            'active':      active,
        })

    # Production-tier models in /models/brain/
    add_model('brats_3d_unet_kaggle_best.keras',
              'Kaggle OncoSense Best', 'Production · 3D U-Net · BraTS 2020',
              badge='Stable')
    add_model('model.keras',
              'Large 3D U-Net', 'Alternative architecture · 372MB',
              badge='Experimental')

    # 2018 epoch checkpoints in /models/brain/models_3D/
    sub_dir = os.path.join(brain_dir, 'models_3D')
    if os.path.isdir(sub_dir):
        for f in sorted(os.listdir(sub_dir)):
            if not f.endswith(('.hdf5', '.h5', '.keras')):
                continue
            m = re.search(r'-(\d+)-([\d.]+)\.(hdf5|h5|keras)$', f)
            if m:
                epoch = int(m.group(1))
                val_acc = float(m.group(2))
                name = f'3D U-Net 2018 · Epoch {epoch:02d}'
                # val_accuracy is ~99% baseline due to BG class imbalance;
                # the meaningful number is the trend (1→2→4→7→8 = better Dice)
                badge = 'Latest' if epoch >= 8 else ('Best' if epoch >= 7 else None)
                sub = f'val_acc {val_acc:.3f} · 93MB'
                add_model(f'models_3D/{f}', name, sub, badge=badge)
            else:
                add_model(f'models_3D/{f}', f, None)

    return jsonify({
        'models': models,
        'currentModel': _brain_model_name.replace('\\', '/'),
        'currentLoaded': brain_model_loaded,
        'currentError': brain_model_error,
    })


@app.route('/api/brain-model-switch', methods=['POST'])
def switch_brain_model():
    """Swap the active brain model. Loads the new one into a temp slot
    first; only commits to the global brain_model after successful load,
    so a failed switch doesn't break the running prediction pipeline.
    """
    global brain_model, brain_model_loaded, brain_model_error
    global BRAIN_MODEL_PATH, _brain_model_name

    data = request.get_json(silent=True) or {}
    new_filename = (data.get('filename') or '').strip().replace('\\', '/')
    if not new_filename:
        return jsonify({'success': False, 'error': 'Missing filename'}), 400

    # Security: filename must stay inside /models/brain/
    safe_path = os.path.normpath(os.path.join(
        MODEL_DIR, 'brain', new_filename.replace('/', os.sep)))
    brain_dir = os.path.normpath(os.path.join(MODEL_DIR, 'brain'))
    if not safe_path.startswith(brain_dir):
        return jsonify({'success': False, 'error': 'Invalid path'}), 400
    if not os.path.exists(safe_path):
        return jsonify({'success': False, 'error': f'Not found: {new_filename}'}), 404

    import time
    t0 = time.time()
    try:
        new_model = _try_load_one(safe_path)
    except Exception as e:
        print(f'[Brain] Model switch failed for {new_filename}: {e}', flush=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'currentModel': _brain_model_name.replace('\\', '/'),
        }), 500

    # Validate compatibility with our segmentation pipeline. The dispatcher
    # supports 5D (3D U-Net) and 4D (2D U-Net) seg models, NOT classifiers.
    in_shape = tuple(new_model.input_shape)
    out_shape = tuple(new_model.output_shape)
    pipeline_type = None
    incompat_reason = None
    if len(in_shape) == 5 and len(out_shape) == 5:
        pipeline_type = '3d-seg'
    elif len(in_shape) == 4 and len(out_shape) == 4:
        pipeline_type = '2d-seg'
    else:
        incompat_reason = (
            f'Model is not a segmentation network. Input {in_shape}, '
            f'output {out_shape}. Pipeline supports 3D U-Net '
            f'(None,D,H,W,C)→(None,D,H,W,K) or 2D U-Net '
            f'(None,H,W,C)→(None,H,W,K). Classifier models with '
            f'flat output shapes are not supported.'
        )
    if incompat_reason:
        print(f'[Brain] Switch REJECTED: {incompat_reason}', flush=True)
        return jsonify({
            'success': False,
            'error': incompat_reason,
            'inputShape':  [None if x is None else int(x) for x in in_shape],
            'outputShape': [None if x is None else int(x) for x in out_shape],
            'currentModel': _brain_model_name.replace('\\', '/'),
        }), 400

    # Commit — only swap globals once new model fully loaded + validated
    brain_model = new_model
    brain_model_loaded = True
    brain_model_error = None
    BRAIN_MODEL_PATH = safe_path
    _brain_model_name = new_filename
    elapsed = time.time() - t0

    print(f'[Brain] Switched to {new_filename} ({brain_model.count_params():,} params, '
          f'pipeline={pipeline_type}, {elapsed:.2f}s)', flush=True)
    return jsonify({
        'success': True,
        'currentModel': new_filename,
        'inputShape':  [None if x is None else int(x) for x in in_shape],
        'outputShape': [None if x is None else int(x) for x in out_shape],
        'pipelineType': pipeline_type,
        'params': int(brain_model.count_params()),
        'loadTimeS': round(elapsed, 2),
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'eeg_model': 'loaded',
        'lung_model': 'loaded' if lung_model_loaded else 'not loaded',
        'brain_model': 'loaded' if brain_model_loaded else ('error' if brain_model_error else 'lazy'),
        'type': 'CNN+BiGRU+Attention / DeepLabV3 / 3D-UNet'
    })


# ===== EXPRESS-COMPAT ALIASES =====
# The legacy FE was originally fronted by an Express proxy that added
# its own endpoints (mainly *-status thin wrappers around the Python
# *-info routes, plus MCP/chat/history features). The Next.js shell now
# proxies /api/* directly to Python, so we mirror those Express routes
# here as thin aliases. Keeps the FE happy with zero changes.

@app.route('/api/model-status', methods=['GET'])
def model_status_alias():
    """Express used /api/model-status; Python has /api/model-info."""
    return model_info()


@app.route('/api/lung-model-status', methods=['GET'])
def lung_model_status_alias():
    return lung_model_info()


@app.route('/api/brain-model-status', methods=['GET'])
def brain_model_status_alias():
    return brain_model_info()


@app.route('/api/mcp/status', methods=['GET'])
def mcp_status_stub():
    """MCP (Model Context Protocol) server status. Was implemented in the
    Express proxy as a thin Node-side feature; not present in this Python
    backend. Return a clean 'unavailable' so the FE shows the right state
    instead of an alarming 404 in the console."""
    return jsonify({
        'available': False,
        'reason': 'MCP server not running in Python backend (was Express-only feature)',
        'tools': [],
    })


@app.route('/api/mcp/execute', methods=['POST'])
def mcp_execute_stub():
    """Same as /api/mcp/status — MCP not wired into Python BE."""
    return jsonify({
        'error': 'MCP not available',
        'detail': 'Run the legacy Express server if you need MCP tool execution',
    }), 503


@app.route('/api/history', methods=['GET'])
def history_stub():
    """Express kept an in-memory analysis history. With Vercel + ngrok the
    backend can be restarted at any time, so persistent history would need
    a database. For now return empty — the FE just shows 'no history'."""
    return jsonify({
        'history': [],
        'note': 'History persistence requires a DB. Not implemented in standalone Python BE.',
    })


@app.route('/api/chat', methods=['POST'])
def chat_stub():
    """Express had a chat-session feature (probably wrapping an LLM API).
    Not implemented in the Python BE. Returns a polite error so the FE
    chat widget shows a message instead of crashing."""
    return jsonify({
        'error': 'Chat feature not available in this backend build',
        'reply': 'Tính năng chat chưa được kích hoạt. Vui lòng dùng các module phân tích chính (EEG / Brain / Lung / Blood).',
    }), 501


@app.route('/api/analyze', methods=['POST'])
def analyze_stub():
    """Generic Express '/api/analyze' endpoint (was a wrapper around EEG
    image analysis). Return 501 so the FE shows 'not implemented' rather
    than 404."""
    return jsonify({
        'error': 'Generic /api/analyze not available — use module-specific endpoints',
        'available': ['/api/predict-edf', '/api/predict-brain', '/api/predict-lung'],
    }), 501

if __name__ == '__main__':
    print("[START] Medical AI Python API starting on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
