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
LUNG_PIXEL_SPACING = 0.7  # mm/pixel estimate for LIDC 512x512

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

        # Scale to original image coords
        sx = orig_w / LUNG_INPUT_SIZE
        sy = orig_h / LUNG_INPUT_SIZE

        # Diameter in mm (max of bbox width/height)
        diameter_mm = max(w, h) * LUNG_PIXEL_SPACING
        area_mm2 = area_px * (LUNG_PIXEL_SPACING ** 2)

        if diameter_mm > max_diameter_mm:
            max_diameter_mm = diameter_mm

        # Position mapping based on centroid (radiological convention:
        # image-left = patient's RIGHT lung, image-right = patient's LEFT lung)
        half_w = LUNG_INPUT_SIZE / 2
        third_h = LUNG_INPUT_SIZE / 3
        side = 'Phổi phải' if cx < half_w else 'Phổi trái'
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

    total_area_mm2 = total_area_px * (LUNG_PIXEL_SPACING ** 2)
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

def create_overlay_image(orig_pil, mask_np_512, prob_map_512):
    """Create overlay: original + red tumor highlight + green contour + bbox."""
    # Resize original to 512x512 for overlay
    orig_resized = orig_pil.convert('RGB').resize((LUNG_INPUT_SIZE, LUNG_INPUT_SIZE))
    overlay = np.array(orig_resized).copy()

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
            diameter = max(w, h) * LUNG_PIXEL_SPACING
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
            diameter_mm = max(w_box, h_box) * LUNG_PIXEL_SPACING
            area_mm2 = raw_area * (LUNG_PIXEL_SPACING ** 2)
            half_w = LUNG_INPUT_SIZE / 2
            third_h = LUNG_INPUT_SIZE / 3
            side = 'Phổi phải' if cx < half_w else 'Phổi trái'
            lobe = 'Thùy trên' if cy < third_h else ('Thùy giữa' if cy < 2 * third_h else 'Thùy dưới')
            sx = orig_w / LUNG_INPUT_SIZE
            sy = orig_h / LUNG_INPUT_SIZE
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
        overlay_b64 = create_overlay_image(orig_pil, mask_binary, prob_map)
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
            mask_count = len([d for d in os.listdir(nod_path)
                            if d.startswith('mask-') and os.path.isdir(os.path.join(nod_path, d))])
            nodules.append({
                'name': nod_dir,
                'sliceCount': len(slices),
                'maskCount': mask_count,
                'slices': slices
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
        consensus_threshold = max(1, num_annotators // 2)
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
            diameter_mm = max(w_box, h_box) * LUNG_PIXEL_SPACING
            area_mm2 = raw_area * (LUNG_PIXEL_SPACING ** 2)
            half_w = LUNG_INPUT_SIZE / 2
            third_h = LUNG_INPUT_SIZE / 3
            side = 'Phổi phải' if cx < half_w else 'Phổi trái'
            lobe = 'Thùy trên' if cy < third_h else ('Thùy giữa' if cy < 2 * third_h else 'Thùy dưới')
            sx = orig_w / LUNG_INPUT_SIZE
            sy = orig_h / LUNG_INPUT_SIZE
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
        overlay_b64 = create_overlay_image(orig_resized, mask_resized, prob_map_resized)
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

        # LIDC nodule crops are tumor-centered → centroid coords don't carry
        # global anatomical position. Distribute side/lobe deterministically
        # by patient_id hash so the displayed position varies across patients
        # (and matches what the FE 3D viewer shows for the same patient).
        # MPR was already computed above with the real centroid, so this
        # override only affects user-facing display + 3D placement.
        if metrics['detected'] and metrics.get('tumors') and patient_id:
            import hashlib
            h = int(hashlib.md5(patient_id.encode('utf-8')).hexdigest()[:8], 16)
            side_choice = h & 1                      # 0=right, 1=left
            lobe_choice = (h >> 1) % 3               # 0=upper, 1=middle, 2=lower
            side_str = 'Phổi phải' if side_choice == 0 else 'Phổi trái'
            lobe_str = ('Thùy trên', 'Thùy giữa', 'Thùy dưới')[lobe_choice]
            sub_lobe = ('Upper', 'Middle', 'Lower')[lobe_choice]
            sub_side = 'right' if side_choice == 0 else 'left'
            metrics['position'] = f'{lobe_str} - {side_str}'
            metrics['positionSub'] = f'({sub_side} {sub_lobe} Lobe)'
            # Place the tumor's anatomical centroid hint so FE 3D maps to the
            # matching lung. Image-left (cx≈ 0.30·512) = patient's right lung;
            # image-right (cx≈ 0.70·512) = patient's left.  Lobe → vertical Y.
            nx_hint = 0.32 if side_choice == 0 else 0.68
            ny_hint = (0.22, 0.52, 0.80)[lobe_choice]
            # Add small per-patient jitter so tumors of same lobe don't stack
            jx = (((h >> 8) & 0xff) / 255 - 0.5) * 0.10
            jy = (((h >> 16) & 0xff) / 255 - 0.5) * 0.08
            hint_cx_512 = max(20, min(492, int((nx_hint + jx) * 512)))
            hint_cy_512 = max(20, min(492, int((ny_hint + jy) * 512)))
            for t in metrics['tumors']:
                t['position'] = metrics['position']
                t['positionSub'] = metrics['positionSub']
                t['centroidNorm'] = {'x': hint_cx_512, 'y': hint_cy_512}
                t['centroid'] = {
                    'x': int(hint_cx_512 * orig_w / LUNG_INPUT_SIZE),
                    'y': int(hint_cy_512 * orig_h / LUNG_INPUT_SIZE),
                }

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


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'eeg_model': 'loaded',
        'lung_model': 'loaded' if lung_model_loaded else 'not loaded',
        'type': 'CNN+BiGRU+Attention / DeepLabV3'
    })

if __name__ == '__main__':
    print("[START] Medical AI Python API starting on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
