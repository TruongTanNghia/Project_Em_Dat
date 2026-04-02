"""
🧠 EEG Seizure Detection - Python API Service
Load trained model (.pkl) và predict từ file .edf
Chạy song song với Node.js server trên port 5000
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import tempfile
from scipy import signal
from scipy.stats import kurtosis, skew

app = Flask(__name__)
CORS(app)

# ===== LOAD MODEL =====
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
THRESHOLD = 0.5

try:
    # Try local re-trained model first (compatible versions)
    local_model = os.path.join(MODEL_DIR, 'eeg_seizure_model_local.pkl')
    if os.path.exists(local_model):
        model = joblib.load(local_model)
        scaler = joblib.load(os.path.join(MODEL_DIR, 'eeg_scaler_local.pkl'))
        feature_names = joblib.load(os.path.join(MODEL_DIR, 'eeg_feature_names_local.pkl'))
        print(f"✅ Model loaded from LOCAL .pkl: {type(model).__name__}")
    else:
        model = joblib.load(os.path.join(MODEL_DIR, 'eeg_seizure_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'eeg_scaler.pkl'))
        feature_names = joblib.load(os.path.join(MODEL_DIR, 'eeg_feature_names.pkl'))
        print(f"✅ Model loaded from Kaggle .pkl: {type(model).__name__}")
    threshold_path = os.path.join(MODEL_DIR, 'eeg_threshold.pkl')
    if os.path.exists(threshold_path):
        THRESHOLD = joblib.load(threshold_path)
except Exception as e:
    print(f"⚠️ Cannot load .pkl ({e}). Re-training from CSV...")
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils import class_weight as cw_module
    
    csv_path = os.path.join(MODEL_DIR, 'eeg_features.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(MODEL_DIR, 'eeg_checkpoint.csv')
    
    df = pd.read_csv(csv_path)
    exclude_cols = ['label', 'patient', 'file', 'start_sec', 'end_sec', 'filename']
    feature_names = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    # Downsample for faster training
    if len(df) > 50000:
        seizure_df = df[df['label'] == 1]
        normal_df = df[df['label'] == 0].sample(n=min(50000, len(df[df['label']==0])), random_state=42)
        df = pd.concat([seizure_df, normal_df]).sample(frac=1, random_state=42)
        print(f"  Downsampled to {len(df)} rows (seizure={len(seizure_df)}, normal={len(normal_df)})")
    
    X = df[feature_names].values
    y = df['label'].values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    sample_weights = cw_module.compute_sample_weight('balanced', y)
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    model.fit(X_scaled, y, sample_weight=sample_weights)
    
    # Save re-trained model
    joblib.dump(model, os.path.join(MODEL_DIR, 'eeg_seizure_model_local.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'eeg_scaler_local.pkl'))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, 'eeg_feature_names_local.pkl'))
    print(f"✅ Re-trained and saved locally!")

print(f"✅ Features: {len(feature_names)}")
print(f"✅ Threshold: {THRESHOLD}")

# ===== EEG CONFIG =====
SAMPLING_RATE = 256
WINDOW_SIZE = 4
WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_SIZE

FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 50)
}

# ===== FEATURE EXTRACTION (giống lúc train) =====
def compute_band_power(data, sf, band):
    low, high = band
    freqs, psd = signal.welch(data, sf, nperseg=min(len(data), 256))
    idx = np.logical_and(freqs >= low, freqs <= high)
    return np.trapz(psd[idx], freqs[idx]) if np.any(idx) else 0

def extract_features_from_window(window_data, sf=256):
    n_channels = window_data.shape[0]
    features = {}
    
    all_band_powers = {band: [] for band in FREQ_BANDS}
    all_stats = {'mean': [], 'std': [], 'kurtosis': [], 'skewness': [],
                 'peak_to_peak': [], 'rms': [], 'zero_crossings': []}
    
    for ch in range(n_channels):
        ch_data = window_data[ch]
        if len(ch_data) == 0 or np.all(ch_data == 0):
            continue
        
        for band_name, band_range in FREQ_BANDS.items():
            bp = compute_band_power(ch_data, sf, band_range)
            all_band_powers[band_name].append(bp)
        
        all_stats['mean'].append(np.mean(ch_data))
        all_stats['std'].append(np.std(ch_data))
        all_stats['kurtosis'].append(kurtosis(ch_data))
        all_stats['skewness'].append(skew(ch_data))
        all_stats['peak_to_peak'].append(np.ptp(ch_data))
        all_stats['rms'].append(np.sqrt(np.mean(ch_data**2)))
        all_stats['zero_crossings'].append(np.sum(np.diff(np.sign(ch_data)) != 0))
    
    for band_name in FREQ_BANDS:
        vals = all_band_powers[band_name]
        features[f'{band_name}_power_mean'] = np.mean(vals) if vals else 0
        features[f'{band_name}_power_std'] = np.std(vals) if vals else 0
    
    d = features.get('delta_power_mean', 1e-10)
    t = features.get('theta_power_mean', 1e-10)
    a = features.get('alpha_power_mean', 1e-10)
    b = features.get('beta_power_mean', 1e-10)
    total = d + t + a + b + features.get('gamma_power_mean', 1e-10)
    
    features['theta_alpha_ratio'] = t / max(a, 1e-10)
    features['delta_alpha_ratio'] = d / max(a, 1e-10)
    features['delta_beta_ratio']  = d / max(b, 1e-10)
    features['theta_beta_ratio']  = t / max(b, 1e-10)
    
    for band_name in FREQ_BANDS:
        features[f'{band_name}_relative'] = features[f'{band_name}_power_mean'] / max(total, 1e-10)
    
    for stat_name, vals in all_stats.items():
        if vals:
            features[f'{stat_name}_mean'] = np.mean(vals)
            features[f'{stat_name}_std'] = np.std(vals)
    
    return features

# ===== API ENDPOINTS =====
@app.route('/api/predict-edf', methods=['POST'])
def predict_edf():
    """Nhận file .edf → trả về Normal/Abnormal"""
    if 'edfFile' not in request.files:
        return jsonify({'error': 'Không có file .edf'}), 400
    
    file = request.files['edfFile']
    if not file.filename.lower().endswith('.edf'):
        return jsonify({'error': 'Chỉ hỗ trợ file .edf'}), 400
    
    # Lưu file tạm
    tmp = tempfile.NamedTemporaryFile(suffix='.edf', delete=False)
    file.save(tmp.name)
    tmp.close()
    
    try:
        import pyedflib
        
        f = pyedflib.EdfReader(tmp.name)
        n_channels = f.signals_in_file
        labels = f.getSignalLabels()
        sf = f.getSampleFrequency(0)
        n_samples = f.getNSamples()[0]
        
        # Chỉ lấy kênh EEG
        eeg_idx = [i for i, lb in enumerate(labels)
                  if lb.strip() not in ['-', ''] and 'ECG' not in lb.upper() 
                  and 'VNS' not in lb.upper()][:23]
        
        if len(eeg_idx) < 3:
            f.close()
            return jsonify({'error': 'File EDF không đủ kênh EEG'}), 400
        
        data = np.array([f.readSignal(i) for i in eeg_idx])
        channel_names = [labels[i] for i in eeg_idx]
        f.close()
        
        # Cắt windows và predict
        step = WINDOW_SAMPLES
        predictions = []
        probabilities = []
        window_results = []
        
        for start in range(0, n_samples - WINDOW_SAMPLES, step):
            end = start + WINDOW_SAMPLES
            window = data[:, start:end]
            
            feats = extract_features_from_window(window, sf)
            
            # Tạo vector features đúng thứ tự
            feat_vector = np.array([feats.get(fn, 0) for fn in feature_names]).reshape(1, -1)
            feat_vector = np.nan_to_num(feat_vector, nan=0, posinf=0, neginf=0)
            feat_vector_scaled = scaler.transform(feat_vector)
            
            prob = model.predict_proba(feat_vector_scaled)[0][1]
            pred = 1 if prob >= THRESHOLD else 0
            
            predictions.append(pred)
            probabilities.append(float(prob))
            
            window_results.append({
                'startSec': round(start / sf, 1),
                'endSec': round(end / sf, 1),
                'prediction': 'Bất thường' if pred == 1 else 'Bình thường',
                'probability': round(float(prob) * 100, 2),
                'bandPowers': {
                    band: round(feats.get(f'{band}_power_mean', 0), 6)
                    for band in FREQ_BANDS
                }
            })
        
        # Tổng kết
        total_windows = len(predictions)
        abnormal_windows = sum(predictions)
        abnormal_ratio = abnormal_windows / max(total_windows, 1)
        avg_prob = np.mean(probabilities)
        
        # Quyết định tổng thể
        if abnormal_ratio > 0.1 or avg_prob > THRESHOLD:
            overall = 'BẤT THƯỜNG'
            severity = 'severe' if abnormal_ratio > 0.3 else 'moderate' if abnormal_ratio > 0.15 else 'mild'
        else:
            overall = 'BÌNH THƯỜNG'
            severity = 'normal'
        
        # Band powers trung bình
        all_feats = []
        for start in range(0, min(n_samples - WINDOW_SAMPLES, step * 10), step):
            window = data[:, start:start + WINDOW_SAMPLES]
            all_feats.append(extract_features_from_window(window, sf))
        
        avg_bands = {}
        for band in FREQ_BANDS:
            vals = [f.get(f'{band}_power_mean', 0) for f in all_feats]
            avg_bands[band] = {
                'power': round(np.mean(vals), 6),
                'relative': round(np.mean([f.get(f'{band}_relative', 0) for f in all_feats]) * 100, 2)
            }
        
        # ===== WAVEFORM DATA FOR VISUALIZATION =====
        # Downsample EEG signal cho frontend vẽ biểu đồ (max 2000 points per channel)
        max_viz_points = 2000
        viz_channels = min(6, len(eeg_idx))  # Max 6 kênh để hiển thị
        downsample_factor = max(1, n_samples // max_viz_points)
        
        waveform_data = {}
        for ch_i in range(viz_channels):
            ch_signal = data[ch_i]
            # Downsample bằng lấy trung bình mỗi block
            n_blocks = len(ch_signal) // downsample_factor
            if n_blocks > 0:
                truncated = ch_signal[:n_blocks * downsample_factor]
                downsampled = truncated.reshape(n_blocks, downsample_factor).mean(axis=1)
                waveform_data[channel_names[ch_i]] = [round(float(v), 2) for v in downsampled]
        
        # Time axis cho waveform (giây)
        waveform_time = [round(i * downsample_factor / sf, 2) for i in range(len(list(waveform_data.values())[0]))]
        
        # Abnormal regions cho highlight (start_sec, end_sec)
        abnormal_regions = [
            {'start': w['startSec'], 'end': w['endSec'], 'prob': w['probability']}
            for w in window_results if w['prediction'] == 'Bất thường'
        ]
        
        result = {
            'success': True,
            'overall': overall,
            'severity': severity,
            'confidence': round(avg_prob * 100, 2),
            'totalWindows': total_windows,
            'abnormalWindows': abnormal_windows,
            'abnormalRatio': round(abnormal_ratio * 100, 2),
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
                'name': type(model).__name__,
                'features': len(feature_names),
                'threshold': THRESHOLD
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
        'model': type(model).__name__,
        'features': len(feature_names),
        'featureNames': feature_names,
        'threshold': THRESHOLD,
        'status': 'active'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'loaded'})

if __name__ == '__main__':
    print("🧠 EEG Python API starting on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
