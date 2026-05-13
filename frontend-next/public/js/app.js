// ===== GLOBAL STATE =====
let selectedFile = null;
let currentAnalysis = null;
let chatSessionId = null;
let charts = {};
let uploadMode = 'edf'; // 'edf' or 'image'

// ===== API BASE (see js/config.js) =====
// When this frontend is deployed on Vercel and the backend lives elsewhere
// (Railway / HF Spaces / etc.), apiUrl() prepends the backend origin. When
// served by the local Node server (same origin), it's a no-op.
function apiUrl(path) {
    var base = (window.APP_CONFIG && window.APP_CONFIG.API_BASE) || '';
    if (!path.startsWith('/')) path = '/' + path;
    return base + path;
}

// ===== TAB NAVIGATION =====
function switchTab(tabName) {
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(`panel-${tabName}`).classList.add('active');
}

// ===== TOAST NOTIFICATIONS =====
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const icons = { success: '✅', error: '❌', info: 'ℹ️' };
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${icons[type]}</span> ${message}`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ===== FILE UPLOAD HANDLING =====
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

uploadZone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// ===== UPLOAD MODE SWITCHING =====
function setUploadMode(mode) {
    uploadMode = mode;
    clearUpload(new Event('click'));
    
    const edfBtn = document.getElementById('modeEdfBtn');
    const imgBtn = document.getElementById('modeImgBtn');
    const fileInput = document.getElementById('fileInput');
    const title = document.getElementById('uploadTitle');
    const hint = document.getElementById('uploadHint');
    
    if (mode === 'edf') {
        edfBtn.className = 'btn btn-primary';
        imgBtn.className = 'btn btn-secondary';
        fileInput.accept = '.edf';
        title.textContent = 'Kéo thả file .EDF vào đây';
        hint.textContent = 'Hỗ trợ: File .EDF (European Data Format)';
    } else {
        edfBtn.className = 'btn btn-secondary';
        imgBtn.className = 'btn btn-primary';
        fileInput.accept = 'image/*';
        title.textContent = 'Kéo thả ảnh EEG vào đây';
        hint.textContent = 'Hỗ trợ: JPG, PNG, BMP, WebP, TIFF (tối đa 20MB)';
    }
}

function handleFile(file) {
    const isEdf = file.name.toLowerCase().endsWith('.edf');
    const isImage = file.type.startsWith('image/');
    
    if (uploadMode === 'edf' && !isEdf) {
        showToast('Vui lòng chọn file .EDF!', 'error');
        return;
    }
    if (uploadMode === 'image' && !isImage) {
        showToast('Vui lòng chọn file ảnh!', 'error');
        return;
    }
    if (file.size > 500 * 1024 * 1024) {
        showToast('File quá lớn! Tối đa 500MB.', 'error');
        return;
    }

    selectedFile = file;
    
    // Show preview in upload zone
    document.getElementById('uploadContent').style.display = 'none';
    document.getElementById('uploadPreview').style.display = 'flex';
    document.getElementById('fileInfo').textContent = 
        `${file.name} • ${(file.size / 1024 / 1024).toFixed(2)} MB`;
    
    if (isImage) {
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('previewImg').style.display = 'block';
            document.getElementById('previewImg').src = e.target.result;
            document.getElementById('edfFileIcon').style.display = 'none';
            document.getElementById('noPreview').style.display = 'none';
            document.getElementById('mainPreviewImg').style.display = 'block';
            document.getElementById('mainPreviewImg').src = e.target.result;
        };
        reader.readAsDataURL(file);
    } else {
        // EDF file
        document.getElementById('previewImg').style.display = 'none';
        document.getElementById('edfFileIcon').style.display = 'block';
    }
    
    document.getElementById('analyzeBtn').disabled = false;
    showToast(`Đã chọn file ${isEdf ? '.EDF' : 'ảnh'} thành công!`, 'success');
}

function clearUpload(e) {
    if (e && e.stopPropagation) e.stopPropagation();
    selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('uploadContent').style.display = '';
    document.getElementById('uploadPreview').style.display = 'none';
    document.getElementById('noPreview').style.display = '';
    document.getElementById('mainPreviewImg').style.display = 'none';
    document.getElementById('edfPredictionResult').style.display = 'none';
    document.getElementById('analyzeBtn').disabled = true;
    document.getElementById('quickStats').style.display = 'none';
    document.getElementById('previewImg').style.display = 'none';
    document.getElementById('edfFileIcon').style.display = 'none';
}

// ===== EEG ANALYSIS =====
async function analyzeEEG() {
    if (!selectedFile) {
        showToast('Vui lòng chọn file trước!', 'error');
        return;
    }

    const isEdf = selectedFile.name.toLowerCase().endsWith('.edf');
    
    if (isEdf) {
        await analyzeEDF();
    } else {
        await analyzeImage();
    }
}

// === ANALYZE EDF FILE (Python ML Model) ===
async function analyzeEDF() {
    const overlay = document.getElementById('loadingOverlay');
    document.getElementById('loadingText').textContent = '🧠 Đang phân tích file EDF bằng AI Model...';
    overlay.classList.add('active');

    try {
        const formData = new FormData();
        formData.append('edfFile', selectedFile);

        const response = await fetch(apiUrl('/api/predict-edf'), {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Lỗi phân tích EDF');
        }

        const data = await response.json();
        displayEdfResults(data);
        
        // Also render charts from EDF band powers
        if (data.frequencyBands) {
            renderChartsFromEdf(data);
        }
        
        currentAnalysis = data;
        updateChatContext({ overallAssessment: data.overall, edfPrediction: data });
        
        showToast('Phân tích EDF hoàn tất! 🎉', 'success');
        addMcpLog(`EDF analyzed: ${data.overall} (${data.confidence}%)`);
    } catch (error) {
        console.error('EDF error:', error);
        showToast(`Lỗi: ${error.message}`, 'error');
    } finally {
        overlay.classList.remove('active');
    }
}
function displayEdfResults(data) {
    // Show prediction result card
    document.getElementById('noPreview').style.display = 'none';
    document.getElementById('mainPreviewImg').style.display = 'none';
    const predDiv = document.getElementById('edfPredictionResult');
    predDiv.style.display = 'block';
    
    // 4-tier severity mapping
    const severityMap = {
        'normal':   { icon: '✅', color: '#10b981', label: 'BÌNH THƯỜNG' },
        'mild':     { icon: '⚠️', color: '#f59e0b', label: 'CẢNH BÁO NHẸ' },
        'moderate': { icon: '🟠', color: '#f97316', label: 'BẤT THƯỜNG' },
        'severe':   { icon: '🔴', color: '#ef4444', label: 'NGHIÊM TRỌNG' }
    };
    const sev = severityMap[data.severity] || severityMap['normal'];
    const hasAnyIssue = data.severity !== 'normal';
    
    document.getElementById('predictionIcon').textContent = sev.icon;
    document.getElementById('predictionLabel').textContent = sev.label;
    document.getElementById('predictionLabel').style.color = sev.color;
    document.getElementById('predictionConfidence').textContent = 
        `Độ tin cậy: ${data.confidence}% | ${data.abnormalRatio}% windows bất thường`;
    document.getElementById('predDuration').textContent = `${data.duration}s`;
    document.getElementById('predChannels').textContent = data.channels;
    document.getElementById('predAbnormal').textContent = `${data.abnormalWindows}/${data.totalWindows}`;
    
    // Show results section with EDF details
    const section = document.getElementById('resultsSection');
    section.classList.add('active');
    document.getElementById('analysisTime').textContent = 
        `Phân tích lúc ${new Date().toLocaleString('vi-VN')} | Model: ${data.modelInfo?.name || 'ML'}`;
    
    const badge = document.getElementById('assessmentBadge');
    badge.className = `assessment-badge ${data.severity}`;
    badge.textContent = sev.label;
    
    // Frequency bands from model
    const bands = data.frequencyBands || {};
    const bandInfo = {
        delta: { label: 'Delta', range: '0.5-4 Hz', class: 'delta' },
        theta: { label: 'Theta', range: '4-8 Hz', class: 'theta' },
        alpha: { label: 'Alpha', range: '8-13 Hz', class: 'alpha' },
        beta: { label: 'Beta', range: '13-30 Hz', class: 'beta' },
        gamma: { label: 'Gamma', range: '30-50 Hz', class: 'gamma' }
    };
    let statsHtml = '';
    for (const [key, info] of Object.entries(bandInfo)) {
        const band = bands[key] || { power: 0, relative: 0 };
        statsHtml += `
            <div class="glass-card stat-card ${info.class}">
                <div class="stat-label">${info.label}</div>
                <div class="stat-value">${band.relative}%</div>
                <div class="stat-range">${info.range}</div>
                <span class="stat-status normal">Power: ${band.power.toFixed(4)}</span>
            </div>`;
    }
    document.getElementById('freqStats').innerHTML = statsHtml;
    
    // Summary
    document.getElementById('summaryText').textContent = 
        `Kết quả phân tích: ${data.overall}. ` +
        `Đã phân tích ${data.totalWindows} đoạn tín hiệu EEG (mỗi đoạn 4 giây) từ ${data.channels} kênh. ` +
        `Phát hiện ${data.abnormalWindows} đoạn bất thường (${data.abnormalRatio}%).`;
    
    // Window results as findings — 4-tier labels
    let findingsHtml = '';
    const iconMap = { severe: '🔴', moderate: '🟠', mild: '⚠️', normal: '✅' };
    const labelMap = { severe: 'Nguy hiểm', moderate: 'Bất thường', mild: 'Nghi ngờ', normal: 'Bình thường' };
    
    // Filter non-normal windows (fallback: use probability if severity missing)
    const allWindows = data.windowResults || [];
    console.log(`[EEG] windowResults: ${allWindows.length} total, sample:`, allWindows[0]);
    const flaggedWindows = allWindows.filter(w => 
        w.severity ? w.severity !== 'normal' : (w.probability >= 20)
    );
    
    if (flaggedWindows.length > 0) {
        // Severity breakdown summary
        const sc = data.severityCounts || {};
        findingsHtml += `<div style="margin-bottom:12px; padding:8px 12px; border-radius:8px; background:rgba(255,255,255,0.03); font-size:12px; color:var(--text-secondary);">
            📊 Phân bố: 
            <span style="color:#10b981">✅ ${sc.normal || 0} bình thường</span> · 
            <span style="color:#f59e0b">⚠️ ${sc.mild || 0} nghi ngờ</span> · 
            <span style="color:#f97316">🟠 ${sc.moderate || 0} bất thường</span> · 
            <span style="color:#ef4444">🔴 ${sc.severe || 0} nguy hiểm</span>
        </div>`;
        
        // Show top 10 flagged windows
        flaggedWindows.slice(0, 10).forEach(w => {
            const sevIcon = iconMap[w.severity] || '⚠️';
            const sevLabel = labelMap[w.severity] || w.prediction;
            const sevColor = w.color || '#f59e0b';
            findingsHtml += `
                <div class="finding-item">
                    <div class="finding-header">
                        <span class="finding-title">${sevIcon} ${sevLabel} tại ${w.startSec}s - ${w.endSec}s</span>
                        <span class="severity-tag ${w.severity}" style="background:${sevColor}; color:white; padding:2px 8px; border-radius:4px; font-size:11px;">${w.probability}%</span>
                    </div>
                    <div class="finding-location" style="font-size:12px; color:var(--text-muted);">Xác suất: ${w.probability}% · Mức: ${sevLabel}</div>
                </div>`;
        });
        if (flaggedWindows.length > 10) {
            findingsHtml += `<p style="color:var(--text-muted); font-size:12px;">... và ${flaggedWindows.length - 10} đoạn khác</p>`;
        }
    } else {
        findingsHtml = '<p style="color:var(--text-muted)">Không phát hiện đoạn bất thường nào.</p>';
    }
    document.getElementById('findingsList').innerHTML = findingsHtml;
    
    const hasAbnormal = data.abnormalWindows > 0;
    const sevDesc = data.severityDescription || '';
    
    // Abnormalities section — show details if ANY abnormal windows exist
    if (hasAbnormal) {
        const sevColors = { normal: '#10b981', mild: '#f59e0b', moderate: '#f97316', severe: '#ef4444' };
        const sevColor = sevColors[data.severity] || '#f59e0b';
        document.getElementById('abnormalitiesList').innerHTML = `
            <div class="finding-item">
                <div class="finding-header">
                    <span class="finding-title" style="color:${sevColor}">${sev.icon} ${sev.label}: ${data.abnormalWindows} đoạn bất thường</span>
                </div>
                <p style="font-size:13px; color:var(--text-secondary); margin-top:4px;">
                    Tỉ lệ: ${data.abnormalRatio}% windows (${data.abnormalWindows}/${data.totalWindows}).<br>
                    ${data.severity === 'normal' ? 'Trong giới hạn bình thường.' 
                      : data.severity === 'mild' ? 'Cần theo dõi thêm, đối chiếu lâm sàng.'
                      : data.severity === 'moderate' ? 'Phát hiện bất thường — cần khám chuyên khoa thần kinh.'
                      : 'Mức độ nghiêm trọng — cần xử lý y khoa kịp thời.'}
                </p>
            </div>`;
    } else {
        document.getElementById('abnormalitiesList').innerHTML = 
            '<p style="color:var(--text-muted)">Không phát hiện bất thường.</p>';
    }
    
    // Recommendations — adapt per severity
    const recMap = {
        normal: 'EEG trong giới hạn bình thường, theo dõi định kỳ',
        mild: 'Phát hiện một số đoạn nghi ngờ — nên tái khám và theo dõi EEG',
        moderate: 'Cần tham khảo ý kiến chuyên gia thần kinh sớm',
        severe: 'CẦN XỬ LÝ Y KHOA NGAY — liên hệ chuyên gia thần kinh'
    };
    document.getElementById('recommendationsList').innerHTML = `
        <div class="recommendation-item"><span class="num">1</span><span>Đối chiếu kết quả với triệu chứng lâm sàng</span></div>
        <div class="recommendation-item"><span class="num">2</span><span>${recMap[data.severity] || recMap.normal}</span></div>
        <div class="recommendation-item"><span class="num">3</span><span>Sử dụng chatbot để được giải thích chi tiết kết quả</span></div>`;
    
    document.getElementById('detailedText').textContent = 
        `File EDF: ${selectedFile.name}, Thời lượng: ${data.duration}s, ` +
        `Kênh: ${(data.channelNames || []).join(', ')}, ` +
        `Tần số lấy mẫu: ${data.samplingRate}Hz. ` +
        `Model: ${data.modelInfo?.name || 'CNN'} (${data.modelInfo?.type || 'Deep Learning'}), ` +
        `Bandpass: ${data.modelInfo?.bandpass || '0.5-40Hz'}, Threshold: ${data.threshold}.`;
    
    // Clinical text — severity-aware
    const clinicalMap = {
        normal: 'Không phát hiện dấu hiệu bất thường đáng kể trên EEG. Hoạt động não trong giới hạn bình thường.',
        mild: `Phát hiện ${data.abnormalWindows} đoạn EEG nghi ngờ bất thường (${data.abnormalRatio}%). Mức cảnh báo nhẹ — cần theo dõi và tái khám.`,
        moderate: `Phát hiện ${data.abnormalWindows} đoạn EEG bất thường rõ ràng (${data.abnormalRatio}%). Có thể liên quan đến hoạt động co giật. Cần khám chuyên khoa thần kinh.`,
        severe: `Phát hiện ${data.abnormalWindows} đoạn EEG nghiêm trọng (${data.abnormalRatio}%). Nghi ngờ cơn co giật (seizure). CẦN XỬ LÝ Y KHOA NGAY.`
    };
    document.getElementById('clinicalText').textContent = clinicalMap[data.severity] || clinicalMap.normal;
    
    // Quick stats
    const qs = document.getElementById('quickStats');
    qs.style.display = 'block';
    document.getElementById('quickStatsContent').innerHTML = `
        <p style="font-size:13px; color:var(--text-secondary); line-height:1.8;">
            <strong style="color:var(--text-primary);">Kết quả:</strong> ${data.overall}<br>
            <strong style="color:var(--text-primary);">Độ tin cậy:</strong> ${data.confidence}%<br>
            <strong style="color:var(--text-primary);">Windows:</strong> ${data.abnormalWindows}/${data.totalWindows} bất thường<br>
            <strong style="color:var(--text-primary);">Model:</strong> ${data.modelInfo?.name || 'ML'}
        </p>`;
    
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderChartsFromEdf(data) {
    document.getElementById('chartsPlaceholder').style.display = 'none';
    document.getElementById('chartsContent').style.display = 'block';
    Object.values(charts).forEach(c => c.destroy());
    charts = {};
    
    const bands = data.frequencyBands || {};
    const labels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'];
    const values = [bands.delta?.relative||0, bands.theta?.relative||0, bands.alpha?.relative||0, bands.beta?.relative||0, bands.gamma?.relative||0];
    const colors = ['#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#ec4899'];
    const commonOpts = { responsive:true, maintainAspectRatio:false, plugins:{legend:{labels:{color:'#94a3b8',font:{family:'Inter',size:12}}}} };
    
    // ===== 1. EEG WAVEFORM CHART (Multi-channel with abnormal highlights) =====
    const waveform = data.waveform;
    if (waveform && waveform.time && waveform.channels) {
        const channelNames = Object.keys(waveform.channels);
        // 23-color palette for all EEG channels
        const chColors = [
            '#3b82f6','#10b981','#f59e0b','#ec4899','#8b5cf6','#06b6d4',
            '#ef4444','#84cc16','#f97316','#6366f1','#14b8a6','#d946ef',
            '#e11d48','#0ea5e9','#a855f7','#22c55e','#eab308','#f43f5e',
            '#2dd4bf','#fb923c','#818cf8','#4ade80','#facc15'
        ];
        const chSpacing = 4; // compact spacing for 23 channels
        
        // Create datasets - offset each channel vertically
        const datasets = [];
        channelNames.forEach((chName, idx) => {
            const rawData = waveform.channels[chName];
            const mean = rawData.reduce((a,b) => a+b, 0) / rawData.length;
            const std = Math.sqrt(rawData.reduce((a,b) => a + (b-mean)**2, 0) / rawData.length) || 1;
            const normalized = rawData.map(v => ((v - mean) / std) + (channelNames.length - idx) * chSpacing);
            
            datasets.push({
                label: chName,
                data: normalized,
                borderColor: chColors[idx % chColors.length],
                borderWidth: 0.5,
                pointRadius: 0,
                fill: false,
                tension: 0
            });
        });
        
        // Annotation boxes for abnormal regions (only moderate + severe from API)
        const annotations = {};
        const sevColorMap = {
            severe:   { bg: 'rgba(239, 68, 68, 0.25)', border: 'rgba(239, 68, 68, 0.6)' },
            moderate: { bg: 'rgba(249, 115, 22, 0.15)', border: 'rgba(249, 115, 22, 0.4)' }
        };
        (data.abnormalRegions || []).forEach((region, idx) => {
            const duration = data.duration || 3600;
            const totalPoints = waveform.time.length;
            const xMin = (region.start / duration) * totalPoints;
            const xMax = (region.end / duration) * totalPoints;
            const sColors = sevColorMap[region.severity] || sevColorMap.moderate;
            
            annotations[`box${idx}`] = {
                type: 'box',
                xMin: Math.floor(xMin),
                xMax: Math.ceil(xMax),
                backgroundColor: sColors.bg,
                borderColor: sColors.border,
                borderWidth: 1,
                label: {
                    display: region.prob >= 70, // Only label severe
                    content: `${region.prob}%`,
                    position: 'start',
                    font: { size: 8, weight: 'bold' },
                    color: '#fff'
                }
            };
        });
        
        // Time labels
        const timeLabels = waveform.time.map(t => {
            const mins = Math.floor(t / 60);
            const secs = Math.floor(t % 60);
            return `${mins}:${String(secs).padStart(2, '0')}`;
        });
        
        // Channel label plugin
        const channelLabelPlugin = {
            id: 'channelLabels',
            afterDatasetsDraw(chart) {
                const ctx = chart.ctx;
                const yScale = chart.scales.y;
                ctx.save();
                ctx.font = 'bold 9px Inter, sans-serif';
                ctx.textAlign = 'right';
                channelNames.forEach((name, idx) => {
                    const yVal = (channelNames.length - idx) * chSpacing;
                    const yPx = yScale.getPixelForValue(yVal);
                    ctx.fillStyle = chColors[idx % chColors.length];
                    ctx.fillText(name, chart.chartArea.left - 4, yPx + 3);
                });
                ctx.restore();
            }
        };
        
        // Set canvas height for 23 channels
        const canvas = document.getElementById('eegWaveformChart');
        canvas.parentElement.style.height = `${Math.max(500, channelNames.length * 28)}px`;
        
        charts.waveform = new Chart(canvas, {
            type: 'line',
            data: { labels: timeLabels, datasets },
            plugins: [channelLabelPlugin],
            options: {
                ...commonOpts,
                animation: false,
                interaction: { mode: 'index', intersect: false },
                layout: { padding: { left: 60 } },
                plugins: {
                    legend: { display: false }, // Too many channels for legend
                    annotation: { annotations },
                    tooltip: {
                        enabled: true,
                        mode: 'nearest',
                        intersect: false,
                        callbacks: {
                            title: (items) => `Thời điểm: ${items[0].label}`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: '#64748b', maxTicksLimit: 20, font: { size: 10 } },
                        title: { display: true, text: 'Thời gian (phút:giây)', color: '#94a3b8', font: { size: 12 } }
                    },
                    y: {
                        grid: { color: 'rgba(255,255,255,0.03)' },
                        ticks: { display: false },
                        title: { display: true, text: `${channelNames.length} kênh EEG`, color: '#94a3b8', font: { size: 12 } }
                    }
                }
            }
        });
    }
    
    // ===== 2. ANOMALY PROBABILITY TIMELINE =====
    const winResults = data.windowResults || [];
    if (winResults.length > 0) {
        const winLabels = winResults.map(w => {
            const mins = Math.floor(w.startSec / 60);
            const secs = Math.floor(w.startSec % 60);
            return `${mins}:${String(secs).padStart(2, '0')}`;
        });
        const winProbs = winResults.map(w => w.probability);
        // 4-tier color for each bar
        const winColors = winProbs.map(p => {
            if (p >= 70) return 'rgba(239,68,68,0.9)';   // severe - red
            if (p >= 40) return 'rgba(249,115,22,0.8)';   // moderate - orange
            if (p >= 20) return 'rgba(245,158,11,0.6)';   // mild - yellow
            return 'rgba(16,185,129,0.3)';                 // normal - green
        });
        
        charts.anomaly = new Chart(document.getElementById('anomalyTimelineChart'), {
            type: 'bar',
            data: {
                labels: winLabels,
                datasets: [{
                    label: 'Xác suất bất thường (%)',
                    data: winProbs,
                    backgroundColor: winColors,
                    borderColor: winColors.map(c => c.replace('0.8', '1').replace('0.5', '0.8')),
                    borderWidth: 1,
                    borderRadius: 2,
                    barPercentage: 1.0,
                    categoryPercentage: 1.0
                }]
            },
            options: {
                ...commonOpts,
                plugins: {
                    legend: { display: false },
                    annotation: {
                        annotations: {
                            thresholdLine: {
                                type: 'line',
                                yMin: (data.threshold || 0.5) * 100,
                                yMax: (data.threshold || 0.5) * 100,
                                borderColor: '#f59e0b',
                                borderWidth: 2,
                                borderDash: [8, 4],
                                label: {
                                    display: true,
                                    content: `Threshold ${((data.threshold || 0.5) * 100).toFixed(0)}%`,
                                    position: 'end',
                                    backgroundColor: 'rgba(245,158,11,0.8)',
                                    color: '#fff',
                                    font: { size: 11, weight: 'bold' }
                                }
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            afterBody: (items) => {
                                const idx = items[0].dataIndex;
                                const w = winResults[idx];
                                return w ? `Kết quả: ${w.prediction}` : '';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: '#64748b', maxTicksLimit: 20, font: { size: 10 } },
                        title: { display: true, text: 'Thời gian', color: '#94a3b8' }
                    },
                    y: {
                        beginAtZero: true, max: 100,
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#94a3b8', callback: v => v + '%' },
                        title: { display: true, text: 'Xác suất (%)', color: '#94a3b8' }
                    }
                }
            }
        });
    }
    
    // ===== 3. BAND POWER CHARTS =====
    const commonOpts2 = { responsive:true, maintainAspectRatio:true, plugins:{legend:{labels:{color:'#94a3b8',font:{family:'Inter',size:12}}}} };
    
    charts.bar = new Chart(document.getElementById('freqBarChart'), { type:'bar', data:{ labels, datasets:[{label:'Relative Power (%)',data:values,backgroundColor:colors.map(c=>c+'99'),borderColor:colors,borderWidth:2,borderRadius:8}]}, options:{...commonOpts2,scales:{y:{beginAtZero:true,grid:{color:'rgba(255,255,255,0.05)'},ticks:{color:'#94a3b8'}},x:{grid:{display:false},ticks:{color:'#94a3b8',font:{weight:'600'}}}}} });
    charts.radar = new Chart(document.getElementById('radarChart'), { type:'radar', data:{ labels, datasets:[{label:'Sóng não',data:values,fill:true,backgroundColor:'rgba(59,130,246,0.15)',borderColor:'#3b82f6',pointBackgroundColor:colors,pointRadius:5}]}, options:{...commonOpts2,scales:{r:{beginAtZero:true,grid:{color:'rgba(255,255,255,0.06)'},pointLabels:{color:'#f1f5f9',font:{size:13,weight:'600'}},ticks:{display:false}}}} });
    charts.pie = new Chart(document.getElementById('pieChart'), { type:'doughnut', data:{ labels, datasets:[{data:values,backgroundColor:colors.map(c=>c+'cc'),borderColor:colors,borderWidth:2}]}, options:{...commonOpts2,cutout:'55%',plugins:{legend:{position:'bottom',labels:{color:'#94a3b8',padding:16}}}} });
    
    // Band powers per window (heatmap style)
    const regionLabels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'];
    const regionDatasets = regionLabels.map((band, idx) => ({
        label: band,
        data: winResults.slice(0, 100).map(w => w.bandPowers?.[band.toLowerCase()] || 0),
        borderColor: colors[idx],
        backgroundColor: colors[idx] + '22',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        fill: true
    }));
    charts.region = new Chart(document.getElementById('regionChart'), {
        type: 'line',
        data: { labels: winResults.slice(0, 100).map(w => `${w.startSec}s`), datasets: regionDatasets },
        options: { ...commonOpts2, scales: { y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }, x: { grid: { display: false }, ticks: { color: '#94a3b8', maxTicksLimit: 10 } } } }
    });
    
    // Auto switch to charts tab
    switchTab('charts');
    
    // Initialize and render 3D EEG visualization
    // Need extra delay so the charts tab is fully visible & container has real dimensions
    const start3D = () => {
        const c = document.getElementById('eeg3dContainer');
        if (c && c.clientWidth > 0 && c.clientHeight > 0) {
            init3DScene();
            render3DEEG(data);
        } else {
            console.log('3D container not ready, retrying in 500ms...');
            setTimeout(start3D, 500);
        }
    };
    setTimeout(start3D, 1000);
}

// === ANALYZE IMAGE (GPT-4 Vision) ===
async function analyzeImage() {
    const overlay = document.getElementById('loadingOverlay');
    document.getElementById('loadingText').textContent = '🧠 Đang phân tích ảnh EEG bằng GPT-4...';
    overlay.classList.add('active');

    try {
        const formData = new FormData();
        formData.append('eegImage', selectedFile);

        const response = await fetch(apiUrl('/api/analyze'), {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Lỗi phân tích');
        }

        const data = await response.json();
        currentAnalysis = data.analysis;
        
        displayResults(data.analysis);
        renderCharts(data.analysis);
        updateChatContext(data.analysis);
        
        showToast('Phân tích hoàn tất! 🎉', 'success');
        addMcpLog('Phân tích EEG hoàn tất - ID: ' + data.id);
    } catch (error) {
        console.error('Analysis error:', error);
        showToast(`Lỗi: ${error.message}`, 'error');
    } finally {
        overlay.classList.remove('active');
    }
}

// ===== DISPLAY RESULTS =====
function displayResults(analysis) {
    const section = document.getElementById('resultsSection');
    section.classList.add('active');

    // Time
    document.getElementById('analysisTime').textContent = 
        `Phân tích lúc ${new Date().toLocaleString('vi-VN')}`;

    // Assessment Badge
    const badge = document.getElementById('assessmentBadge');
    const assessment = (analysis.overallAssessment || '').toLowerCase();
    let badgeClass = 'normal';
    if (assessment.includes('nặng') || assessment.includes('severe')) badgeClass = 'severe';
    else if (assessment.includes('trung bình') || assessment.includes('moderate')) badgeClass = 'moderate';
    else if (assessment.includes('nhẹ') || assessment.includes('mild')) badgeClass = 'mild';
    badge.className = `assessment-badge ${badgeClass}`;
    badge.textContent = analysis.overallAssessment || 'Đang đánh giá';

    // Frequency Band Stats
    const bands = analysis.frequencyBands || {};
    const bandInfo = {
        delta: { label: 'Delta', range: '0.5-4 Hz', class: 'delta' },
        theta: { label: 'Theta', range: '4-8 Hz', class: 'theta' },
        alpha: { label: 'Alpha', range: '8-13 Hz', class: 'alpha' },
        beta: { label: 'Beta', range: '13-30 Hz', class: 'beta' },
        gamma: { label: 'Gamma', range: '30-100 Hz', class: 'gamma' }
    };

    let statsHtml = '';
    for (const [key, info] of Object.entries(bandInfo)) {
        const band = bands[key] || { power: 0, status: 'normal' };
        statsHtml += `
            <div class="glass-card stat-card ${info.class}">
                <div class="stat-label">${info.label}</div>
                <div class="stat-value">${band.power}%</div>
                <div class="stat-range">${info.range}</div>
                <span class="stat-status ${band.status}">${band.status === 'normal' ? '✓ Bình thường' : '⚠ Bất thường'}</span>
            </div>`;
    }
    document.getElementById('freqStats').innerHTML = statsHtml;

    // Summary
    document.getElementById('summaryText').textContent = analysis.patientSummary || 'Không có tóm tắt.';

    // Findings
    const findings = analysis.findings || [];
    let findingsHtml = '';
    findings.forEach(f => {
        findingsHtml += `
            <div class="finding-item">
                <div class="finding-header">
                    <span class="finding-title">${f.finding || f}</span>
                    <span class="severity-tag ${f.severity || 'normal'}">${f.severity || 'N/A'}</span>
                </div>
                ${f.location ? `<div class="finding-location">📍 ${f.location}</div>` : ''}
            </div>`;
    });
    document.getElementById('findingsList').innerHTML = findingsHtml || '<p style="color:var(--text-muted)">Không có phát hiện đặc biệt.</p>';

    // Abnormalities
    const abnormalities = analysis.abnormalities || [];
    let abHtml = '';
    abnormalities.forEach(a => {
        abHtml += `
            <div class="finding-item">
                <div class="finding-header">
                    <span class="finding-title">${a.type}</span>
                </div>
                <p style="font-size:13px; color:var(--text-secondary); margin-top:4px;">${a.description}</p>
                ${a.clinicalSignificance ? `<p style="font-size:12px; color:var(--accent-amber); margin-top:6px;">⚕️ ${a.clinicalSignificance}</p>` : ''}
            </div>`;
    });
    document.getElementById('abnormalitiesList').innerHTML = abHtml || '<p style="color:var(--text-muted)">Không phát hiện bất thường.</p>';

    // Recommendations
    const recs = analysis.recommendations || [];
    let recsHtml = '';
    recs.forEach((r, i) => {
        recsHtml += `
            <div class="recommendation-item">
                <span class="num">${i + 1}</span>
                <span>${r}</span>
            </div>`;
    });
    document.getElementById('recommendationsList').innerHTML = recsHtml || '<p style="color:var(--text-muted)">Không có khuyến nghị.</p>';

    // Detailed Analysis
    document.getElementById('detailedText').textContent = analysis.detailedAnalysis || 'Chưa có phân tích chi tiết.';

    // Clinical Correlation
    document.getElementById('clinicalText').textContent = analysis.clinicalCorrelation || 'Cần thêm thông tin lâm sàng.';

    // Quick stats card
    const qs = document.getElementById('quickStats');
    qs.style.display = 'block';
    document.getElementById('quickStatsContent').innerHTML = `
        <p style="font-size:13px; color:var(--text-secondary); line-height:1.8;">
            <strong style="color:var(--text-primary);">Đánh giá:</strong> ${analysis.overallAssessment || 'N/A'}<br>
            <strong style="color:var(--text-primary);">Phát hiện:</strong> ${findings.length} mục<br>
            <strong style="color:var(--text-primary);">Bất thường:</strong> ${abnormalities.length} mục<br>
            <strong style="color:var(--text-primary);">Khuyến nghị:</strong> ${recs.length} mục
        </p>`;

    // Scroll to results
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ===== CHART RENDERING =====
function renderCharts(analysis) {
    document.getElementById('chartsPlaceholder').style.display = 'none';
    document.getElementById('chartsContent').style.display = 'block';

    // Destroy old charts
    Object.values(charts).forEach(c => c.destroy());
    charts = {};

    const bands = analysis.frequencyBands || {};
    const labels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'];
    const values = [
        bands.delta?.power || 0,
        bands.theta?.power || 0,
        bands.alpha?.power || 0,
        bands.beta?.power || 0,
        bands.gamma?.power || 0
    ];

    const colors = ['#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#ec4899'];
    const colorsAlpha = colors.map(c => c + '33');

    // Common chart options
    const commonOpts = {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            legend: { labels: { color: '#94a3b8', font: { family: 'Inter', size: 12 } } }
        }
    };

    // 1. Bar Chart
    charts.bar = new Chart(document.getElementById('freqBarChart'), {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Năng lượng (%)',
                data: values,
                backgroundColor: colors.map(c => c + '99'),
                borderColor: colors,
                borderWidth: 2,
                borderRadius: 8,
                borderSkipped: false
            }]
        },
        options: {
            ...commonOpts,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8', font: { family: 'Inter' } }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8', font: { family: 'Inter', weight: '600' } }
                }
            }
        }
    });

    // 2. Radar Chart
    charts.radar = new Chart(document.getElementById('radarChart'), {
        type: 'radar',
        data: {
            labels,
            datasets: [{
                label: 'Năng lượng sóng não',
                data: values,
                fill: true,
                backgroundColor: 'rgba(59, 130, 246, 0.15)',
                borderColor: '#3b82f6',
                pointBackgroundColor: colors,
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 5
            }]
        },
        options: {
            ...commonOpts,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    angleLines: { color: 'rgba(255,255,255,0.06)' },
                    pointLabels: { color: '#f1f5f9', font: { family: 'Inter', size: 13, weight: '600' } },
                    ticks: { display: false }
                }
            }
        }
    });

    // 3. Pie/Doughnut Chart
    charts.pie = new Chart(document.getElementById('pieChart'), {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: colors.map(c => c + 'cc'),
                borderColor: colors,
                borderWidth: 2,
                hoverOffset: 8
            }]
        },
        options: {
            ...commonOpts,
            cutout: '55%',
            plugins: {
                ...commonOpts.plugins,
                legend: { position: 'bottom', labels: { color: '#94a3b8', padding: 16, font: { family: 'Inter', size: 12 } } }
            }
        }
    });

    // 4. Brain Regions Chart
    const regions = analysis.brainRegions || {};
    const regionLabels = ['Trán (Frontal)', 'Thái dương (Temporal)', 'Đỉnh (Parietal)', 'Chẩm (Occipital)', 'Trung tâm (Central)'];
    const regionValues = [
        regions.frontal?.activity || 0,
        regions.temporal?.activity || 0,
        regions.parietal?.activity || 0,
        regions.occipital?.activity || 0,
        regions.central?.activity || 0
    ];
    const regionColors = ['#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4'];

    charts.region = new Chart(document.getElementById('regionChart'), {
        type: 'polarArea',
        data: {
            labels: regionLabels,
            datasets: [{
                data: regionValues,
                backgroundColor: regionColors.map(c => c + '88'),
                borderColor: regionColors,
                borderWidth: 2
            }]
        },
        options: {
            ...commonOpts,
            scales: {
                r: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    ticks: { display: false }
                }
            },
            plugins: {
                ...commonOpts.plugins,
                legend: { position: 'bottom', labels: { color: '#94a3b8', padding: 12, font: { family: 'Inter', size: 11 } } }
            }
        }
    });
}

// ===== CHATBOT =====
async function sendChat() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    if (!message) return;

    input.value = '';
    appendMessage('user', message);
    appendTypingIndicator();

    try {
        const body = {
            message,
            sessionId: chatSessionId
        };

        // Include analysis context if available
        if (currentAnalysis && !chatSessionId) {
            body.analysisContext = currentAnalysis;
        }

        const response = await fetch(apiUrl('/api/chat'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        removeTypingIndicator();

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Lỗi chat');
        }

        const data = await response.json();
        chatSessionId = data.sessionId;
        appendMessage('assistant', data.reply);

    } catch (error) {
        removeTypingIndicator();
        appendMessage('assistant', `❌ Lỗi: ${error.message}`);
    }
}

function quickChat(message) {
    document.getElementById('chatInput').value = message;
    sendChat();
}

function appendMessage(role, content) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = `message ${role}`;

    const avatar = role === 'assistant' ? '🧠' : '👤';
    
    // Render markdown for assistant messages
    let rendered = content;
    if (role === 'assistant' && typeof marked !== 'undefined') {
        try {
            rendered = marked.parse(content);
        } catch(e) {
            rendered = content.replace(/\n/g, '<br>');
        }
    } else {
        rendered = content.replace(/\n/g, '<br>');
    }

    div.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">${rendered}</div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function appendTypingIndicator() {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.id = 'typingIndicator';
    div.innerHTML = `
        <div class="message-avatar">🧠</div>
        <div class="message-content">
            <div class="typing-indicator"><span></span><span></span><span></span></div>
        </div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function removeTypingIndicator() {
    const el = document.getElementById('typingIndicator');
    if (el) el.remove();
}

function updateChatContext(analysis) {
    const ctx = document.getElementById('chatAnalysisContext');
    ctx.style.display = 'block';
    document.getElementById('chatContextText').textContent = 
        `Đã liên kết kết quả phân tích EEG (${analysis.overallAssessment || 'N/A'}). Chatbot sẽ sử dụng context này để trả lời chính xác hơn.`;
}

// ===== MCP SERVER =====
async function checkMcpStatus() {
    try {
        const response = await fetch(apiUrl('/api/mcp/status'));
        const data = await response.json();

        // Header status
        document.getElementById('mcpDot').classList.remove('offline');
        document.getElementById('mcpStatusText').textContent = 'MCP Active';

        // Dashboard
        document.getElementById('mcpDashDot').classList.remove('offline');
        document.getElementById('mcpDashStatus').textContent = '🟢 Đang hoạt động';
        document.getElementById('mcpServerName').textContent = data.serverName;
        document.getElementById('mcpVersion').textContent = data.version;
        document.getElementById('mcpAnalysisCount').textContent = data.analysisCount;
        document.getElementById('mcpUptime').textContent = formatUptime(data.uptime);

        // Tools
        renderMcpTools(data.tools);
        addMcpLog('Kết nối MCP Server thành công');
        showToast('MCP Server đang hoạt động!', 'success');

    } catch (error) {
        document.getElementById('mcpDot').classList.add('offline');
        document.getElementById('mcpStatusText').textContent = 'MCP Offline';
        document.getElementById('mcpDashDot').classList.add('offline');
        document.getElementById('mcpDashStatus').textContent = '🔴 Offline';
        addMcpLog('LỖI kết nối MCP: ' + error.message);
        showToast('Không thể kết nối MCP Server', 'error');
    }
}

function renderMcpTools(tools) {
    const grid = document.getElementById('mcpToolsGrid');
    grid.innerHTML = '';

    const toolIcons = {
        'analyze_eeg': '🔬',
        'generate_report': '📄',
        'get_analysis_history': '📜'
    };

    tools.forEach(tool => {
        const card = document.createElement('div');
        card.className = 'glass-card mcp-tool-card';
        card.innerHTML = `
            <div class="mcp-tool-name">${toolIcons[tool.name] || '🛠️'} ${tool.name}</div>
            <div class="mcp-tool-desc">${tool.description}</div>
            <button class="mcp-tool-btn" onclick="executeMcpTool('${tool.name}')">
                ▶ Thực thi
            </button>`;
        grid.appendChild(card);
    });
}

async function executeMcpTool(toolName) {
    addMcpLog(`Thực thi tool: ${toolName}...`);
    
    try {
        let params = {};
        
        if (toolName === 'get_analysis_history') {
            params = { limit: 10 };
        } else if (toolName === 'generate_report' || toolName === 'analyze_eeg') {
            // Use latest analysis if available
            if (currentAnalysis) {
                const histRes = await fetch(apiUrl('/api/history'));
                const histData = await histRes.json();
                if (histData.history.length > 0) {
                    params = { analysisId: histData.history[0].id, imageId: histData.history[0].id };
                }
            } else {
                showToast('Chưa có bản phân tích nào. Hãy upload và phân tích ảnh EEG trước!', 'info');
                addMcpLog('Chưa có dữ liệu phân tích.');
                return;
            }
        }

        const response = await fetch(apiUrl('/api/mcp/execute'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tool: toolName, params })
        });

        const data = await response.json();

        if (data.success) {
            addMcpLog(`✅ Tool ${toolName} thành công!`);
            
            if (data.report) {
                addMcpLog('--- BÁO CÁO ---\n' + data.report.substring(0, 500) + '...');
                // Also show in chat
                appendMessage('assistant', `📄 **Báo Cáo Y Khoa:**\n\n${data.report}`);
                switchTab('chat');
            } else if (data.history) {
                addMcpLog(`Tìm thấy ${data.history.length} bản phân tích.`);
            } else if (data.result) {
                addMcpLog('Kết quả: ' + JSON.stringify(data.result).substring(0, 200));
            }
        } else {
            addMcpLog(`❌ Lỗi: ${data.error}`);
        }

    } catch (error) {
        addMcpLog(`❌ Lỗi thực thi: ${error.message}`);
        showToast(`Lỗi MCP: ${error.message}`, 'error');
    }
}

function addMcpLog(message) {
    const log = document.getElementById('mcpLog');
    const time = new Date().toLocaleTimeString('vi-VN');
    log.innerHTML += `<div class="mcp-log-entry"><span class="mcp-log-time">[${time}]</span> ${message}</div>`;
    log.scrollTop = log.scrollHeight;
}

function formatUptime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${h}h ${m}m ${s}s`;
}

// ===== MODEL STATUS CHECK =====
async function checkModelStatus() {
    try {
        const response = await fetch(apiUrl('/api/model-status'));
        const data = await response.json();
        const dot = document.getElementById('modelDot');
        const text = document.getElementById('modelStatusText');
        if (data.status === 'ok') {
            dot.classList.remove('offline');
            text.textContent = 'ML Model ✓';
        } else {
            dot.classList.add('offline');
            text.textContent = 'ML Offline';
        }
    } catch {
        document.getElementById('modelDot').classList.add('offline');
        document.getElementById('modelStatusText').textContent = 'ML Offline';
    }
}

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', () => {
    checkMcpStatus();
    checkModelStatus();
    setUploadMode('edf');
    setInterval(checkMcpStatus, 30000);
    setInterval(checkModelStatus, 30000);
});

// ===== 3D EEG VISUALIZATION — PREMIUM (Three.js) =====
let scene3d, camera3d, renderer3d, controls3d;
let eeg3dAnimating = false;  // auto-rotate off by default; user toggles via button
let eeg3dAbnormalVisible = true;
let eeg3dParticlesVisible = true;
let eeg3dSurfaceVisible = false;
let abnormalMarkers3d = [];
let particleSystem3d = null;
let surfaceMesh3d = null;
let eeg3dData = null;
let cameraTarget3d = { x: 25, y: 12, z: 15 };
let cameraPos3d = { x: 120, y: 60, z: 80 };  // Start far for fly-in
let cameraLerping = false;
let glowSpheres3d = [];
let channelTubes3d = [];

// Render-loop lifecycle: track the active RAF + observers so init3DScene
// can be called repeatedly (per analysis) without stacking duplicate loops
// or leaking listeners. Loop is paused when tab hidden or section offscreen.
let animFrameId3d = null;
let resizeObs3d = null;
let intersectObs3d = null;
let is3dVisible = true;

// Hover tooltip state — shows abnormal region info when cursor is over a sphere
let tooltip3d = null;
let raycaster3d = null;
let mouse3d = null;
let hoverHandler3d = null;
let leaveHandler3d = null;

function dispose3DObject(obj) {
    if (!obj) return;
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
        if (Array.isArray(obj.material)) obj.material.forEach(m => m && m.dispose());
        else obj.material.dispose();
    }
    if (obj.texture) obj.texture.dispose();
}

function teardown3DScene() {
    if (animFrameId3d !== null) {
        cancelAnimationFrame(animFrameId3d);
        animFrameId3d = null;
    }
    if (resizeObs3d) { resizeObs3d.disconnect(); resizeObs3d = null; }
    if (intersectObs3d) { intersectObs3d.disconnect(); intersectObs3d = null; }
    if (renderer3d && renderer3d.domElement) {
        if (hoverHandler3d) renderer3d.domElement.removeEventListener('mousemove', hoverHandler3d);
        if (leaveHandler3d) renderer3d.domElement.removeEventListener('mouseleave', leaveHandler3d);
    }
    hoverHandler3d = null;
    leaveHandler3d = null;
    if (tooltip3d && tooltip3d.parentNode) {
        tooltip3d.parentNode.removeChild(tooltip3d);
    }
    tooltip3d = null;
    raycaster3d = null;
    mouse3d = null;
    if (scene3d) {
        const remaining = [];
        scene3d.traverse(c => remaining.push(c));
        remaining.forEach(c => dispose3DObject(c));
        scene3d.clear && scene3d.clear();
    }
    if (renderer3d) {
        renderer3d.dispose();
        if (renderer3d.forceContextLoss) renderer3d.forceContextLoss();
        if (renderer3d.domElement && renderer3d.domElement.parentNode) {
            renderer3d.domElement.parentNode.removeChild(renderer3d.domElement);
        }
        renderer3d = null;
    }
    scene3d = null;
    camera3d = null;
    controls3d = null;
    particleSystem3d = null;
    surfaceMesh3d = null;
    abnormalMarkers3d = [];
    glowSpheres3d = [];
    channelTubes3d = [];
}

function init3DScene() {
    const container = document.getElementById('eeg3dContainer');
    if (!container) return;

    if (typeof THREE === 'undefined') {
        console.warn('Three.js not loaded yet, retrying...');
        const ld = document.getElementById('eeg3dLoading');
        if (ld) ld.textContent = '⏳ Đang tải thư viện 3D...';
        setTimeout(init3DScene, 1000);
        return;
    }

    // Fully tear down previous scene: cancels old RAF loop, disposes GPU
    // resources, disconnects observers. Without this, each analysis stacked
    // another render loop on top of the previous one.
    teardown3DScene();
    while (container.querySelector('canvas')) container.querySelector('canvas').remove();
    
    let width = container.clientWidth;
    let height = container.clientHeight;
    
    // Fallback if container not yet laid out
    if (width === 0 || height === 0) {
        console.warn('3D container has 0 dimensions, retrying in 500ms...');
        setTimeout(init3DScene, 500);
        return;
    }
    
    // Scene
    scene3d = new THREE.Scene();
    scene3d.fog = new THREE.FogExp2(0x050510, 0.005);
    
    // Camera — start far for fly-in animation
    camera3d = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
    camera3d.position.set(120, 60, 80);
    
    // Renderer — shadows off (nothing in the scene meaningfully receives them
    // and PCFSoftShadowMap is one of the most expensive features in WebGL).
    // Cap pixelRatio at 1.5 to halve fill rate on retina/4K without losing AA.
    renderer3d = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer3d.setSize(width, height);
    renderer3d.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer3d.setClearColor(0x000000, 0);
    renderer3d.shadowMap.enabled = false;
    container.appendChild(renderer3d.domElement);
    
    // Controls
    controls3d = new THREE.OrbitControls(camera3d, renderer3d.domElement);
    controls3d.enableDamping = true;
    controls3d.dampingFactor = 0.06;
    controls3d.minDistance = 10;
    controls3d.maxDistance = 250;
    controls3d.target.set(25, 12, 15);
    controls3d.update();
    
    // ===== LIGHTING (lean: 1 ambient + 1 directional + 1 accent point) =====
    // Went from 5 lights (ambient + directional + 3 point) to 3. Phong runs
    // per-pixel per-light, so halving the light count is a big win on GPUs
    // that render many meshes (23 EEG ribbons × many verts). Bumped ambient
    // slightly to compensate for the lost fill.
    scene3d.add(new THREE.AmbientLight(0x303050, 0.55));

    const mainLight = new THREE.DirectionalLight(0xffffff, 0.7);
    mainLight.position.set(40, 60, 30);
    scene3d.add(mainLight);

    const purpleLight = new THREE.PointLight(0x8b5cf6, 1.2, 90);
    purpleLight.position.set(10, 30, 10);
    scene3d.add(purpleLight);
    
    // ===== STAR FIELD BACKGROUND =====
    createStarField();
    
    // ===== GRID FLOOR (Premium) =====
    const gridSize = 100;
    const gridGeo = new THREE.PlaneGeometry(gridSize, gridSize, 50, 50);
    const gridMat = new THREE.MeshBasicMaterial({
        color: 0x0a0a2e,
        wireframe: true,
        transparent: true,
        opacity: 0.15
    });
    const gridMesh = new THREE.Mesh(gridGeo, gridMat);
    gridMesh.rotation.x = -Math.PI / 2;
    gridMesh.position.set(25, -2, 15);
    scene3d.add(gridMesh);
    
    // Ground glow plane
    const glowGeo = new THREE.PlaneGeometry(80, 60);
    const glowMat = new THREE.MeshBasicMaterial({
        color: 0x8b5cf6,
        transparent: true,
        opacity: 0.03,
        side: THREE.DoubleSide
    });
    const glowPlane = new THREE.Mesh(glowGeo, glowMat);
    glowPlane.rotation.x = -Math.PI / 2;
    glowPlane.position.set(25, -1.5, 15);
    scene3d.add(glowPlane);
    
    // ===== AXIS LABELS =====
    createAxisLabel3D('Thời gian →', 55, -4, 15, '#3b82f6');
    createAxisLabel3D('↑ Biên độ', -2, 28, 15, '#10b981');
    createAxisLabel3D('Kênh EEG →', 25, -4, 38, '#f59e0b');
    
    // ===== PARTICLE SYSTEM =====
    createParticleSystem();
    
    // Handle resize — stored at module level so teardown can disconnect it
    resizeObs3d = new ResizeObserver(() => {
        if (!camera3d || !renderer3d) return;
        const w = container.clientWidth, h = container.clientHeight;
        if (w === 0 || h === 0) return;
        camera3d.aspect = w / h;
        camera3d.updateProjectionMatrix();
        renderer3d.setSize(w, h);
    });
    resizeObs3d.observe(container);

    // Pause render loop when 3D container leaves viewport (user switched tab
    // or scrolled away). Resume when it comes back.
    intersectObs3d = new IntersectionObserver((entries) => {
        for (const entry of entries) {
            is3dVisible = entry.isIntersecting;
            if (is3dVisible && animFrameId3d === null && renderer3d && !document.hidden) {
                animate3D();
            }
        }
    }, { threshold: 0.01 });
    intersectObs3d.observe(container);

    // Hover tooltip for abnormal markers — raycaster hits glow spheres and
    // displays channel/time/probability info in a floating div.
    tooltip3d = document.createElement('div');
    tooltip3d.style.cssText =
        'position:absolute; pointer-events:none; background:rgba(15,15,25,0.95); ' +
        'border:1px solid #ef4444; border-radius:8px; padding:8px 12px; color:#fff; ' +
        'font-size:12px; line-height:1.5; z-index:100; display:none; white-space:nowrap; ' +
        'box-shadow:0 4px 16px rgba(239,68,68,0.35); backdrop-filter:blur(6px);';
    container.appendChild(tooltip3d);

    raycaster3d = new THREE.Raycaster();
    mouse3d = new THREE.Vector2();

    hoverHandler3d = (event) => {
        if (!renderer3d || !camera3d || abnormalMarkers3d.length === 0) return;
        const rect = renderer3d.domElement.getBoundingClientRect();
        mouse3d.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse3d.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster3d.setFromCamera(mouse3d, camera3d);
        // Hover hits every sphere (primary + secondary) so the tooltip shows
        // the specific channel the cursor is over, not only the hottest one.
        const hits = raycaster3d.intersectObjects(abnormalMarkers3d, false);
        if (hits.length > 0 && hits[0].object.userData.region) {
            const r = hits[0].object.userData.region;
            tooltip3d.innerHTML =
                '<div style="color:#ef4444; font-weight:bold; margin-bottom:4px;">🔴 Vùng bất thường</div>' +
                '<div><b>Kênh:</b> ' + (r.topChannelName || 'N/A') +
                    (r.rank ? ' <span style="color:#94a3b8;">(#' + r.rank + ')</span>' : '') + '</div>' +
                '<div><b>Thời gian:</b> ' + r.start + 's – ' + r.end + 's</div>' +
                '<div><b>Xác suất CNN:</b> ' + r.prob + '%</div>' +
                (r.peakZScore != null ? '<div><b>Độ nổi bật kênh:</b> ' + r.peakZScore + 'σ</div>' : '') +
                '<div><b>Mức độ:</b> ' + (r.label || r.severity) + '</div>';
            tooltip3d.style.left = (event.clientX - rect.left + 14) + 'px';
            tooltip3d.style.top = (event.clientY - rect.top + 14) + 'px';
            tooltip3d.style.display = 'block';
            renderer3d.domElement.style.cursor = 'pointer';
        } else {
            tooltip3d.style.display = 'none';
            renderer3d.domElement.style.cursor = 'grab';
        }
    };

    leaveHandler3d = () => {
        if (tooltip3d) tooltip3d.style.display = 'none';
        if (renderer3d && renderer3d.domElement) renderer3d.domElement.style.cursor = 'grab';
    };

    renderer3d.domElement.addEventListener('mousemove', hoverHandler3d);
    renderer3d.domElement.addEventListener('mouseleave', leaveHandler3d);

    // Hide loading
    const ld = document.getElementById('eeg3dLoading');
    if (ld) ld.style.display = 'none';

    // Fly-in animation
    cameraLerping = true;
    cameraPos3d = { x: 60, y: 35, z: 50 };
    cameraTarget3d = { x: 25, y: 12, z: 15 };

    animate3D();
}

function createStarField() {
    const starCount = 800;  // was 2000 — background stars don't need that density
    const positions = new Float32Array(starCount * 3);
    const colors = new Float32Array(starCount * 3);
    const sizes = new Float32Array(starCount);
    
    for (let i = 0; i < starCount; i++) {
        positions[i * 3] = (Math.random() - 0.5) * 400;
        positions[i * 3 + 1] = (Math.random() - 0.5) * 400;
        positions[i * 3 + 2] = (Math.random() - 0.5) * 400;
        
        const brightness = 0.3 + Math.random() * 0.7;
        const tint = Math.random();
        colors[i * 3] = brightness * (tint > 0.7 ? 0.8 : 0.6);
        colors[i * 3 + 1] = brightness * (tint > 0.3 ? 0.7 : 0.5);
        colors[i * 3 + 2] = brightness;
        
        sizes[i] = 0.3 + Math.random() * 1.5;
    }
    
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    const mat = new THREE.PointsMaterial({
        size: 0.8,
        vertexColors: true,
        transparent: true,
        opacity: 0.6,
        sizeAttenuation: true
    });
    
    const stars = new THREE.Points(geo, mat);
    stars.userData.isStars = true;
    scene3d.add(stars);
}

function createParticleSystem() {
    const count = 150;  // was 300 — each animated particle re-uploads to GPU every frame
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    
    const palette = [
        [0.54, 0.36, 0.96],  // purple
        [0.23, 0.51, 0.96],  // blue
        [0.02, 0.71, 0.83],  // cyan
        [0.06, 0.72, 0.51],  // green
    ];
    
    for (let i = 0; i < count; i++) {
        positions[i * 3] = Math.random() * 60 - 5;
        positions[i * 3 + 1] = Math.random() * 30;
        positions[i * 3 + 2] = Math.random() * 40 - 5;
        
        const c = palette[Math.floor(Math.random() * palette.length)];
        colors[i * 3] = c[0];
        colors[i * 3 + 1] = c[1];
        colors[i * 3 + 2] = c[2];
    }
    
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    const mat = new THREE.PointsMaterial({
        size: 0.4,
        vertexColors: true,
        transparent: true,
        opacity: 0.5,
        sizeAttenuation: true
    });
    
    particleSystem3d = new THREE.Points(geo, mat);
    particleSystem3d.userData.isParticles = true;
    scene3d.add(particleSystem3d);
}

function createAxisLabel3D(text, x, y, z, color) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 300;
    canvas.height = 64;
    ctx.font = 'bold 26px Inter, Arial, sans-serif';
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
    ctx.fillText(text, 150, 42);
    
    const texture = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({ map: texture, transparent: true, opacity: 0.7 });
    const sprite = new THREE.Sprite(mat);
    sprite.position.set(x, y, z);
    sprite.scale.set(18, 4, 1);
    scene3d.add(sprite);
}

// ===== AMPLITUDE → COLOR GRADIENT =====
function amplitudeToColor(val, channelColor) {
    // val: -1 to 1 normalized amplitude
    const absVal = Math.abs(val);
    const base = new THREE.Color(channelColor);
    
    if (absVal > 0.8) {
        // High amplitude → warm (red/orange)
        return new THREE.Color().lerpColors(base, new THREE.Color(0xff4444), absVal * 0.7);
    } else if (absVal > 0.4) {
        // Medium → base color brighter
        return new THREE.Color().lerpColors(base, new THREE.Color(0xffffff), absVal * 0.3);
    }
    // Low → base color dimmer
    return new THREE.Color().lerpColors(new THREE.Color(0x1a1a3a), base, 0.4 + absVal * 0.6);
}

function render3DEEG(data) {
    if (!scene3d || !data.waveform) return;
    eeg3dData = data;
    
    // Remove + dispose old EEG meshes. Without dispose(), repeatedly
    // re-rendering after each analysis leaked geometry/material into VRAM,
    // which eventually forced browser GC and caused frame hitches.
    const toRemove = [];
    scene3d.traverse(c => { if (c.userData.isEEG) toRemove.push(c); });
    toRemove.forEach(o => {
        scene3d.remove(o);
        dispose3DObject(o);
    });
    abnormalMarkers3d = [];
    glowSpheres3d = [];
    channelTubes3d = [];
    surfaceMesh3d = null;
    
    const waveform = data.waveform;
    const channelNames = Object.keys(waveform.channels);
    const chColors = [0x3b82f6, 0x10b981, 0xf59e0b, 0xec4899, 0x8b5cf6, 0x06b6d4];
    const abnormalRegions = data.abnormalRegions || [];
    const duration = data.duration || 3600;
    const nCh = channelNames.length;
    
    // Store all channel data for surface mesh
    const allChannelPoints = [];
    
    channelNames.forEach((chName, chIdx) => {
        const rawData = waveform.channels[chName];
        const n = rawData.length;
        const mean = rawData.reduce((a,b) => a+b, 0) / n;
        const std = Math.sqrt(rawData.reduce((a,b) => a + (b-mean)**2, 0) / n) || 1;
        
        // Downsample
        const maxPts = 400;
        const step = Math.max(1, Math.floor(n / maxPts));
        const sampled = [];
        for (let i = 0; i < n; i += step) sampled.push((rawData[i] - mean) / std);
        
        const nPts = sampled.length;
        const xScale = 50 / nPts;
        const zPos = chIdx * 6;
        const baseColor = chColors[chIdx % chColors.length];
        const channelPoints = [];
        
        // ===== TUBE-STYLE RIBBON (thicker, with gradient colors) =====
        const ribbonW = 0.25;
        const positions = [], colors = [], indices = [];
        
        for (let i = 0; i < nPts; i++) {
            const x = i * xScale;
            const y = sampled[i] * 3.5 + 12;
            const timeSec = (i * step / n) * duration;
            channelPoints.push({ x, y, z: zPos });
            
            let isAbnormal = false;
            for (const r of abnormalRegions) {
                if (timeSec >= r.start && timeSec <= r.end) { isAbnormal = true; break; }
            }
            
            // 4 vertices per point for tube-like ribbon cross section
            positions.push(x, y + ribbonW, zPos - ribbonW * 0.5);
            positions.push(x, y + ribbonW, zPos + ribbonW * 0.5);
            positions.push(x, y - ribbonW, zPos + ribbonW * 0.5);
            positions.push(x, y - ribbonW, zPos - ribbonW * 0.5);
            
            const c = isAbnormal 
                ? new THREE.Color(0xff3333) 
                : amplitudeToColor(sampled[i], baseColor);
            const glow = isAbnormal ? 1.0 : 0.85;
            
            for (let v = 0; v < 4; v++) {
                const dim = v < 2 ? glow : glow * 0.65;
                colors.push(c.r * dim, c.g * dim, c.b * dim);
            }
            
            if (i < nPts - 1) {
                const vi = i * 4;
                // Top face
                indices.push(vi, vi+1, vi+5); indices.push(vi, vi+5, vi+4);
                // Right face
                indices.push(vi+1, vi+2, vi+6); indices.push(vi+1, vi+6, vi+5);
                // Bottom face
                indices.push(vi+2, vi+3, vi+7); indices.push(vi+2, vi+7, vi+6);
                // Left face
                indices.push(vi+3, vi, vi+4); indices.push(vi+3, vi+4, vi+7);
            }
        }
        
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geo.setIndex(indices);
        geo.computeVertexNormals();
        
        const mat = new THREE.MeshPhongMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.92,
            shininess: 60,
            specular: new THREE.Color(0x333366)
        });
        
        const mesh = new THREE.Mesh(geo, mat);
        mesh.userData.isEEG = true;
        mesh.userData.channelName = chName;
        scene3d.add(mesh);
        channelTubes3d.push(mesh);
        
        // ===== GLOW LINE on top =====
        const linePts = [];
        for (let i = 0; i < nPts; i++) {
            linePts.push(i * xScale, sampled[i] * 3.5 + 12 + ribbonW + 0.05, zPos);
        }
        const lineGeo = new THREE.BufferGeometry();
        lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(linePts, 3));
        const lineMat = new THREE.LineBasicMaterial({ 
            color: baseColor, 
            transparent: true, 
            opacity: 0.4 
        });
        const line = new THREE.Line(lineGeo, lineMat);
        line.userData.isEEG = true;
        scene3d.add(line);
        
        // ===== CHANNEL LABEL =====
        createChannelLabel3D(chName, -4, 12, zPos, chColors[chIdx % chColors.length]);
        
        allChannelPoints.push(channelPoints);
    });
    
    // ===== ABNORMAL REGION MARKERS (Glow Spheres only) =====
    // Previously also drew large red BoxGeometry columns + a PointLight per
    // region. With 70+ abnormal regions in a typical file those overlapped
    // into a solid red pulsing wall and made WebGL juggle 70+ lights, which
    // dominated the frame. Spheres alone still mark the locations clearly
    // and they self-illuminate via emissive material without needing lights.
    // Ranking abnormal regions by channel activity — the tricky part.
    //
    // The CNN only outputs ONE probability per 4s × 23-channel window, so we
    // can't ask it which channel fired. We have to infer from the waveform.
    //
    // Naive approach (what we tried before): rank channels by max(|raw value|)
    // in the region's time range. This fails on real EEG because channels
    // have per-channel DC offsets / baseline drift — a flat channel sitting
    // at +500 µV has abs=500, beating an actually-spiking channel whose
    // centered signal peaks at ±50 µV. That's why the previous markers
    // landed on quiet channels and missed the tall columns.
    //
    // Correct approach: z-score each channel against its own mean/std BEFORE
    // ranking. This matches how the visualization draws the ribbons
    // (sampled = (raw − mean) / std), so "tall column in the 3D view"
    // corresponds exactly to "large |z|" here. Drift has no effect because
    // it's subtracted out.
    //
    // Bonus: we also use the z-score peak's signed VALUE to position the
    // sphere's Y on top of the spike (instead of a fixed y=22 above the
    // whole stack). And we use the X of the peak sample, not the region
    // midpoint, so the sphere sits exactly on the visible spike.
    const TOP_K = 5;
    const waveformTimeArr = waveform.time || [];
    const channelDataArr = channelNames.map(n => waveform.channels[n] || []);

    // Per-channel stats on the same downsampled array the viz renders
    const channelStats = channelDataArr.map(chData => {
        if (!chData.length) return { mean: 0, std: 1 };
        let sum = 0;
        for (let i = 0; i < chData.length; i++) sum += chData[i];
        const mean = sum / chData.length;
        let sq = 0;
        for (let i = 0; i < chData.length; i++) sq += (chData[i] - mean) ** 2;
        const std = Math.sqrt(sq / chData.length) || 1;
        return { mean, std };
    });

    const pickTopChannels = (region) => {
        if (!waveformTimeArr.length || !channelDataArr.length) return [];

        // Locate waveform sample indices that fall inside [region.start, region.end].
        // Abnormal regions are 4s wide and the waveform is heavily downsampled
        // (one point per ~4s of EEG), so a region may span 0–1 points; when
        // that happens we snap to the nearest sample to the region midpoint.
        let startIdx = -1, endIdx = -1;
        for (let i = 0; i < waveformTimeArr.length; i++) {
            const t = waveformTimeArr[i];
            if (t > region.end) break;
            if (t >= region.start) {
                if (startIdx === -1) startIdx = i;
                endIdx = i;
            }
        }
        if (startIdx === -1) {
            const mid = (region.start + region.end) / 2;
            let closest = 0, closestDiff = Infinity;
            for (let i = 0; i < waveformTimeArr.length; i++) {
                const d = Math.abs(waveformTimeArr[i] - mid);
                if (d < closestDiff) { closestDiff = d; closest = i; }
            }
            startIdx = endIdx = closest;
        }

        const ranked = [];
        for (let ci = 0; ci < channelDataArr.length; ci++) {
            const chData = channelDataArr[ci];
            if (!chData.length) continue;
            const { mean, std } = channelStats[ci];
            let absPeak = 0;
            let signedAtPeak = 0;
            let peakIdx = startIdx;
            for (let i = startIdx; i <= endIdx && i < chData.length; i++) {
                const z = (chData[i] - mean) / std;
                const a = Math.abs(z);
                if (a > absPeak) {
                    absPeak = a;
                    signedAtPeak = z;
                    peakIdx = i;
                }
            }
            ranked.push({ idx: ci, peak: absPeak, signed: signedAtPeak, peakIdx });
        }
        ranked.sort((a, b) => b.peak - a.peak);
        return ranked.slice(0, TOP_K);
    };

    abnormalRegions.forEach(region => {
        const tops = pickTopChannels(region);
        if (!tops.length) return;

        region.topChannel = tops[0].idx;
        region.topChannelName = channelNames[tops[0].idx];

        tops.forEach((entry, rank) => {
            const chIdx = entry.idx;
            const isPrimary = rank === 0;

            // X: place sphere exactly above the peak sample, not the region midpoint
            const xAtPeak = (waveformTimeArr[entry.peakIdx] / duration) * 50;
            // Y: follow the waveform — y = sampled * 3.5 + 12 in render3DEEG,
            // plus a small offset so the sphere sits above the ribbon top.
            // Clamp so huge spikes don't fling the sphere off-screen.
            const sphereY = Math.max(2, Math.min(48, entry.signed * 3.5 + 12 + 1.5));
            const sphereZ = chIdx * 6;

            const radius = isPrimary ? 1.0 : 0.6;
            const segs = isPrimary ? 14 : 8;
            const sphereGeo = new THREE.SphereGeometry(radius, segs, segs);
            const sphereMat = new THREE.MeshPhongMaterial({
                color: 0xff4444,
                emissive: 0xff2222,
                emissiveIntensity: isPrimary ? 0.9 : 0.55,
                transparent: true,
                opacity: isPrimary ? 0.9 : 0.55
            });
            const sphere = new THREE.Mesh(sphereGeo, sphereMat);
            sphere.position.set(xAtPeak, sphereY, sphereZ);
            sphere.userData.isEEG = true;
            sphere.userData.isAbnormalMarker = true;
            sphere.userData.region = Object.assign({}, region, {
                topChannel: chIdx,
                topChannelName: channelNames[chIdx],
                rank: rank + 1,
                peakZScore: Number(entry.peak.toFixed(2))
            });
            scene3d.add(sphere);
            abnormalMarkers3d.push(sphere);
            if (isPrimary) glowSpheres3d.push(sphere);
        });
    });
    
    // ===== BUILD SURFACE MESH (hidden by default) =====
    buildSurfaceMesh(allChannelPoints, nCh);
    
    // ===== INFO OVERLAY =====
    const infoEl = document.getElementById('eeg3dInfo');
    const infoText = document.getElementById('eeg3dInfoText');
    if (infoEl && infoText) {
        infoEl.style.display = 'block';
        infoText.textContent = `${nCh} kênh | ${Math.round(duration)}s | ${abnormalRegions.length} vùng bất thường`;
    }
    
    // Update controls
    const centerZ = (nCh - 1) * 3;
    controls3d.target.set(25, 12, centerZ);
    
    // Fly-in camera
    cameraLerping = true;
    cameraPos3d = { x: 55, y: 30, z: centerZ + 40 };
    cameraTarget3d = { x: 25, y: 12, z: centerZ };
}

function buildSurfaceMesh(allChannelPoints, nCh) {
    if (nCh < 2 || !allChannelPoints[0]) return;
    
    const nPts = allChannelPoints[0].length;
    const positions = [], colors = [], indices = [];
    
    for (let ch = 0; ch < nCh; ch++) {
        for (let i = 0; i < nPts; i++) {
            const pt = allChannelPoints[ch][i];
            positions.push(pt.x, pt.y, pt.z);
            
            const t = i / nPts;
            const h = ch / nCh;
            colors.push(0.2 + t * 0.3, 0.1 + h * 0.5, 0.5 + (1-t) * 0.4);
            
            if (ch < nCh - 1 && i < nPts - 1) {
                const curr = ch * nPts + i;
                const next = (ch + 1) * nPts + i;
                indices.push(curr, next, curr + 1);
                indices.push(next, next + 1, curr + 1);
            }
        }
    }
    
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();
    
    const mat = new THREE.MeshPhongMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.15,
        shininess: 20,
        wireframe: false
    });
    
    surfaceMesh3d = new THREE.Mesh(geo, mat);
    surfaceMesh3d.userData.isEEG = true;
    surfaceMesh3d.visible = eeg3dSurfaceVisible;
    scene3d.add(surfaceMesh3d);
}

function createChannelLabel3D(text, x, y, z, color) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 200;
    canvas.height = 48;
    
    // Glow effect
    ctx.shadowColor = '#' + color.toString(16).padStart(6, '0');
    ctx.shadowBlur = 12;
    ctx.font = 'bold 22px Inter, Arial, sans-serif';
    ctx.fillStyle = '#' + color.toString(16).padStart(6, '0');
    ctx.textAlign = 'right';
    ctx.fillText(text, 190, 32);
    
    const texture = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({ map: texture, transparent: true, opacity: 0.9 });
    const sprite = new THREE.Sprite(mat);
    sprite.position.set(x, y, z);
    sprite.scale.set(10, 2.5, 1);
    sprite.userData.isEEG = true;
    scene3d.add(sprite);
}

// ===== ANIMATION LOOP =====
// Single-owner loop: stops itself when renderer is torn down, tab is hidden,
// or the 3D section scrolls out of view. Any of those cases pause work so the
// GPU/CPU can idle. Resume is triggered by visibilitychange / intersection.
function animate3D() {
    if (!renderer3d || !scene3d || !camera3d) { animFrameId3d = null; return; }
    if (document.hidden || !is3dVisible) { animFrameId3d = null; return; }

    animFrameId3d = requestAnimationFrame(animate3D);

    const time = Date.now() * 0.001;

    // Smooth camera lerp
    if (cameraLerping) {
        camera3d.position.x += (cameraPos3d.x - camera3d.position.x) * 0.03;
        camera3d.position.y += (cameraPos3d.y - camera3d.position.y) * 0.03;
        camera3d.position.z += (cameraPos3d.z - camera3d.position.z) * 0.03;
        controls3d.target.x += (cameraTarget3d.x - controls3d.target.x) * 0.03;
        controls3d.target.y += (cameraTarget3d.y - controls3d.target.y) * 0.03;
        controls3d.target.z += (cameraTarget3d.z - controls3d.target.z) * 0.03;

        const dist = Math.abs(camera3d.position.x - cameraPos3d.x) +
                     Math.abs(camera3d.position.y - cameraPos3d.y) +
                     Math.abs(camera3d.position.z - cameraPos3d.z);
        if (dist < 0.5) cameraLerping = false;
    }

    if (controls3d) {
        controls3d.autoRotate = eeg3dAnimating;
        controls3d.autoRotateSpeed = 0.4;
        controls3d.update();
    }

    // Animate particles only when visible — skipping this avoids a 300-vertex
    // GPU buffer reupload every frame when particles are toggled off.
    if (particleSystem3d && particleSystem3d.visible && eeg3dParticlesVisible) {
        const pos = particleSystem3d.geometry.attributes.position.array;
        for (let i = 0; i < pos.length; i += 3) {
            pos[i + 1] += Math.sin(time + pos[i] * 0.1) * 0.01;
            pos[i] += Math.cos(time * 0.5 + pos[i + 2] * 0.05) * 0.005;
        }
        particleSystem3d.geometry.attributes.position.needsUpdate = true;
    }

    // Pulse glow spheres
    glowSpheres3d.forEach((sphere, idx) => {
        if (!sphere.visible) return;
        const scale = 0.8 + Math.sin(time * 3 + idx) * 0.4;
        sphere.scale.set(scale, scale, scale);
        sphere.material.emissiveIntensity = 0.5 + Math.sin(time * 2.5 + idx * 0.7) * 0.4;
    });

    renderer3d.render(scene3d, camera3d);
}

// Resume the loop when the user returns to the tab. If the scene was torn
// down or the section is offscreen, animate3D() will no-op on its own.
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && renderer3d && is3dVisible && animFrameId3d === null) {
        animate3D();
    }
});

// ===== VIEW PRESETS =====
function set3DView(preset) {
    if (!camera3d || !controls3d) return;
    const centerZ = eeg3dData ? (Object.keys(eeg3dData.waveform.channels).length - 1) * 3 : 15;
    
    cameraLerping = true;
    cameraTarget3d = { x: 25, y: 12, z: centerZ };
    
    switch (preset) {
        case 'perspective':
            cameraPos3d = { x: 55, y: 30, z: centerZ + 40 };
            break;
        case 'top':
            cameraPos3d = { x: 25, y: 70, z: centerZ + 0.1 };
            break;
        case 'side':
            cameraPos3d = { x: 80, y: 12, z: centerZ };
            break;
        case 'front':
            cameraPos3d = { x: 25, y: 15, z: centerZ + 60 };
            break;
        case 'cinematic':
            cameraPos3d = { x: -15, y: 25, z: centerZ + 30 };
            break;
    }
    showToast(`Góc nhìn: ${preset.charAt(0).toUpperCase() + preset.slice(1)}`, 'info');
}

function reset3DCamera() {
    set3DView('perspective');
}

function toggle3DAnimation() {
    eeg3dAnimating = !eeg3dAnimating;
    const btn = document.getElementById('btnToggle3DRotate');
    if (btn) {
        btn.textContent = eeg3dAnimating ? '⏸️ Dừng xoay' : '▶️ Bật xoay';
        btn.style.background = eeg3dAnimating ? 'rgba(139, 92, 246, 0.25)' : '';
        btn.style.borderColor = eeg3dAnimating ? '#8b5cf6' : '';
    }
    // If the loop paused (e.g. scene stood still and was torn down), kick it
    // back on so the autorotate state takes effect immediately.
    if (eeg3dAnimating && renderer3d && is3dVisible && !document.hidden && animFrameId3d === null) {
        animate3D();
    }
    showToast(eeg3dAnimating ? '⏯️ Bật xoay tự động' : '⏸️ Đã dừng xoay', 'info');
}

function toggle3DAbnormal() {
    eeg3dAbnormalVisible = !eeg3dAbnormalVisible;
    abnormalMarkers3d.forEach(m => { m.visible = eeg3dAbnormalVisible; });
    const btn = document.getElementById('btnToggle3DAbnormal');
    if (btn) {
        btn.textContent = eeg3dAbnormalVisible ? '🔴 Ẩn chấm đỏ' : '⚫ Hiện chấm đỏ';
        btn.style.background = eeg3dAbnormalVisible ? 'rgba(239, 68, 68, 0.25)' : '';
        btn.style.borderColor = eeg3dAbnormalVisible ? '#ef4444' : '';
    }
    // Hide tooltip when markers disappear so it doesn't stay stuck on-screen
    if (!eeg3dAbnormalVisible && tooltip3d) tooltip3d.style.display = 'none';
    showToast(eeg3dAbnormalVisible ? '🔴 Đang hiển thị chấm bất thường' : '⚫ Đã ẩn chấm bất thường', 'info');
}

function toggle3DParticles() {
    eeg3dParticlesVisible = !eeg3dParticlesVisible;
    if (particleSystem3d) particleSystem3d.visible = eeg3dParticlesVisible;
    showToast(eeg3dParticlesVisible ? '✨ Particles: BẬT' : '✨ Particles: TẮT', 'info');
}

function toggle3DSurface() {
    eeg3dSurfaceVisible = !eeg3dSurfaceVisible;
    if (surfaceMesh3d) surfaceMesh3d.visible = eeg3dSurfaceVisible;
    showToast(eeg3dSurfaceVisible ? '🌊 Surface mesh: BẬT' : '🌊 Surface mesh: TẮT', 'info');
}

/* ============================================================ */
/* ===== MULTI-MODULE SUITE (Brain / Lung / Blood mock) ======= */
/* ============================================================ */

// Switch between top-level modules. Each module lives in its own
// .module-container div; we just toggle the .hidden class. EEG module
// keeps its internal tab navigation (switchTab) untouched.
function switchModule(name) {
    document.querySelectorAll('.module-container').forEach(m => {
        m.classList.toggle('hidden', m.id !== 'module-' + name);
    });
    document.querySelectorAll('.nav-module').forEach(b => {
        b.classList.toggle('active', b.dataset.module === name);
    });
    // Scroll to top on switch so header is visible
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Hash router: when the user arrives from the Next.js landing page via
// `/legacy.html#brain` (or eeg/lung/blood), auto-switch to that module
// instead of always landing on the default EEG tab.
(function () {
    var VALID = ['eeg', 'brain', 'lung', 'blood'];
    function applyHash() {
        var h = (window.location.hash || '').replace(/^#/, '').toLowerCase();
        if (VALID.indexOf(h) >= 0) {
            try { switchModule(h); } catch (e) { /* DOM not ready yet */ }
        }
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', applyHash);
    } else {
        applyHash();
    }
    window.addEventListener('hashchange', applyHash);
})();

// ============ BRAIN FILE INPUT FEEDBACK (upload visual confirmation) ============
(function () {
    if (window.__brainFileWired) return;
    window.__brainFileWired = true;

    function detectModalitySlot(filename) {
        var n = (filename || '').toLowerCase();
        // Reject segmentation ground-truth files — they're labels, not MRI
        if (/_?seg(\.|_|$)/.test(n)) return 'reject-seg';
        if (n.indexOf('t1ce') >= 0 || n.indexOf('t1c') >= 0 || n.indexOf('t1gd') >= 0) return 't1c';
        if (n.indexOf('flair') >= 0) return 'flair';
        if (n.indexOf('t2') >= 0) return 't2';
        if (n.indexOf('t1') >= 0) return 't1';
        return null;
    }

    function shortName(name, max) {
        max = max || 18;
        if (!name) return '';
        if (name.length <= max) return name;
        var ext = '';
        var dot = name.lastIndexOf('.');
        if (dot > 0 && name.length - dot < 8) { ext = name.slice(dot); name = name.slice(0, dot); }
        return name.slice(0, max - ext.length - 1) + '…' + ext;
    }

    function _setBrainMode(mode) {
        var m2d = document.getElementById('brainInputMode2D');
        var mni = document.getElementById('brainInputModeNifti');
        if (m2d) m2d.classList.toggle('is-active', mode === '2d');
        if (mni) mni.classList.toggle('is-active', mode === 'nifti');
    }

    // U-Net tab — 4 MRI slots
    var brainFi = document.getElementById('brainFileInput');
    if (brainFi) {
        brainFi.addEventListener('change', function (e) {
            var files = Array.from(e.target.files || []);
            if (files.length === 0) return;

            // Filter out seg files (BraTS ground truth, not a modality input)
            var rejectedSeg = files.filter(function (f) {
                return /_?seg(\.|_|$)/.test(f.name.toLowerCase());
            });
            files = files.filter(function (f) {
                return !/_?seg(\.|_|$)/.test(f.name.toLowerCase());
            });
            if (rejectedSeg.length > 0) {
                showToast('Bỏ qua ' + rejectedSeg.length + ' file *_seg.nii.gz (đó là nhãn ground truth, không phải MRI)', 'error');
            }
            if (files.length === 0) {
                showToast('Vui lòng chọn file MRI modality (flair/t1/t1ce/t2), không phải *_seg', 'error');
                e.target.value = '';
                return;
            }

            var isNifti = files.some(function (f) { return /\.nii(\.gz)?$/i.test(f.name); });
            _setBrainMode(isNifti ? 'nifti' : '2d');

            // Reset every slot to empty state
            document.querySelectorAll('#brainMriViews .mri-slot').forEach(function (s) {
                s.classList.remove('has-file');
                var name = s.getAttribute('data-slot') || '';
                var label = ({ t1: 'T1', t1c: 'T1c', t2: 'T2', flair: 'FLAIR' })[name] || name.toUpperCase();
                var empty = s.querySelector('.mri-slot-empty');
                if (empty) empty.innerHTML = '📎<br>' + label;
            });

            // 2D mode: only show preview in the first slot (others stay placeholder
            // since the BE replicates the single image to all 4 channels anyway).
            if (!isNifti && files.length === 1) {
                var f0 = files[0];
                var slotEl = document.querySelector('#brainMriViews .mri-slot[data-slot="t1"]');
                if (slotEl) {
                    slotEl.classList.add('has-file');
                    var url = URL.createObjectURL(f0);
                    var empty = slotEl.querySelector('.mri-slot-empty');
                    if (empty) {
                        empty.innerHTML = '<div class="slot-preview" style="background-image:url(\'' + url + '\');"></div>' +
                            '<small class="fname">' + shortName(f0.name) + '</small>';
                    }
                }
                showToast('Demo 2D · 1 ảnh → 4 channels', 'success');
                return;
            }

            // Assign files to slots
            var slotOrder = ['t1', 't1c', 't2', 'flair'];
            var nextFallback = 0;
            files.forEach(function (f) {
                var slot = detectModalitySlot(f.name);
                if (!slot || document.querySelector('#brainMriViews .mri-slot[data-slot="' + slot + '"].has-file')) {
                    // Couldn't detect, or slot already taken — use next empty slot
                    while (nextFallback < slotOrder.length) {
                        var candidate = slotOrder[nextFallback++];
                        var elc = document.querySelector('#brainMriViews .mri-slot[data-slot="' + candidate + '"]');
                        if (elc && !elc.classList.contains('has-file')) { slot = candidate; break; }
                    }
                }
                var slotEl = document.querySelector('#brainMriViews .mri-slot[data-slot="' + slot + '"]');
                if (slotEl) {
                    slotEl.classList.add('has-file');
                    var empty = slotEl.querySelector('.mri-slot-empty');
                    if (empty) {
                        empty.innerHTML = '<span class="check">✓</span><br>' +
                            '<small class="fname">' + shortName(f.name) + '</small>';
                    }
                }
            });

            showToast('Đã chọn ' + files.length + ' file MRI', 'success');
        });
    }

    // YOLO tab — single image preview
    var yoloFi = document.getElementById('yoloFileInput');
    if (yoloFi) {
        yoloFi.addEventListener('change', function (e) {
            var f = e.target.files && e.target.files[0];
            if (!f) return;
            var canvas = document.getElementById('yoloCanvas');
            if (canvas) {
                var url = URL.createObjectURL(f);
                canvas.innerHTML = '<img src="' + url + '" alt="Selected">' +
                    '<div class="yolo-file-tag">📎 ' + shortName(f.name, 32) + '</div>';
            }
            showToast('Đã chọn ' + shortName(f.name, 28), 'success');
        });
    }
})();

// ============ SKETCHFAB GLASS BRAIN — Viewer API integration ============
// Loads the Glass Brain Sketchfab model into #brainSketchfabFrame via their
// Viewer API. After analysis, we animate the camera + show a floating
// "Khối u · X mm" overlay near the brain (we can't inject a custom 3D mesh
// into Sketchfab's sandboxed scene, but camera control + overlay works).
var brainSketchfabApi = null;
var brainSketchfabReady = false;
var brainSketchfabInitTries = 0;

function initBrainSketchfab() {
    if (brainSketchfabApi || brainSketchfabReady) return;
    if (typeof Sketchfab === 'undefined') {
        // SDK still loading — retry up to 10× over ~5s
        if (brainSketchfabInitTries++ < 10) {
            setTimeout(initBrainSketchfab, 500);
        } else {
            console.warn('[BrainSketchfab] SDK never loaded');
        }
        return;
    }
    var iframe = document.getElementById('brainSketchfabFrame');
    if (!iframe) return;
    var uid = '9bcd3705024146cb9b482bd65295f040';     // Glass Brain
    try {
        var client = new Sketchfab('1.12.1', iframe);
        client.init(uid, {
            success: function (api) {
                api.start();
                api.addEventListener('viewerready', function () {
                    brainSketchfabApi = api;
                    brainSketchfabReady = true;
                    api.setAutoRotate(0.4, function () {});
                    // Override background color to match our dark theme.
                    // We're using transparent: 0 because Glass Brain (translucent
                    // material) needs an environment to render visibly — but we
                    // can still recolor the background to integrate with the UI.
                    try {
                        if (api.setBackground) {
                            api.setBackground({ color: [0.045, 0.063, 0.110, 1.0] }, function () {});
                        }
                    } catch (e) {}
                    console.log('[BrainSketchfab] viewer ready');
                });
            },
            error: function () { console.error('[BrainSketchfab] init error'); },
            ui_theme: 'dark',
            ui_infos: 0,
            ui_inspector: 0,
            ui_help: 0,
            ui_settings: 0,
            ui_watermark: 0,
            ui_watermark_link: 0,
            ui_hint: 0,
            ui_ar: 0,
            ui_vr: 0,
            ui_animations: 0,
            ui_stop: 0,
            ui_controls: 1,
            ui_fullscreen: 1,
            autostart: 1,
            // transparent: 0 — Glass Brain needs an environment to reflect;
            // a transparent background makes the translucent model render invisible.
            transparent: 0,
            preload: 1,
        });
    } catch (e) {
        console.error('[BrainSketchfab] init exception:', e);
    }
}

// Sketchfab init is now lazy — only fires when the user explicitly switches
// to the "Glass Brain (Sketchfab)" tab. The Glass Brain model is intentionally
// translucent and may render invisible under our theme, so the safer default
// is the Custom Three.js brain (always controllable + always visible).
// Pre-init the Custom Three.js brain when user opens the Brain module so
// they immediately see a 3D brain (even without running analysis).
(function () {
    if (window.__brainModuleNavWired) return;
    window.__brainModuleNavWired = true;
    document.addEventListener('click', function (e) {
        var btn = e.target.closest && e.target.closest('.nav-module');
        if (!btn) return;
        if (btn.dataset && btn.dataset.module === 'brain') {
            setTimeout(function () { initBrainCustom3D(null); }, 120);
        }
    });
    // Also kick off on page-load if Brain module is the currently visible one
    function maybeInit() {
        var mb = document.getElementById('module-brain');
        if (mb && !mb.classList.contains('hidden') && !brainCustomInited) {
            initBrainCustom3D(null);
        }
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () { setTimeout(maybeInit, 200); });
    } else {
        setTimeout(maybeInit, 200);
    }
})();

// ============ CUSTOM THREE.JS BRAIN — fallback ellipsoid + 3D tumor mesh ====
// Option B: a fully-controlled scene where we can inject a red tumor mesh
// inside the brain at the segmentation centroid. Loads /models/brain.glb if
// the user provides one, otherwise renders an ellipsoid stand-in.
var brainCustomScene, brainCustomCamera, brainCustomRenderer, brainCustomControls;
var brainCustomAnimId = null;
var brainCustomInited = false;
var brainCustomTumorParts = [];
var brainCustomModel = null;

function disposeBrainCustom3D() {
    if (brainCustomAnimId) { cancelAnimationFrame(brainCustomAnimId); brainCustomAnimId = null; }
    if (brainCustomRenderer) { try { brainCustomRenderer.dispose(); } catch (e) {} brainCustomRenderer = null; }
    brainCustomScene = null;
    brainCustomCamera = null;
    brainCustomControls = null;
    brainCustomTumorParts = [];
    brainCustomModel = null;
    brainCustomInited = false;
}

function initBrainCustom3D(tumorData) {
    try {
        var stage = document.getElementById('brainCustom3DStage');
        var canvas = document.getElementById('brainCustom3DCanvas');
        var ph = document.getElementById('brainCustom3DPlaceholder');
        if (!stage || !canvas || typeof THREE === 'undefined') return;
        if (ph) ph.style.display = 'none';
        disposeBrainCustom3D();

        var rect = stage.getBoundingClientRect();
        var w = Math.max(rect.width, 320), h = Math.max(rect.height, 240);

        brainCustomScene = new THREE.Scene();
        brainCustomScene.background = null;
        brainCustomCamera = new THREE.PerspectiveCamera(40, w / h, 0.1, 500);
        brainCustomCamera.position.set(0, 4, 32);
        brainCustomCamera.lookAt(0, 0, 0);

        brainCustomRenderer = new THREE.WebGLRenderer({
            canvas: canvas, antialias: true, alpha: true,
        });
        brainCustomRenderer.setClearColor(0x000000, 0);
        brainCustomRenderer.setSize(w, h, false);
        brainCustomRenderer.setPixelRatio(window.devicePixelRatio > 1 ? 2 : 1);

        brainCustomControls = new THREE.OrbitControls(brainCustomCamera, canvas);
        brainCustomControls.enableDamping = true;
        brainCustomControls.autoRotate = true;
        brainCustomControls.autoRotateSpeed = 0.6;
        brainCustomControls.target.set(0, 0, 0);

        // Warm clinical lighting: key from upper-right, soft fill from front,
        // cool rim from back. Reads the cortex wrinkles as brain tissue.
        brainCustomScene.add(new THREE.AmbientLight(0xffeede, 0.45));
        var dl = new THREE.DirectionalLight(0xfff2e8, 1.4);
        dl.position.set(18, 22, 20);
        brainCustomScene.add(dl);
        var fill = new THREE.DirectionalLight(0xfde0d4, 0.65);
        fill.position.set(-15, 8, 18);
        brainCustomScene.add(fill);
        var rim = new THREE.DirectionalLight(0xc0d4ff, 0.75);
        rim.position.set(0, -3, -25);
        brainCustomScene.add(rim);
        var top = new THREE.HemisphereLight(0xfff5ec, 0x3a2030, 0.35);
        brainCustomScene.add(top);

        function addEllipsoidBrain() {
            // Procedural brain: take a hi-poly sphere and displace its
            // vertices using a sum of sinusoidal "noise" terms — gives
            // the surface a sulci/gyri (cortical wrinkle) look that reads
            // as a real brain instead of a smooth ellipsoid.
            function buildBrainGeo(seed) {
                var geo = new THREE.SphereGeometry(1, 160, 110);
                var pos = geo.attributes.position;
                var color = new THREE.BufferAttribute(new Float32Array(pos.count * 3), 3);
                var v = new THREE.Vector3();
                for (var i = 0; i < pos.count; i++) {
                    v.fromBufferAttribute(pos, i);
                    var x = v.x, y = v.y, z = v.z;
                    // Multi-octave sinusoid → wrinkle pattern
                    var w1 = Math.sin(x * 6.2 + seed) * Math.cos(y * 5.8) * Math.sin(z * 7.1) * 0.085;
                    var w2 = Math.sin(x * 11.3 + 0.7) * Math.cos(y * 12.1 + seed * 0.3) * Math.sin(z * 10.7) * 0.045;
                    var w3 = Math.sin(x * 19.7) * Math.cos(y * 22.3) * Math.sin(z * 18.5 + seed * 0.5) * 0.022;
                    var w4 = Math.sin(x * 34) * Math.cos(y * 31) * Math.sin(z * 36) * 0.010;
                    var d = 1 + w1 + w2 + w3 + w4;
                    // Central fissure: dent along x=0 line on top hemisphere
                    var fissure = Math.exp(-x * x * 30) * Math.max(0, y) * 0.05;
                    d -= fissure;
                    v.multiplyScalar(d);
                    pos.setXYZ(i, v.x, v.y, v.z);
                    // Vertex color: darker in valleys (low d), lighter on peaks
                    var t = (d - 0.85) / 0.30;
                    t = Math.max(0, Math.min(1, t));
                    // Cream/pink brain tissue palette
                    var r = 0.78 + t * 0.16;
                    var g = 0.62 + t * 0.18;
                    var b = 0.66 + t * 0.16;
                    color.setXYZ(i, r, g, b);
                }
                geo.setAttribute('color', color);
                geo.computeVertexNormals();
                return geo;
            }

            var mat = new THREE.MeshPhongMaterial({
                vertexColors: true,
                transparent: true, opacity: 0.78,
                emissive: 0x442233, emissiveIntensity: 0.12,
                side: THREE.FrontSide,
                shininess: 22, specular: 0xffe6e0,
                flatShading: false,
            });
            var brainGeo = buildBrainGeo(1.0);
            var brain = new THREE.Mesh(brainGeo, mat);
            brain.scale.set(6.6, 5.6, 7.6);
            brainCustomScene.add(brain);

            // Inner glow shell — tumor LIGHTS UP through the translucent cortex
            var innerGlow = new THREE.Mesh(
                new THREE.SphereGeometry(1, 48, 32),
                new THREE.MeshBasicMaterial({
                    color: 0xff4466, transparent: true, opacity: 0.04,
                    side: THREE.BackSide, depthWrite: false,
                })
            );
            innerGlow.scale.copy(brain.scale).multiplyScalar(0.94);
            brainCustomScene.add(innerGlow);

            // Cerebellum: smaller wrinkled blob at posterior-inferior
            var cereGeo = buildBrainGeo(7.3);
            var cere = new THREE.Mesh(cereGeo, mat.clone());
            cere.scale.set(2.6, 1.7, 2.3);
            cere.position.set(0, -4.4, -3.4);
            brainCustomScene.add(cere);

            // Brainstem: smooth cylinder tapering to spinal cord
            var stemMat = new THREE.MeshPhongMaterial({
                color: 0xc89a8e, transparent: true, opacity: 0.85,
                shininess: 18, specular: 0xffe6e0,
            });
            var stem = new THREE.Mesh(
                new THREE.CylinderGeometry(0.85, 1.15, 3.6, 32), stemMat
            );
            stem.position.set(0, -5.5, -0.2);
            brainCustomScene.add(stem);

            brainCustomModel = brain;
            window._brainBounds = { x: 6.6, y: 5.6, z: 7.6 };
        }

        function tryLoadGLB(onDone) {
            if (typeof THREE.GLTFLoader === 'undefined') { addEllipsoidBrain(); return onDone(); }
            var loader = new THREE.GLTFLoader();
            // Probe known brain GLB paths in order — first one that loads wins.
            // Project-root /models/brain/human-brain.glb is served via a
            // server.js fallback static handler.
            var candidates = [
                '/models/brain/human-brain.glb',
                '/models/brain.glb',
            ];

            function tryNext(idx) {
                if (idx >= candidates.length) {
                    console.warn('[BrainCustom3D] No GLB found — using procedural fallback');
                    addEllipsoidBrain();
                    return onDone();
                }
                // Pre-flight: verify the URL actually returns a binary file,
                // not HTML (catch-all routes can silently serve index.html for
                // a missing asset, which GLTFLoader then chokes on).
                fetch(candidates[idx], { method: 'HEAD' }).then(function (r) {
                    var ct = (r.headers.get('Content-Type') || '').toLowerCase();
                    if (!r.ok || ct.indexOf('html') >= 0) {
                        console.warn('[BrainCustom3D] skip', candidates[idx],
                                     '(status=' + r.status + ', type=' + ct + ')');
                        return tryNext(idx + 1);
                    }
                    console.log('[BrainCustom3D] loading', candidates[idx],
                                '(' + ct + ', ' + (r.headers.get('Content-Length') || '?') + ' bytes)');
                    doLoad(idx);
                }).catch(function () { doLoad(idx); });   // fall through if HEAD blocked
            }
            function doLoad(idx) {
                loader.load(candidates[idx],
                    function (gltf) {
                        var m = gltf.scene;
                        m.traverse(function (c) {
                            if (c.isMesh) {
                                // X-ray translucent brain — tumor inside MUST be
                                // visible. Opacity ~0.28 lets red glow read through
                                // strongly; depthWrite off prevents brain from
                                // occluding the tumor's render order.
                                c.material = new THREE.MeshPhongMaterial({
                                    color: 0xe8c9bc, transparent: true, opacity: 0.28,
                                    emissive: 0x2a1418, emissiveIntensity: 0.08,
                                    side: THREE.DoubleSide, depthWrite: false,
                                    shininess: 30, specular: 0xffe6e0,
                                });
                                c.renderOrder = 1;
                            }
                        });
                        var box = new THREE.Box3().setFromObject(m);
                        var sz = box.getSize(new THREE.Vector3());
                        var ct = box.getCenter(new THREE.Vector3());
                        var mx = Math.max(sz.x, sz.y, sz.z);
                        var sc = 14 / mx;
                        m.scale.setScalar(sc);
                        m.position.set(-ct.x * sc, -ct.y * sc, -ct.z * sc);
                        brainCustomScene.add(m);
                        brainCustomModel = m;
                        var fb = new THREE.Box3().setFromObject(m);
                        window._brainBounds = {
                            x: (fb.max.x - fb.min.x) / 2,
                            y: (fb.max.y - fb.min.y) / 2,
                            z: (fb.max.z - fb.min.z) / 2,
                        };
                        console.log('[BrainCustom3D] loaded GLB:', candidates[idx]);
                        onDone();
                    },
                    undefined,
                    function () { tryNext(idx + 1); }
                );
            }
            tryNext(0);
        }

        tryLoadGLB(function () {
            // Auto-fit camera to actual brain bounds so the model fills the
            // viewport regardless of how the user's GLB was authored.
            var b = window._brainBounds || { x: 7, y: 6, z: 8 };
            var maxDim = Math.max(b.x, b.y, b.z);
            var fitDist = maxDim / Math.tan((brainCustomCamera.fov / 2) * Math.PI / 180) * 1.35;
            brainCustomCamera.position.set(0, maxDim * 0.4, fitDist);
            brainCustomCamera.lookAt(0, 0, 0);
            if (brainCustomControls) {
                brainCustomControls.target.set(0, 0, 0);
                brainCustomControls.update();
            }

            addBrainCustomTumor(tumorData);
            animateBrainCustom();
            brainCustomInited = true;
        });
    } catch (e) {
        console.error('[BrainCustom3D] init failed:', e);
    }
}

// Clear existing tumor parts (mesh + glows + light + line) so we can swap in
// a new tumor without rebuilding the whole brain scene.
function clearBrainCustomTumor() {
    if (!brainCustomScene) return;
    brainCustomTumorParts.forEach(function (obj) {
        if (obj && brainCustomScene) brainCustomScene.remove(obj);
        if (obj && obj.geometry && obj.geometry.dispose) obj.geometry.dispose();
        if (obj && obj.material && obj.material.dispose) obj.material.dispose();
    });
    brainCustomTumorParts = [];
}

function updateBrainCustomTumor(tumorData) {
    clearBrainCustomTumor();
    addBrainCustomTumor(tumorData);
}

// Build a Three.js BufferGeometry mesh from a backend marching-cubes payload
// (vertices in normalized [0,1] coords + face indices). One mesh per BraTS
// class (NCR / ED / ET), each translucent and colored to match the 2D
// segmentation overlay.
function renderTumorMesh(tumorMesh, b) {
    var maxB = Math.max(b.x, b.y, b.z);
    var fill = 0.85;
    // BraTS palette — must match the 2D nilearn overlay so the user can
    // cross-reference what they see in the slice panels.
    var palette = {
        'ed':  { color: 0xffd84a, emissive: 0x554200, opacity: 0.45 },
        'ncr': { color: 0xff2a44, emissive: 0x661018, opacity: 0.88 },
        'et':  { color: 0xff44b4, emissive: 0x5a1244, opacity: 0.85 },
    };
    // Render ED first (large halo of edema) so NCR + ET show through on top
    var renderOrder = { 'ed': 9, 'ncr': 11, 'et': 12 };

    var totalFaces = 0;
    var centroidSum = new THREE.Vector3();
    var centroidCount = 0;

    ['ed', 'ncr', 'et'].forEach(function (key) {
        var data = (tumorMesh.classes || {})[key];
        if (!data || !data.vertices || data.vertices.length === 0) return;

        var scale = data.scale || 10000;     // inverse-quantization factor
        var srcV = data.vertices;             // flat [x,y,z,...] × scale (0..scale)
        var srcF = data.faces;                // flat [a,b,c,...]
        var vCount = srcV.length / 3;
        var positions = new Float32Array(srcV.length);

        for (var i = 0; i < srcV.length; i += 3) {
            // Normalized [0, 1] in voxel space
            var nxN = srcV[i]     / scale;
            var nyN = srcV[i + 1] / scale;
            var nzN = srcV[i + 2] / scale;
            // Centered [-1, +1]
            var nx = nxN * 2 - 1;
            var ny = nyN * 2 - 1;
            var nz = nzN * 2 - 1;
            // World coords (same mapping as voxel-cloud renderer)
            positions[i]     = -nx * b.x * fill;   // radiological L/R flip
            positions[i + 1] = -ny * b.y * fill;   // image-top → world-up
            positions[i + 2] =  nz * b.z * fill;
            centroidSum.x += positions[i];
            centroidSum.y += positions[i + 1];
            centroidSum.z += positions[i + 2];
            centroidCount++;
        }

        var geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geo.setIndex(srcF);
        geo.computeVertexNormals();
        // Bounding sphere needed for frustum culling
        geo.computeBoundingSphere();

        var p = palette[key];
        var mat = new THREE.MeshPhongMaterial({
            color: p.color,
            transparent: true, opacity: p.opacity,
            emissive: p.emissive, emissiveIntensity: 0.55,
            depthTest: false, depthWrite: false,
            side: THREE.DoubleSide,
            shininess: 35, specular: 0xffffff,
            flatShading: false,
        });
        var mesh = new THREE.Mesh(geo, mat);
        mesh.renderOrder = renderOrder[key];
        mesh.userData.classKey = key;
        mesh.userData.baseEmissive = 0.55;
        brainCustomScene.add(mesh);
        brainCustomTumorParts.push(mesh);
        totalFaces += data.faceCount || (srcF.length / 3);
    });

    if (totalFaces === 0) {
        console.warn('[BrainTumor] tumorMesh arrived but had 0 faces total');
        return;
    }

    // Compute centroid of the whole tumor and add a small accent light.
    // (Previously we drew a vertical indicator line from centroid → top of
    // brain, but with translucent cortex it reads as a pink stick coming out
    // of the tumor — anatomically misleading. Dropped.)
    var centroid = centroidSum.divideScalar(centroidCount);
    var light = new THREE.PointLight(0xff3355, 1.4, maxB * 1.4);
    light.position.copy(centroid);
    brainCustomScene.add(light);
    brainCustomTumorParts.push(light);

    console.log('[BrainTumor] mesh rendered — total faces:', totalFaces,
                'centroid:', centroid.toArray().map(function (v) { return v.toFixed(2); }).join(','));
}

// Legacy voxel-cloud renderer — kept for older BE payloads that don't have
// the marching-cubes mesh. Same color palette + scale rules.
function renderTumorVoxelCloud(voxelCloud, b) {
    var grid = voxelCloud.gridSize || 128;
    var maxB = Math.max(b.x, b.y, b.z);
    var fill = 0.85;   // keep cloud well inside the brain cortex

    var ncrN = ((voxelCloud.ncr || []).length / 3) | 0;
    var edN  = ((voxelCloud.ed  || []).length / 3) | 0;
    var etN  = ((voxelCloud.et  || []).length / 3) | 0;
    var total = ncrN + edN + etN;
    if (total === 0) {
        console.warn('[BrainTumor] voxelCloud arrived but empty — no shape to render');
        return;
    }

    // Cube size scales inversely with sample density. With ~5500 sampled
    // voxels filling a tumor bbox of ~50³ voxels (a 50mm tumor), the average
    // spacing between sampled voxels is small enough that cubeSize = 4× the
    // base voxel size in world units makes adjacent cubes overlap cleanly
    // into a continuous "mass" — same shape as the 2D segmentation overlay.
    var voxelWorld = (maxB * 2) / grid;        // size of 1 voxel in world units
    var cubeSize = voxelWorld * 2.6;           // overlap factor for solid look

    // BraTS-style color palette (matches the segmentation overlay on the left)
    var classes = [
        { key: 'ed',  color: 0xffd84a, emissive: 0x66520a, opacity: 0.70, label: 'ED'  },
        { key: 'ncr', color: 0xff2a44, emissive: 0x661018, opacity: 0.92, label: 'NCR' },
        { key: 'et',  color: 0xff44b4, emissive: 0x5a1244, opacity: 0.92, label: 'ET'  },
    ];

    classes.forEach(function (cls) {
        var arr = voxelCloud[cls.key] || [];
        var count = (arr.length / 3) | 0;
        if (count === 0) return;

        var geo = new THREE.BoxGeometry(cubeSize, cubeSize, cubeSize);
        var mat = new THREE.MeshPhongMaterial({
            color: cls.color, transparent: true, opacity: cls.opacity,
            emissive: cls.emissive, emissiveIntensity: 0.85,
            depthTest: false, depthWrite: false,
            shininess: 20, specular: 0xffffff,
        });
        var mesh = new THREE.InstancedMesh(geo, mat, count);
        mesh.renderOrder = 10;
        mesh.userData.classKey = cls.key;
        mesh.userData.baseEmissive = 0.85;

        var dummy = new THREE.Object3D();
        for (var i = 0; i < count; i++) {
            var vx = arr[i * 3];
            var vy = arr[i * 3 + 1];
            var vz = arr[i * 3 + 2];
            // Map voxel coord (0..grid) → normalized (-1..+1) → world (-b..+b)
            var nx = (vx / grid) * 2 - 1;
            var ny = (vy / grid) * 2 - 1;
            var nz = (vz / grid) * 2 - 1;
            dummy.position.set(
                -nx * b.x * fill,    // radiological L/R flip
                -ny * b.y * fill,    // image-top → world-up
                 nz * b.z * fill,
            );
            // Random tilt per cube to break up the regular voxel grid look
            dummy.rotation.set(
                (vx * 0.37) % 1.2,
                (vy * 0.53) % 1.2,
                (vz * 0.41) % 1.2,
            );
            dummy.updateMatrix();
            mesh.setMatrixAt(i, dummy.matrix);
        }
        mesh.instanceMatrix.needsUpdate = true;
        brainCustomScene.add(mesh);
        brainCustomTumorParts.push(mesh);
    });

    // Centroid for the indicator line + a small (not huge) accent point light.
    // NO outer halo shells — they obscure the actual segmentation shape.
    var allX = 0, allY = 0, allZ = 0, allN = 0;
    classes.forEach(function (cls) {
        var arr = voxelCloud[cls.key] || [];
        for (var i = 0; i < arr.length; i += 3) {
            allX += arr[i]; allY += arr[i + 1]; allZ += arr[i + 2]; allN++;
        }
    });
    var ncx = (allX / allN / grid) * 2 - 1;
    var ncy = (allY / allN / grid) * 2 - 1;
    var ncz = (allZ / allN / grid) * 2 - 1;
    var wpos = new THREE.Vector3(
        -ncx * b.x * fill,
        -ncy * b.y * fill,
         ncz * b.z * fill,
    );

    // Subtle red point light to make surrounding cortex glow faintly
    var light = new THREE.PointLight(0xff3355, 1.4, maxB * 1.4);
    light.position.copy(wpos);
    brainCustomScene.add(light);
    brainCustomTumorParts.push(light);

    console.log('[BrainTumor] voxel cloud rendered:',
        'NCR=' + ncrN, 'ED=' + edN, 'ET=' + etN,
        'cubeSize=' + cubeSize.toFixed(3),
        'centroid=' + wpos.toArray().map(function (v) { return v.toFixed(2); }).join(','));
}

function addBrainCustomTumor(tumorData) {
    if (!tumorData || !tumorData.detected) return;
    var c = tumorData.centroid128 || { x: 64, y: 64, z: 64 };
    var bbox = tumorData.bbox128 || { xmin: 54, xmax: 74, ymin: 54, ymax: 74, zmin: 54, zmax: 74 };
    var b = window._brainBounds || { x: 7, y: 6, z: 8 };

    // === DIAGNOSTIC: log exactly what the BE sent us so we can tell whether
    // the marching-cubes path is being skipped because of a missing field,
    // empty classes, or some other reason. ===
    console.log('[BrainTumor:v2-mesh] === response inspection ===');
    console.log('  top-level keys:', Object.keys(tumorData).join(', '));
    if (tumorData.tumorMesh) {
        var tm = tumorData.tumorMesh;
        var cs = tm.classes || {};
        console.log('  tumorMesh present. gridSize=' + tm.gridSize + '. Per-class:',
            'NCR=' + ((cs.ncr || {}).faceCount || 0) + ' faces,',
            'ED=' + ((cs.ed || {}).faceCount || 0) + ' faces,',
            'ET=' + ((cs.et || {}).faceCount || 0) + ' faces');
    } else {
        console.warn('  tumorMesh field is MISSING — backend is on old code (restart needed)');
    }
    if (tumorData.voxelCloud) {
        var vc = tumorData.voxelCloud;
        console.log('  voxelCloud present:',
            'NCR=' + ((vc.ncr || []).length / 3),
            'ED=' + ((vc.ed || []).length / 3),
            'ET=' + ((vc.et || []).length / 3));
    } else {
        console.warn('  voxelCloud field is MISSING');
    }

    // 1st choice: surface mesh from marching cubes (continuous smooth shape)
    if (tumorData.tumorMesh && tumorData.tumorMesh.classes) {
        var cls = tumorData.tumorMesh.classes;
        var hasAny = (cls.ncr && cls.ncr.faceCount > 0) ||
                     (cls.ed  && cls.ed.faceCount  > 0) ||
                     (cls.et  && cls.et.faceCount  > 0);
        if (hasAny) {
            console.log('[BrainTumor] ✓ using marching-cubes mesh renderer');
            renderTumorMesh(tumorData.tumorMesh, b);
            return;
        }
        console.warn('[BrainTumor] tumorMesh present but ALL classes empty — falling back to voxel cloud');
    }
    // 2nd choice: voxel cloud (instanced cubes — approximation of shape)
    if (tumorData.voxelCloud && tumorData.voxelCloud.gridSize) {
        console.log('[BrainTumor] ✓ using voxel-cloud renderer');
        renderTumorVoxelCloud(tumorData.voxelCloud, b);
        return;
    }
    console.warn('[BrainTumor] ⚠ FALLBACK to single sphere — backend hasn\'t been restarted with new code');

    // Map centroid from 128³ voxel space to brain world coords (within 55%
    // of bounds so it stays solidly inside the brain volume, not on cortex).
    var nx = (c.x / 128) * 2 - 1;        // -1..+1
    var ny = (c.y / 128) * 2 - 1;
    var nz = (c.z / 128) * 2 - 1;
    var pos = {
        x: -nx * b.x * 0.55,    // flip X for radiological consistency
        y: -ny * b.y * 0.55,    // image-top → upper in 3D
        z:  nz * b.z * 0.55,
    };

    // Tumor size proportional to bbox extent, with a higher floor so the
    // tumor is ALWAYS clearly visible against the brain (even tiny LIDC
    // nodules). Scaled relative to brain bounds, not absolute units.
    var bw = bbox.xmax - bbox.xmin + 1;
    var bh = bbox.ymax - bbox.ymin + 1;
    var bd = bbox.zmax - bbox.zmin + 1;
    var minVis = Math.min(b.x, b.y, b.z) * 0.06;   // ≥6% of brain min half-axis
    var maxVis = Math.min(b.x, b.y, b.z) * 0.22;   // ≤22% (was 45 — too big, ate the whole brain)
    var sizeX = Math.max(minVis, Math.min(maxVis, bw / 128 * b.x * 0.9));
    var sizeY = Math.max(minVis, Math.min(maxVis, bh / 128 * b.y * 0.9));
    var sizeZ = Math.max(minVis, Math.min(maxVis, bd / 128 * b.z * 0.9));

    // Irregular tumor blob — start from a sphere then vertex-displace using
    // multi-octave sinusoidal noise. Not as accurate as a marching-cubes
    // mesh, but reads as a tumor-shaped MASS rather than a perfect ball.
    var blobGeo = new THREE.SphereGeometry(1, 56, 36);
    var pAttr = blobGeo.attributes.position;
    var v = new THREE.Vector3();
    for (var i = 0; i < pAttr.count; i++) {
        v.fromBufferAttribute(pAttr, i);
        var lx = v.x, ly = v.y, lz = v.z;
        var d = 1
            + Math.sin(lx * 4.0) * Math.cos(ly * 3.7) * Math.sin(lz * 4.3) * 0.18
            + Math.sin(lx * 7.3 + 1.2) * Math.cos(ly * 8.1) * Math.sin(lz * 6.9) * 0.10
            + Math.sin(lx * 13.7) * Math.cos(ly * 12.3) * Math.sin(lz * 14.1) * 0.05;
        v.multiplyScalar(d);
        pAttr.setXYZ(i, v.x, v.y, v.z);
    }
    blobGeo.computeVertexNormals();
    var tumor = new THREE.Mesh(blobGeo, new THREE.MeshPhongMaterial({
        color: 0xff2a44, transparent: true, opacity: 0.92,
        emissive: 0xaa0011, emissiveIntensity: 0.75,
        shininess: 50, specular: 0xff8888,
        depthTest: false, depthWrite: false,
    }));
    tumor.scale.set(sizeX, sizeY, sizeZ);
    tumor.position.set(pos.x, pos.y, pos.z);
    tumor.renderOrder = 10;
    brainCustomScene.add(tumor);
    brainCustomTumorParts.push(tumor);

    // Single subtle glow halo (was 4 shells — too big, ate the whole brain)
    var glowR = Math.max(sizeX, sizeY, sizeZ);
    var halo = new THREE.Mesh(
        new THREE.SphereGeometry(glowR * 1.4, 24, 16),
        new THREE.MeshBasicMaterial({
            color: 0xff3355, transparent: true, opacity: 0.20,
            side: THREE.BackSide,
            depthTest: false, depthWrite: false,
        })
    );
    halo.position.copy(tumor.position);
    halo.renderOrder = 9;
    brainCustomScene.add(halo);
    brainCustomTumorParts.push(halo);

    // Point light at tumor — illuminates surrounding cortex with red glow
    var light = new THREE.PointLight(0xff2244, 1.6, glowR * 4);
    light.position.copy(tumor.position);
    brainCustomScene.add(light);
    brainCustomTumorParts.push(light);

    console.log('[BrainTumor] placed at', pos, 'size', [sizeX, sizeY, sizeZ],
                'brain bounds', b);
}

function animateBrainCustom() {
    if (!brainCustomInited && !brainCustomScene) return;
    brainCustomAnimId = requestAnimationFrame(animateBrainCustom);
    var t = performance.now() * 0.001;

    // Pulse tumor cluster — emissive intensity ripple across all class meshes
    // (NCR / ED / ET), plus subtle scale pulse on the first part to give the
    // whole cluster a "living" feel through the translucent brain.
    if (brainCustomTumorParts.length > 0) {
        var emPulse = 0.55 + Math.sin(t * 2.0) * 0.30;
        brainCustomTumorParts.forEach(function (part, idx) {
            if (part.material && part.material.emissive &&
                part.userData && part.userData.baseEmissive != null) {
                part.material.emissiveIntensity = emPulse;
            }
        });
        var lead = brainCustomTumorParts[0];
        if (lead && lead.isInstancedMesh) {
            if (!lead.userData.baseScale) lead.userData.baseScale = lead.scale.clone();
            var bs = lead.userData.baseScale;
            var pulse = 1 + Math.sin(t * 2.4) * 0.04;
            lead.scale.set(bs.x * pulse, bs.y * pulse, bs.z * pulse);
        }
    }

    // Position L/R labels
    if (brainCustomRenderer && brainCustomCamera) {
        var stage = document.getElementById('brainCustom3DStage');
        if (stage) {
            var rect = stage.getBoundingClientRect();
            var b = window._brainBounds || { x: 7, y: 6, z: 8 };
            function place(elId, world) {
                var el = document.getElementById(elId);
                if (!el) return;
                var p = world.clone().project(brainCustomCamera);
                var sx = (p.x * 0.5 + 0.5) * rect.width;
                var sy = (-p.y * 0.5 + 0.5) * rect.height;
                el.style.transform = 'translate(' + (sx - 8) + 'px, ' + (sy - 8) + 'px)';
                el.style.opacity = (p.z > -1 && p.z < 1) ? '0.7' : '0';
            }
            place('brainCustomLabelL', new THREE.Vector3(-b.x * 0.7, 0, 0));
            place('brainCustomLabelR', new THREE.Vector3(+b.x * 0.7, 0, 0));
        }
    }

    if (brainCustomControls) brainCustomControls.update();
    if (brainCustomRenderer && brainCustomScene && brainCustomCamera) {
        brainCustomRenderer.render(brainCustomScene, brainCustomCamera);
    }
}

// ============ BRAIN 3D MODE SWITCHER (Sketchfab / Custom) ============
function setBrain3DMode(mode) {
    var root = document.getElementById('brainVizStage');
    if (!root) return;
    root.querySelectorAll('.brain-3d-mode').forEach(function (b) {
        b.classList.toggle('is-active', b.getAttribute('data-3d-mode') === mode);
    });
    root.querySelectorAll('.brain-3d-pane').forEach(function (p) {
        p.classList.toggle('is-active', p.getAttribute('data-3d-pane') === mode);
        if (p.hasAttribute('hidden') !== false) { p.hidden = false; }
        if (p.getAttribute('data-3d-pane') !== mode) p.hidden = true;
        else p.hidden = false;
    });
    // Re-fit Three.js canvas when its pane becomes visible. Also lazy-init
    // Sketchfab the first time user picks that tab — we no longer auto-init
    // on page load because the Glass Brain model often renders invisible.
    if (mode === 'custom') {
        setTimeout(function () {
            var stage = document.getElementById('brainCustom3DStage');
            if (stage && brainCustomRenderer && brainCustomCamera) {
                var r = stage.getBoundingClientRect();
                brainCustomRenderer.setSize(r.width, r.height, false);
                brainCustomCamera.aspect = r.width / r.height;
                brainCustomCamera.updateProjectionMatrix();
            } else if (!brainCustomInited) {
                initBrainCustom3D(null);
            }
        }, 50);
    } else if (mode === 'sketchfab') {
        setTimeout(initBrainSketchfab, 80);
    }
}
(function () {
    if (window.__brain3DModeWired) return;
    window.__brain3DModeWired = true;
    document.addEventListener('click', function (e) {
        var btn = e.target.closest && e.target.closest('.brain-3d-mode');
        if (!btn) return;
        var mode = btn.getAttribute('data-3d-mode');
        if (mode) setBrain3DMode(mode);
    });
})();

// Track Sketchfab annotation IDs so we can clear them on re-analysis
var brainSketchfabAnnotationIds = [];

// Three.js overlay scene that paints a red tumor ON TOP of the Sketchfab
// iframe and stays synced with their camera — this is how we get a
// "tumor inside the brain" effect even though we can't inject geometry
// into the sandboxed iframe.
var brainSfOverlay = {
    scene: null, camera: null, renderer: null,
    tumor: null, glows: [],
    rafId: null, polling: false,
    // Geometry state
    tumorOffset: null,       // {x,y,z} normalized in [-0.5, 0.5] relative to brain radius
    tumorRadiusNorm: 0.08,   // tumor radius as fraction of brain radius
    scaleRef: 0,             // brain radius in Sketchfab world units (locked on 1st valid poll)
    pollTick: 0,             // count for periodic FOV resync
};

function disposeBrainSfOverlay() {
    if (brainSfOverlay.rafId) { cancelAnimationFrame(brainSfOverlay.rafId); brainSfOverlay.rafId = null; }
    if (brainSfOverlay.renderer) { try { brainSfOverlay.renderer.dispose(); } catch (e) {} brainSfOverlay.renderer = null; }
    brainSfOverlay.scene = null;
    brainSfOverlay.camera = null;
    brainSfOverlay.tumor = null;
    brainSfOverlay.glows = [];
    brainSfOverlay.tumorOffset = null;
    brainSfOverlay.scaleRef = 0;
    brainSfOverlay.brainCenter = null;
    brainSfOverlay.pollTick = 0;
}

function initBrainSfTumorOverlay(tumorData) {
    if (typeof THREE === 'undefined') return;
    var canvas = document.getElementById('brainSfTumorOverlay');
    var stage = document.getElementById('brainVizStage');
    if (!canvas || !stage) return;
    if (!brainSketchfabApi) {
        // Sketchfab not ready yet — retry shortly
        setTimeout(function () { initBrainSfTumorOverlay(tumorData); }, 600);
        return;
    }

    disposeBrainSfOverlay();
    var rect = stage.getBoundingClientRect();
    var w = Math.max(rect.width, 320), h = Math.max(rect.height, 240);

    brainSfOverlay.scene = new THREE.Scene();
    brainSfOverlay.camera = new THREE.PerspectiveCamera(45, w / h, 0.001, 1000);
    brainSfOverlay.renderer = new THREE.WebGLRenderer({
        canvas: canvas, antialias: true, alpha: true,
    });
    brainSfOverlay.renderer.setClearColor(0x000000, 0);
    brainSfOverlay.renderer.setSize(w, h, false);
    brainSfOverlay.renderer.setPixelRatio(window.devicePixelRatio > 1 ? 2 : 1);

    // Normalize tumor centroid (BraTS 128³ voxel space) → axis-swapped offset
    // suitable for a Y-up viewer:
    //   voxel x (R-L)   → world x  (flipped: voxel +x is anatomical-left → world -x)
    //   voxel z (S-I)   → world y  (superior up)
    //   voxel y (A-P)   → world z  (anterior toward viewer)
    var c = tumorData && tumorData.centroid128 ? tumorData.centroid128 : { x: 64, y: 64, z: 64 };
    var nx = (c.x / 128) - 0.5;   // [-0.5, 0.5]
    var ny = (c.y / 128) - 0.5;
    var nz = (c.z / 128) - 0.5;
    brainSfOverlay.tumorOffset = {
        x: -nx * 0.95,
        y:  nz * 0.85,
        z:  ny * 0.85,
    };

    // Tumor size scales with detected volume (kept conservative so it stays
    // visibly inside the brain even when our scaleRef heuristic is off).
    var volume = (tumorData && tumorData.volumeCm3) ? tumorData.volumeCm3 : 20;
    brainSfOverlay.tumorRadiusNorm = Math.max(0.06, Math.min(0.14, Math.log(volume + 1) * 0.035));

    // Tumor core (unit sphere — scaled per-frame in poll)
    brainSfOverlay.tumor = new THREE.Mesh(
        new THREE.SphereGeometry(1, 32, 24),
        new THREE.MeshBasicMaterial({
            color: 0xff3355, transparent: true, opacity: 0.92,
            depthTest: false, depthWrite: false,
        })
    );
    brainSfOverlay.tumor.renderOrder = 10;
    brainSfOverlay.scene.add(brainSfOverlay.tumor);

    // Glow halo shells (BackSide so they read as radiance through tissue)
    [1.6, 2.4, 3.2].forEach(function (s, i) {
        var g = new THREE.Mesh(
            new THREE.SphereGeometry(1, 20, 14),
            new THREE.MeshBasicMaterial({
                color: 0xff3355, transparent: true,
                opacity: 0.25 - i * 0.07, side: THREE.BackSide,
                depthTest: false, depthWrite: false,
            })
        );
        g.renderOrder = 9 - i;
        brainSfOverlay.scene.add(g);
        brainSfOverlay.glows.push({ mesh: g, sFactor: s });
    });

    // Sync FOV once at init (kept refreshed in poll every ~30 frames)
    if (brainSketchfabApi.getFov) {
        try {
            brainSketchfabApi.getFov(function (err, fov) {
                if (!err && fov && brainSfOverlay.camera) {
                    brainSfOverlay.camera.fov = fov;
                    brainSfOverlay.camera.updateProjectionMatrix();
                }
            });
        } catch (e) {}
    }

    brainSfOverlay.polling = true;
    pollSketchfabCameraAndRender();
}

// Poll Sketchfab camera each frame + render overlay synced to it.
// Sketchfab's getCameraLookAt is async (callback-based) — we kick off a new
// poll on each callback to keep up with 60fps as best we can. The brain is
// always at world origin in Sketchfab, so we anchor the tumor to (0,0,0)
// scaled by a brain-radius reference computed from the initial camera dolly.
function pollSketchfabCameraAndRender() {
    if (!brainSfOverlay.polling || !brainSfOverlay.renderer || !brainSketchfabApi) {
        return;
    }
    try {
        brainSketchfabApi.getCameraLookAt(function (err, lookAt) {
            if (err || !lookAt || !brainSfOverlay.camera) {
                brainSfOverlay.rafId = requestAnimationFrame(pollSketchfabCameraAndRender);
                return;
            }
            var cam = brainSfOverlay.camera;
            var p = lookAt.position, tg = lookAt.target;
            cam.position.set(p[0], p[1], p[2]);
            cam.up.set(0, 1, 0);                       // Sketchfab default: Y-up
            cam.lookAt(tg[0], tg[1], tg[2]);

            // Lock brain center + scale on first valid poll. Sketchfab auto-fits
            // the model so initial target ≈ model center, and initial
            // |camera - target| ≈ 2.4 × brain radius. We use the target (not
            // world origin) because some models aren't centered at (0,0,0).
            var dx = p[0] - tg[0], dy = p[1] - tg[1], dz = p[2] - tg[2];
            var distTarget = Math.sqrt(dx*dx + dy*dy + dz*dz);
            if (!brainSfOverlay.scaleRef && distTarget > 0.05) {
                brainSfOverlay.scaleRef = distTarget / 2.4;
                brainSfOverlay.brainCenter = [tg[0], tg[1], tg[2]];
            }
            var R = brainSfOverlay.scaleRef || (distTarget / 2.4) || 1.0;
            var C = brainSfOverlay.brainCenter || [tg[0], tg[1], tg[2]];

            // Periodically refresh FOV (user may have changed it via API/zoom)
            brainSfOverlay.pollTick = (brainSfOverlay.pollTick || 0) + 1;
            if (brainSfOverlay.pollTick % 30 === 0 && brainSketchfabApi.getFov) {
                try {
                    brainSketchfabApi.getFov(function (e2, fov) {
                        if (!e2 && fov && brainSfOverlay.camera &&
                            Math.abs(brainSfOverlay.camera.fov - fov) > 0.1) {
                            brainSfOverlay.camera.fov = fov;
                            brainSfOverlay.camera.updateProjectionMatrix();
                        }
                    });
                } catch (e3) {}
            }

            // Resize if stage changed
            var stage = document.getElementById('brainVizStage');
            if (stage) {
                var rect = stage.getBoundingClientRect();
                if (Math.abs(brainSfOverlay.renderer.domElement.width - rect.width) > 1 ||
                    Math.abs(brainSfOverlay.renderer.domElement.height - rect.height) > 1) {
                    brainSfOverlay.renderer.setSize(rect.width, rect.height, false);
                    cam.aspect = rect.width / rect.height;
                    cam.updateProjectionMatrix();
                }
            }

            // Tumor placement: world position = brainCenter + offset × R
            var off = brainSfOverlay.tumorOffset || { x: 0, y: 0, z: 0 };
            var wx = C[0] + off.x * R, wy = C[1] + off.y * R, wz = C[2] + off.z * R;
            var rBase = brainSfOverlay.tumorRadiusNorm * R;

            var t = performance.now() * 0.001;
            var pulse = 1 + Math.sin(t * 2.4) * 0.12;
            if (brainSfOverlay.tumor) {
                brainSfOverlay.tumor.position.set(wx, wy, wz);
                brainSfOverlay.tumor.scale.setScalar(rBase * pulse);
            }
            brainSfOverlay.glows.forEach(function (gi, i) {
                gi.mesh.position.set(wx, wy, wz);
                gi.mesh.scale.setScalar(rBase * gi.sFactor * pulse);
                if (gi.mesh.material) {
                    gi.mesh.material.opacity = (0.25 - i * 0.07) * (0.6 + Math.sin(t * 1.6 + i) * 0.35);
                }
            });

            brainSfOverlay.renderer.render(brainSfOverlay.scene, brainSfOverlay.camera);
            brainSfOverlay.rafId = requestAnimationFrame(pollSketchfabCameraAndRender);
        });
    } catch (e) {
        brainSfOverlay.rafId = requestAnimationFrame(pollSketchfabCameraAndRender);
    }
}

// Cinematic move + annotation + floating overlay when an analysis completes.
// Camera move and annotation placement are deferred until the Three.js overlay
// has computed the brain-radius reference (scaleRef) from Sketchfab's actual
// camera distance — so the pin lands at the same world position as the red
// tumor mesh we render in the overlay.
function highlightBrainTumor(diameterMm, locationHint, tumorData) {
    var ovEl = document.getElementById('brainSketchfabOverlay');
    var szEl = document.getElementById('brainSketchfabOverlaySize');
    if (ovEl && szEl) {
        ovEl.hidden = false;
        szEl.textContent = (typeof diameterMm === 'number'
            ? diameterMm.toFixed(1) + ' mm' : (diameterMm || '—'));
    }

    if (!tumorData) return;

    // Clear previous Sketchfab annotations first
    if (brainSketchfabApi && brainSketchfabApi.removeAnnotation) {
        brainSketchfabAnnotationIds.forEach(function (id) {
            try { brainSketchfabApi.removeAnnotation(id, function () {}); } catch (e) {}
        });
        brainSketchfabAnnotationIds = [];
    }

    // Start the Three.js overlay (this kicks off polling + computes scaleRef)
    initBrainSfTumorOverlay(tumorData);
    // Custom mode: if brain already loaded, just swap the tumor (fast — no
    // GLB reload). Otherwise do full init.
    if (brainCustomInited && brainCustomModel) {
        updateBrainCustomTumor(tumorData);
    } else {
        initBrainCustom3D(tumorData);
    }

    // Wait until scaleRef + brainCenter are known, then place annotation pin
    // at the exact same world location as the Three.js tumor mesh.
    // We deliberately do NOT call setCameraLookAt or gotoAnnotation here —
    // those move the Sketchfab camera, and an incorrectly placed cinematic
    // (or auto-goto annotation that snaps inside the model) makes the brain
    // invisible. The overlay tumor follows whatever camera Sketchfab uses,
    // so the user can orbit normally.
    var waited = 0;
    function placeWhenReady() {
        if (!brainSketchfabApi || !brainSfOverlay.scaleRef ||
            !brainSfOverlay.tumorOffset || !brainSfOverlay.brainCenter) {
            if (waited++ < 40) {   // up to ~4s
                setTimeout(placeWhenReady, 100);
            }
            return;
        }
        var R = brainSfOverlay.scaleRef;
        var off = brainSfOverlay.tumorOffset;
        var C = brainSfOverlay.brainCenter;
        var wx = C[0] + off.x * R, wy = C[1] + off.y * R, wz = C[2] + off.z * R;

        // Annotation pin at the tumor center (no auto-goto — leaving camera
        // alone so the user keeps the default fit-to-view of the brain)
        if (brainSketchfabApi.createAnnotationFromScenePosition) {
            var cc = tumorData.classCounts || {};
            var title = 'Khối u' + (typeof diameterMm === 'number' ? ' · ' + diameterMm.toFixed(1) + ' mm' : '');
            var desc = 'Thể tích: ' + (tumorData.volumeCm3 || 0) + ' cm³\n' +
                       'NCR: ' + (cc.NCR || 0) + ' · ED: ' + (cc.ED || 0) + ' · ET: ' + (cc.ET || 0) + '\n' +
                       'Confidence: ' + (tumorData.confidence || 0) + '%';
            try {
                brainSketchfabApi.createAnnotationFromScenePosition(
                    [wx, wy, wz], [0, 1, 0], title, desc, null,
                    function (err, idx) {
                        if (!err && idx != null) {
                            brainSketchfabAnnotationIds.push(idx);
                        }
                    }
                );
            } catch (e) { console.warn('annotation failed:', e); }
        }
    }
    setTimeout(placeWhenReady, 200);
}

// ============ BRAIN MODULE TABS (U-Net 3D / YOLOv8 2D) ============
(function () {
    if (window.__brainTabsWired) return;
    window.__brainTabsWired = true;
    document.addEventListener('click', function (e) {
        var btn = e.target.closest && e.target.closest('.brain-tab');
        if (!btn) return;
        var tab = btn.getAttribute('data-brain-tab');
        if (!tab) return;
        var module = document.getElementById('module-brain');
        module.querySelectorAll('.brain-tab').forEach(function (t) {
            t.classList.toggle('is-active', t.getAttribute('data-brain-tab') === tab);
        });
        module.querySelectorAll('.brain-panel').forEach(function (p) {
            p.classList.toggle('is-active', p.getAttribute('data-brain-panel') === tab);
        });
    });
})();

// ============ BRAIN YOLO (2D detection) ============
async function runBrainYolo() {
    var fi = document.getElementById('yoloFileInput');
    if (!fi || !fi.files || fi.files.length === 0) {
        showToast('Chọn ảnh MRI trước', 'error');
        return;
    }
    var statusEl = document.getElementById('yoloStatus');
    if (statusEl) { statusEl.hidden = true; statusEl.textContent = ''; }

    // Use brain loading overlay but override steps for YOLO pipeline
    BRAIN_LOADING_STEPS['yolo'] = [
        { text: 'Đọc ảnh MRI...', tech: 'PIL · numpy', pct: 20 },
        { text: 'Resize · letterbox 640×640...', tech: 'ultralytics preprocessing', pct: 40 },
        { text: 'YOLOv8 detection inference...', tech: 'PyTorch · single-shot detector', pct: 75 },
        { text: 'Non-max suppression · vẽ bounding box...', tech: 'NMS IoU=0.45', pct: 95 },
    ];
    startBrainLoading('yolo');

    try {
        var fd = new FormData();
        fd.append('image', fi.files[0]);
        var apiBase = (window.APP_CONFIG && window.APP_CONFIG.API_BASE) || '';
        var resp = await fetch(apiBase + '/api/predict-brain-yolo', { method: 'POST', body: fd });
        var r = await resp.json();
        if (r.error) throw new Error(r.error + (r.detail ? ': ' + r.detail : ''));
        displayYoloResult(r);
        showToast('⚡ Phát hiện xong (' + (r.detections.length) + ' đối tượng)', 'success');
    } catch (e) {
        console.error('YOLO error:', e);
        showToast('Lỗi: ' + e.message, 'error');
    } finally {
        stopBrainLoading();
    }
}

function displayYoloResult(r) {
    var canvas = document.getElementById('yoloCanvas');
    if (canvas && r.overlayImage) {
        canvas.innerHTML = '<img src="' + r.overlayImage + '" ' +
            'style="width:100%;height:100%;object-fit:contain;border-radius:8px;" alt="YOLO Detection">';
    }

    document.getElementById('yoloCount').textContent = r.detections.length;
    document.getElementById('yoloTopClass').textContent = r.topClass || '—';
    document.getElementById('yoloTopConf').textContent = r.topConfidence
        ? (r.topConfidence * 100).toFixed(1) + '%' : '—';
    document.getElementById('yoloTime').textContent = (r.inferenceTimeS || 0).toFixed(2) + ' s';
    document.getElementById('yoloModelName').textContent = (r.modelInfo && r.modelInfo.weights) || 'YOLOv8';

    var list = document.getElementById('yoloDetectionList');
    if (list) {
        if (r.detections.length === 0) {
            list.innerHTML = '<li class="reco-item empty">Không phát hiện đối tượng nào</li>';
        } else {
            list.innerHTML = r.detections.map(function (d, i) {
                var bb = d.bbox.map(function (v) { return Math.round(v); }).join(', ');
                return '<li class="reco-item">' +
                    '<b>#' + (i + 1) + ' ' + d.class + '</b>' +
                    ' · ' + (d.confidence * 100).toFixed(1) + '%' +
                    ' <small style="display:block;color:var(--text-muted);font-family:monospace;">bbox: ' + bb + '</small>' +
                    '</li>';
            }).join('');
        }
    }

    // Cinematic Sketchfab move + overlay (YOLO doesn't give 3D coords, use class as seed)
    if (r.detections.length > 0) {
        var bbCenter = r.detections[0].bbox;
        var bboxDiag = Math.sqrt(
            Math.pow(bbCenter[2] - bbCenter[0], 2) +
            Math.pow(bbCenter[3] - bbCenter[1], 2)
        );
        // Synthesize a minimal tumorData for the 3D viewers (no 3D coords from YOLO)
        var hash = 0, seed = r.topClass || 'yolo';
        for (var i = 0; i < seed.length; i++) hash = ((hash << 5) - hash + seed.charCodeAt(i)) | 0;
        var fakeData = {
            detected: true,
            volumeCm3: Math.round(bboxDiag * 0.5),
            confidence: Math.round((r.topConfidence || 0) * 100),
            classCounts: { NCR: 0, ED: 0, ET: r.detections.length },
            centroid128: {
                x: 40 + ((hash & 0x7f) % 50),
                y: 40 + (((hash >> 7) & 0x7f) % 50),
                z: 50 + (((hash >> 14) & 0x7f) % 30),
            },
            bbox128: { xmin: 50, xmax: 78, ymin: 50, ymax: 78, zmin: 60, zmax: 75 },
        };
        highlightBrainTumor('~' + Math.round(bboxDiag) + ' px',
                            'yolo-' + (r.topClass || ''), fakeData);
    }

    // Warning banner if fallback COCO model used
    var statusEl = document.getElementById('yoloStatus');
    if (statusEl && r.warning) {
        statusEl.hidden = false;
        statusEl.innerHTML = '⚠️ <b>Chưa có brain-tumor weights</b>. Đang dùng YOLOv8n COCO (detect đồ vật chung). ' +
            'Tải weights brain-tumor về <code>models/brain/yolo_brain_tumor.pt</code> để kết quả chính xác.';
    } else if (statusEl) {
        statusEl.hidden = true;
    }
}

// ============ BRAIN TUMOR (MRI) — Real AI (3D U-Net BraTS) ============
// Detect modality from filename so we can label NIfTI uploads correctly.
function _brainModalityFromFilename(name) {
    var n = (name || '').toLowerCase();
    if (n.indexOf('t1ce') >= 0 || n.indexOf('t1c') >= 0 || n.indexOf('t1gd') >= 0) return 't1ce';
    if (n.indexOf('flair') >= 0) return 'flair';
    if (n.indexOf('t2') >= 0) return 't2';
    if (n.indexOf('t1') >= 0) return 't1';
    return null;
}

// ============ BRAIN MODEL SELECTOR (hot-swap between checkpoints) ============
async function loadBrainModelsList() {
    var sel = document.getElementById('brainModelSelect');
    var meta = document.getElementById('brainModelMeta');
    var statusEl = document.getElementById('brainModelSelectorStatus');
    if (!sel) return;
    try {
        var apiBase = (window.APP_CONFIG && window.APP_CONFIG.API_BASE) || '';
        var resp = await fetch(apiBase + '/api/brain-models');
        console.log('[BrainModel] /api/brain-models status:', resp.status);
        var rawText = await resp.text();
        console.log('[BrainModel] response (first 300 chars):', rawText.slice(0, 300));
        if (!resp.ok) {
            throw new Error('HTTP ' + resp.status + ': ' + rawText.slice(0, 100));
        }
        var data;
        try { data = JSON.parse(rawText); }
        catch (parseErr) { throw new Error('Invalid JSON: ' + rawText.slice(0, 100)); }
        if (data.error) {
            throw new Error(data.error + (data.detail ? ' (' + data.detail + ')' : ''));
        }
        if (!data || !Array.isArray(data.models)) {
            throw new Error('Missing models field. Got keys: ' + Object.keys(data || {}).join(','));
        }

        sel.innerHTML = '';
        var groups = { production: [], experimental: [], checkpoint: [] };
        data.models.forEach(function (m) {
            if (m.filename.indexOf('models_3D/') === 0) groups.checkpoint.push(m);
            else if (m.badge === 'Experimental') groups.experimental.push(m);
            else groups.production.push(m);
        });

        function appendGroup(label, items) {
            if (!items.length) return;
            var og = document.createElement('optgroup');
            og.label = label;
            items.forEach(function (m) {
                var opt = document.createElement('option');
                opt.value = m.filename;
                var badge = m.badge ? ' [' + m.badge + ']' : '';
                opt.textContent = m.displayName + ' — ' + m.subtitle + badge;
                if (m.active) opt.selected = true;
                og.appendChild(opt);
            });
            sel.appendChild(og);
        }
        appendGroup('Production', groups.production);
        appendGroup('Experimental', groups.experimental);
        appendGroup('2018 Training Checkpoints', groups.checkpoint);

        var active = data.models.find(function (m) { return m.active; });
        if (active && meta) {
            meta.textContent = '✓ ' + active.displayName + ' · ' + active.sizeMb + ' MB';
        }
        if (statusEl) {
            statusEl.textContent = data.currentLoaded ? 'Loaded' : 'Not loaded';
            statusEl.className = 'brain-model-status ' + (data.currentLoaded ? 'is-ok' : 'is-warn');
        }
    } catch (e) {
        console.error('[BrainModel] List failed:', e);
        sel.innerHTML = '<option value="">— Lỗi load model list —</option>';
        if (meta) meta.textContent = e.message;
    }
}

async function switchBrainModel(filename) {
    var sel = document.getElementById('brainModelSelect');
    var meta = document.getElementById('brainModelMeta');
    var statusEl = document.getElementById('brainModelSelectorStatus');
    if (!sel || !filename) return;
    sel.disabled = true;
    if (meta) {
        meta.textContent = '⏳ Đang load model...';
        meta.classList.add('is-loading');
    }
    if (statusEl) {
        statusEl.textContent = 'Switching';
        statusEl.className = 'brain-model-status is-switching';
    }
    try {
        var apiBase = (window.APP_CONFIG && window.APP_CONFIG.API_BASE) || '';
        var resp = await fetch(apiBase + '/api/brain-model-switch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: filename }),
        });
        var data = await resp.json();
        if (data.success) {
            var sh = (data.inputShape || []).slice(1).join('×');
            var p = (data.params || 0).toLocaleString();
            var pipeBadge = data.pipelineType === '2d-seg' ? ' [2D]' :
                            data.pipelineType === '3d-seg' ? ' [3D]' : '';
            if (meta) meta.textContent = '✓ Loaded' + pipeBadge + ': ' + p + ' params · input ' + sh + ' · ' + data.loadTimeS + 's';
            if (statusEl) {
                statusEl.textContent = 'Loaded' + pipeBadge.trim();
                statusEl.className = 'brain-model-status is-ok';
            }
            showToast('✓ Đã chuyển sang model mới' + pipeBadge, 'success');
        } else {
            var errMsg = data.error || 'Load failed';
            // Detect "incompatible architecture" vs other failures
            var incompat = (errMsg.indexOf('not a segmentation') >= 0 ||
                           errMsg.indexOf('not compatible') >= 0 ||
                           errMsg.indexOf('Classifier') >= 0);
            if (meta) {
                meta.innerHTML = '✗ <b>Incompatible</b><br>' +
                    '<small style="opacity:0.7">' + errMsg.replace(/</g, '&lt;') + '</small>' +
                    (data.inputShape ? '<br><code style="font-size:9px">input ' + data.inputShape.join('×') + ' → output ' + (data.outputShape || []).join('×') + '</code>' : '');
            }
            if (statusEl) {
                statusEl.textContent = incompat ? 'Incompatible' : 'Error';
                statusEl.className = 'brain-model-status is-error';
            }
            showToast(incompat
                ? '⚠️ Model không tương thích pipeline segmentation'
                : 'Lỗi: ' + errMsg, 'error');
            // Reload list to show actual current model (rollback indicator)
            await loadBrainModelsList();
        }
    } catch (e) {
        if (meta) meta.textContent = '✗ Network error: ' + e.message;
        showToast('Lỗi kết nối: ' + e.message, 'error');
    } finally {
        sel.disabled = false;
        if (meta) meta.classList.remove('is-loading');
    }
}

// Wire selector + auto-load on first brain-module open
(function () {
    if (window.__brainModelSelectorWired) return;
    window.__brainModelSelectorWired = true;
    document.addEventListener('change', function (e) {
        if (e.target && e.target.id === 'brainModelSelect' && e.target.value) {
            switchBrainModel(e.target.value);
        }
    });
    document.addEventListener('click', function (e) {
        var btn = e.target.closest && e.target.closest('.nav-module');
        if (!btn) return;
        if (btn.dataset && btn.dataset.module === 'brain') {
            // Lazy-load model list once user opens the brain module
            setTimeout(loadBrainModelsList, 250);
        }
    });
    // Also load on DOM ready if brain module is already visible
    function maybeLoad() {
        var mb = document.getElementById('module-brain');
        if (mb && !mb.classList.contains('hidden')) {
            loadBrainModelsList();
        }
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () { setTimeout(maybeLoad, 400); });
    } else {
        setTimeout(maybeLoad, 400);
    }
})();

async function runBrainAnalysis() {
    // Fire loading IMMEDIATELY on click so the UI feels responsive — we use
    // the 'init' pipeline (1 step) until we've validated input and know
    // whether the user uploaded NIfTI or a single 2D image. Once known, we
    // re-call startBrainLoading() with the correct pipeline.
    startBrainLoading('init');
    _setBrainAnalyzeBtnBusy(true);

    var fi = document.getElementById('brainFileInput');
    if (!fi || !fi.files || fi.files.length === 0) {
        _abortBrainLoading();
        showToast('Chọn ảnh MRI (NIfTI .nii/.nii.gz hoặc PNG/JPG) trước', 'error');
        return;
    }

    // Split files: ground-truth seg goes in a separate field for Dice scoring,
    // 4 MRI modalities are the actual model input.
    var allFiles = Array.from(fi.files);
    var segFile  = allFiles.find(function (f) { return /_?seg(\.|_|$)/.test(f.name.toLowerCase()); });
    var files = allFiles.filter(function (f) { return f !== segFile; });
    if (files.length === 0) {
        _abortBrainLoading();
        showToast('Chỉ có file *_seg — cần upload thêm 4 file MRI (flair/t1/t1ce/t2)', 'error');
        return;
    }
    var fd = new FormData();
    var isNiftiInput = files.some(f => /\.nii(\.gz)?$/i.test(f.name));

    if (isNiftiInput) {
        var assigned = {};
        files.forEach(function (f, idx) {
            if (!/\.nii(\.gz)?$/i.test(f.name)) return;
            var mod = _brainModalityFromFilename(f.name);
            if (!mod) {
                mod = ['flair', 't1', 't1ce', 't2'][idx] || 'flair';
            }
            if (!assigned[mod]) {
                fd.append(mod, f);
                assigned[mod] = true;
            }
        });
        if (Object.keys(assigned).length === 0) {
            _abortBrainLoading();
            showToast('Không nhận được file NIfTI hợp lệ', 'error');
            return;
        }
        // Ground truth (if user uploaded it) — server uses for Dice scoring
        if (segFile) {
            fd.append('gt_seg', segFile);
            console.log('[Brain] Ground-truth seg attached:', segFile.name);
        }
    } else {
        fd.append('image', files[0]);
    }

    // Now we know the pipeline — switch loading steps from 'init' to real one
    startBrainLoading(isNiftiInput ? 'nifti' : '2d');

    try {
        var apiBase = (window.APP_CONFIG && window.APP_CONFIG.API_BASE) || '';
        var resp = await fetch(apiBase + '/api/predict-brain', { method: 'POST', body: fd });
        var result = await resp.json();
        if (result.error) throw new Error(result.error + (result.detail ? ': ' + result.detail : ''));
        displayBrainResult(result, isNiftiInput);
        showToast('🧠 Phân tích xong', 'success');
    } catch (e) {
        console.error('Brain analysis error:', e);
        showToast('Lỗi: ' + e.message, 'error');
    } finally {
        stopBrainLoading();
        _setBrainAnalyzeBtnBusy(false);
    }
}

// Helper: toggle the "Phân tích bằng 3D U-Net" button between idle + busy
// state. Disabled + spinner icon + greyed-out text while inference runs.
function _setBrainAnalyzeBtnBusy(busy) {
    var btns = document.querySelectorAll('button[onclick*="runBrainAnalysis"]');
    btns.forEach(function (b) {
        if (busy) {
            if (!b.dataset.origHtml) b.dataset.origHtml = b.innerHTML;
            b.disabled = true;
            b.classList.add('is-busy');
            b.innerHTML = '<span class="brain-btn-spinner"></span> Đang phân tích...';
        } else {
            b.disabled = false;
            b.classList.remove('is-busy');
            if (b.dataset.origHtml) b.innerHTML = b.dataset.origHtml;
        }
    });
}

// Abort loading WITHOUT the "✓ Hoàn tất" flash (used when input validation
// fails — there was no real work, no need to fake completion).
function _abortBrainLoading() {
    if (_brainLoadingTimer) { clearInterval(_brainLoadingTimer); _brainLoadingTimer = null; }
    var ov = document.getElementById('brainLoadingOverlay');
    if (ov) ov.classList.remove('active');
    _setBrainAnalyzeBtnBusy(false);
}

// ============ BRAIN ANALYSIS LOADING ANIMATION ============
// Cycles through realistic pipeline steps while the backend runs the 3D
// U-Net inference. Progress bar advances stage-by-stage; on success we
// snap to 100% and fade out. The actual step durations are heuristic
// (BraTS inference takes ~3-12s depending on hardware) — the animation
// just needs to FEEL alive, not match real backend timing.
var BRAIN_LOADING_STEPS = {
    'init': [
        { text: 'Đang kiểm tra dữ liệu MRI...', tech: 'validating modality files', pct: 4 },
    ],
    'nifti': [
        { text: 'Đọc file NIfTI (FLAIR · T1 · T1c · T2)...', tech: 'nibabel · 4 modalities', pct: 8 },
        { text: 'Z-score normalize per modality...', tech: 'z = (x - μ) / σ trên brain voxels', pct: 18 },
        { text: 'Brain-bbox crop → resize về 128³...', tech: 'auto-detect brain extent · scipy.zoom', pct: 28 },
        { text: 'Forward pass với 4-way TTA...', tech: 'flips D/H/W · batched predict · avg softmax', pct: 55 },
        { text: 'Hậu xử lý: smooth + morphology + CC filter...', tech: 'gaussian σ=0.8 · closing · top-K CCs', pct: 75 },
        { text: 'Marching cubes per-class mesh...', tech: 'skimage · NCR · ED · ET isosurfaces', pct: 86 },
        { text: 'Render multi-view + Dice scoring...', tech: 'nilearn plot_roi · BraTS metrics', pct: 95 },
    ],
    '2d': [
        { text: 'Đọc ảnh MRI...', tech: 'PIL · numpy ndarray', pct: 18 },
        { text: 'Replicate sang 4 channel modality...', tech: 'demo pipeline · single-image', pct: 30 },
        { text: 'Resize về 128×128×128...', tech: 'opencv · INTER_AREA', pct: 42 },
        { text: 'AI đang suy luận 3D U-Net...', tech: 'TensorFlow forward pass', pct: 68 },
        { text: 'Tổng hợp segmentation 4-class...', tech: 'softmax → argmax', pct: 84 },
        { text: 'Render multi-view overlay...', tech: 'nilearn plot_roi', pct: 95 },
    ],
};
var _brainLoadingTimer = null;
var _brainLoadingStepIdx = 0;
var _brainLoadingSteps = [];

function startBrainLoading(mode) {
    var ov = document.getElementById('brainLoadingOverlay');
    if (!ov) return;
    _brainLoadingSteps = BRAIN_LOADING_STEPS[mode] || BRAIN_LOADING_STEPS['2d'];
    _brainLoadingStepIdx = 0;
    ov.classList.add('active');
    _advanceBrainLoadingStep();
    if (_brainLoadingTimer) clearInterval(_brainLoadingTimer);
    // Step cadence: nifti pipeline = single-pass + 4-way TTA (~10-15s).
    // 7 steps × ~1.8s each fits comfortably in that window.
    var stepMs = (mode === 'nifti') ? 1800 : 1600;
    _brainLoadingTimer = setInterval(_advanceBrainLoadingStep, stepMs);
}

function _advanceBrainLoadingStep() {
    var step = _brainLoadingSteps[_brainLoadingStepIdx];
    if (!step) return;
    var stepEl = document.getElementById('brainLoadingStep');
    var techEl = document.getElementById('brainLoadingTech');
    var barEl  = document.getElementById('brainLoadingBar');
    if (stepEl) {
        stepEl.style.opacity = '0';
        setTimeout(function () {
            stepEl.textContent = step.text;
            stepEl.style.opacity = '1';
        }, 180);
    }
    if (techEl) techEl.textContent = step.tech;
    if (barEl)  barEl.style.width  = step.pct + '%';
    if (_brainLoadingStepIdx < _brainLoadingSteps.length - 1) {
        _brainLoadingStepIdx++;
    }
}

function stopBrainLoading() {
    if (_brainLoadingTimer) {
        clearInterval(_brainLoadingTimer);
        _brainLoadingTimer = null;
    }
    var barEl = document.getElementById('brainLoadingBar');
    var stepEl = document.getElementById('brainLoadingStep');
    var ov = document.getElementById('brainLoadingOverlay');
    if (barEl) barEl.style.width = '100%';
    if (stepEl) stepEl.textContent = '✓ Hoàn tất phân tích';
    setTimeout(function () {
        if (ov) ov.classList.remove('active');
    }, 380);
}

function displayBrainResult(result, isNiftiInput) {
    var now = new Date();
    var dateStr = now.toLocaleDateString('vi-VN');
    var pid = 'BRA' + now.getFullYear().toString().slice(2) +
              (now.getMonth()+1).toString().padStart(2,'0') +
              now.getDate().toString().padStart(2,'0') + '_' +
              Math.floor(Math.random()*999).toString().padStart(3,'0');

    document.getElementById('brainPatientId').textContent = pid;
    document.getElementById('brainScanDate').textContent = dateStr;
    document.getElementById('brainInfoId').textContent = pid;
    document.getElementById('brainInfoAge').textContent = '—';
    document.getElementById('brainInfoSex').textContent = '—';
    document.getElementById('brainInfoDate').textContent = dateStr;
    document.getElementById('brainInfoSymptom').textContent = '—';
    document.getElementById('brainInfoHistory').textContent = '—';

    var detected = !!result.detected;
    var conf = result.confidence || 0;
    var volCm3 = result.volumeCm3 || 0;
    var diam = result.maxDiameterMm || 0;

    document.getElementById('brainConfidence').textContent = conf.toFixed(1) + '%';
    document.getElementById('brainVolume').textContent = volCm3.toFixed(2) + ' cm³';
    document.getElementById('brainType').textContent = detected ? 'Nghi ngờ khối u' : 'Không phát hiện';

    // === PREPROCESSING PIPELINE PREVIEW (Kaggle-style 2×3 grid) ===
    var pipelineCard = document.getElementById('brainPipelineCard');
    var pipelineImg  = document.getElementById('brainPipelinePreview');
    if (pipelineCard && pipelineImg) {
        if (result.pipelinePreview) {
            pipelineImg.src = result.pipelinePreview;
            pipelineCard.hidden = false;
        } else {
            pipelineCard.hidden = true;
        }
    }

    // === GROUND TRUTH COMPARISON ===
    var gtCard = document.getElementById('brainGTCard');
    if (gtCard) {
        var gt = result.groundTruth;
        if (gt && gt.present && gt.dice) {
            gtCard.hidden = false;
            // Format Dice as percentage; color-code by quality bands
            function setDice(id, barId, val) {
                var el = document.getElementById(id);
                var bar = document.getElementById(barId);
                if (!el || !bar) return;
                var pct = (val * 100);
                el.textContent = pct.toFixed(1) + '%';
                bar.style.width = Math.min(100, pct) + '%';
                // Color: <50 = poor (red), 50-70 = fair (yellow),
                //        70-85 = good (green), >85 = excellent (cyan)
                var color;
                if (pct < 50)      color = '#ff4466';
                else if (pct < 70) color = '#ffd84a';
                else if (pct < 85) color = '#4ade80';
                else               color = '#22d3ee';
                bar.style.background = color;
                el.style.color = color;
            }
            setDice('brainDiceWT', 'brainDiceWTBar', gt.dice.WT);
            setDice('brainDiceTC', 'brainDiceTCBar', gt.dice.TC);
            setDice('brainDiceET', 'brainDiceETBar', gt.dice.ET);
            document.getElementById('brainGTVolume').textContent =
                (gt.gtVolumeCm3 || 0).toFixed(2) + ' cm³';
            var diff = gt.volumeDiffCm3 || 0;
            var diffEl = document.getElementById('brainGTVolumeDiff');
            diffEl.textContent = (diff >= 0 ? '+' : '') + diff.toFixed(2) + ' cm³';
            diffEl.style.color = Math.abs(diff) < 5 ? '#4ade80' :
                                 Math.abs(diff) < 15 ? '#ffd84a' : '#ff4466';

            // Wire toggle to swap between prediction overlay and GT overlay
            var toggleBtn = document.getElementById('brainGTToggle');
            if (toggleBtn) {
                toggleBtn.dataset.predImg     = result.overlayImage || '';
                toggleBtn.dataset.gtImg       = result.groundTruthOverlay || '';
                toggleBtn.dataset.showingGt   = 'false';
                toggleBtn.onclick = function () {
                    var showingGt = toggleBtn.dataset.showingGt === 'true';
                    var nextGt    = !showingGt;
                    toggleBtn.dataset.showingGt = nextGt ? 'true' : 'false';
                    toggleBtn.textContent      = nextGt ? 'Đang xem GT — bấm để về AI' : 'Hiện overlay GT';
                    toggleBtn.classList.toggle('is-active', nextGt);
                    // Swap multi-view image (the big 1×3 overlay below)
                    var ovImg = document.querySelector('#brainOverlayCanvas img');
                    if (ovImg) ovImg.src = nextGt
                        ? toggleBtn.dataset.gtImg
                        : toggleBtn.dataset.predImg;
                };
            }
        } else {
            gtCard.hidden = true;
        }
    }

    document.getElementById('brainSize').textContent = volCm3.toFixed(2) + ' cm³';
    var locEl = document.getElementById('brainLocation');
    var locSubEl = document.getElementById('brainLocationSub');
    if (locEl) locEl.textContent = detected ? 'Đã định vị (xem 3D)' : '—';
    if (locSubEl) locSubEl.textContent = '';
    var c = result.centroid128 || { x: 0, y: 0, z: 0 };
    document.getElementById('brainCoords').textContent =
        detected ? (c.x + ', ' + c.y + ', ' + c.z + ' (128³ space)') : '—';
    document.getElementById('brainGrade').textContent =
        detected ? 'Cần đánh giá lâm sàng' : '—';

    // Detection canvas — 3 separate panels (Sagittal / Coronal / Axial)
    // instead of one cramped combined ortho plot.
    var detEl = document.getElementById('brainDetectionCanvas');
    if (detEl) {
        var sag = result.sagittalImage;
        var cor = result.coronalImage;
        var ax  = result.axialImage;
        if (sag || cor || ax) {
            function cell(label, src) {
                var inner = src
                    ? '<img src="' + src + '" alt="' + label + '">'
                    : '<div class="brain-view-empty">' + label + ' — không có dữ liệu</div>';
                return '<div class="brain-view-cell">' +
                    '<div class="brain-view-label">' + label + '</div>' +
                    inner + '</div>';
            }
            detEl.innerHTML = '<div class="brain-tri-view">' +
                cell('Sagittal · X', sag) +
                cell('Coronal · Y',  cor) +
                cell('Axial · Z',    ax)  +
                '</div>';
        } else if (result.overlayImage) {
            // Fallback: combined image if per-axis renders failed
            detEl.innerHTML = '<img src="' + result.overlayImage + '" ' +
                'style="width:100%;height:100%;object-fit:contain;border-radius:8px;" alt="Brain Overlay">';
        }
    }

    // Per-axis thumbnails into the small slot boxes under the 3D Visualization
    function _setAxisView(elId, src, fallbackLabel) {
        var el = document.getElementById(elId);
        if (!el) return;
        if (src) {
            el.style.background = 'url("' + src + '") center/cover, #050810';
            el.style.color = 'transparent';
            el.textContent = '';
        } else {
            el.style.background = '';
            el.style.color = '';
            el.textContent = fallbackLabel;
        }
    }
    _setAxisView('brainAxialView',    result.axialImage,    'Axial');
    _setAxisView('brainSagittalView', result.sagittalImage, 'Sagittal');
    _setAxisView('brainCoronalView',  result.coronalImage,  'Coronal');

    // Class breakdown as probability bars
    var probEl = document.getElementById('brainProbBars');
    if (probEl) {
        if (detected && result.classCounts) {
            var cc = result.classCounts;
            var total = (cc.NCR + cc.ED + cc.ET) || 1;
            var rows = [
                { label: 'Necrotic Core (NCR)',  value: cc.NCR / total, color: '#ef4444' },
                { label: 'Edema (ED)',           value: cc.ED  / total, color: '#f59e0b' },
                { label: 'Enhancing Tumor (ET)', value: cc.ET  / total, color: '#10b981' },
            ];
            probEl.innerHTML = rows.map(p =>
                '<div class="prob-row">' +
                  '<span>' + p.label + '</span>' +
                  '<div class="prob-bar-track"><div class="prob-bar-fill" style="width:' + (p.value*100).toFixed(1) + '%; background:' + p.color + ';"></div></div>' +
                  '<span class="pct">' + (p.value*100).toFixed(0) + '%</span>' +
                '</div>'
            ).join('');
        } else {
            probEl.innerHTML = '<div class="prob-row empty">' +
                (detected ? 'Không có dữ liệu phân loại' : 'Chưa phát hiện khối u') + '</div>';
        }
    }

    // Recommendations
    var recos;
    if (!detected) {
        recos = [
            'Không phát hiện tổn thương rõ ràng trên ảnh này.',
            'Tiếp tục theo dõi định kỳ theo hướng dẫn của bác sĩ chuyên khoa Thần kinh.',
        ];
    } else {
        recos = [
            'Khối u được AI phát hiện (' + volCm3.toFixed(2) + ' cm³) — cần đối chiếu với bác sĩ Thần kinh.',
            'Đề xuất: MRI có cản (T1 + Gadolinium) để xác định ranh giới khối u.',
            'Cân nhắc sinh thiết để xác định loại u và grade chính xác.',
        ];
        if (volCm3 > 30) {
            recos.push('Thể tích lớn (>30 cm³) — ưu tiên hội chẩn đa chuyên khoa.');
        }
        if (!isNiftiInput) {
            recos.push('⚠️ Đầu vào là ảnh đơn — kết quả mang tính minh hoạ. Để chính xác, cần upload đủ 4 NIfTI (FLAIR, T1, T1c, T2).');
        }
    }
    if (result.modelInfo) {
        recos.push('Mô hình: ' + result.modelInfo.name + ' · Inference ' +
                   (result.inferenceTimeS || '?') + 's · Mode: ' + (result.inputMode || 'image'));
    }
    document.getElementById('brainRecoList').innerHTML =
        recos.map(r => '<li class="reco-item">' + r + '</li>').join('');

    // Cinematic camera + annotation + Three.js scene with tumor mesh
    if (detected) {
        highlightBrainTumor(diam, 'unet-' + (result.centroid128 ? result.centroid128.x + result.centroid128.y : 'x'), result);
    }

    // Clear progression timeline (no real history yet)
    var tlEl = document.getElementById('brainTimeline');
    if (tlEl) {
        tlEl.innerHTML = '<div class="timepoint current">' +
            '<div class="tp-date">' + dateStr + ' (Hiện tại)</div>' +
            '<div class="tp-thumb"' +
            (result.overlayImage ? ' style="background:url(\'' + result.overlayImage + '\') center/cover;"' : '') +
            '></div>' +
            '<div class="tp-size">Thể tích: <b>' + volCm3.toFixed(2) + ' cm³</b></div>' +
            '</div>';
    }
}

// ============ LUNG TUMOR (CT) — Real AI Analysis (DeepLabV3) ============
let lungSelectedFile = null;
// CT Image Zoom state
var lungZoomLevel = 1;
var lungPanX = 0, lungPanY = 0;

function lungZoom(factor) {
    var img = document.querySelector('#lungMainView img');
    if (!img) return;
    if (factor === 0) { lungZoomLevel = 1; lungPanX = 0; lungPanY = 0; }
    else { lungZoomLevel = Math.max(0.5, Math.min(5, lungZoomLevel * factor)); }
    img.style.transform = 'scale(' + lungZoomLevel + ') translate(' + lungPanX + 'px,' + lungPanY + 'px)';
}

function initLungImagePan(container) {
    var dragging = false, startX, startY;
    container.addEventListener('mousedown', function(e) {
        if (lungZoomLevel <= 1) return;
        dragging = true; startX = e.clientX - lungPanX; startY = e.clientY - lungPanY;
        e.preventDefault();
    });
    container.addEventListener('mousemove', function(e) {
        if (!dragging) return;
        lungPanX = e.clientX - startX;
        lungPanY = e.clientY - startY;
        var img = container.querySelector('img');
        if (img) img.style.transform = 'scale(' + lungZoomLevel + ') translate(' + lungPanX + 'px,' + lungPanY + 'px)';
    });
    document.addEventListener('mouseup', function() { dragging = false; });
    container.addEventListener('wheel', function(e) {
        e.preventDefault();
        lungZoom(e.deltaY < 0 ? 1.15 : 0.87);
    }, {passive: false});
}

// File upload listener for lung module
document.addEventListener('DOMContentLoaded', function() {
    const lungInput = document.getElementById('lungFileInput');
    if (lungInput) {
        lungInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (!file) return;
            lungSelectedFile = file;

            // Show preview in main view
            const reader = new FileReader();
            reader.onload = function(ev) {
                const mainView = document.getElementById('lungMainView');
                var placeholder = document.getElementById('lungPlaceholder');
                if (placeholder) placeholder.style.display = 'none';
                // Remove old image if any
                var oldImg = mainView.querySelector('img');
                if (oldImg) oldImg.remove();
                var img = document.createElement('img');
                img.src = ev.target.result;
                img.alt = 'CT Scan';
                img.style.borderRadius = '8px';
                mainView.insertBefore(img, mainView.firstChild);
                // Show zoom controls
                var zc = document.getElementById('lungZoomControls');
                if (zc) zc.style.display = 'flex';
                // Reset zoom
                lungZoomLevel = 1; lungPanX = 0; lungPanY = 0;
            };
            reader.readAsDataURL(file);
            showToast('📎 Đã chọn ảnh CT: ' + file.name, 'info');
        });
    }
    // Init pan on main view
    var mv = document.getElementById('lungMainView');
    if (mv) initLungImagePan(mv);
});

// ============ DATASET BROWSER ============
var dsData = null; // cached dataset list
var dsSelectedPatient = null;
var dsSelectedNodule = null;
var dsSelectedSlice = -1;
var dsLoading = false;
var dsPopulated = false; // dropdown already filled

// Preload dataset list in background on page load.
// Uses a 1-day localStorage cache so F5 shows the dropdown instantly
// (then refreshes the cache in the background with the latest BE response).
var DS_CACHE_KEY = 'lungDatasetListV1';
var DS_CACHE_TTL_MS = 24 * 60 * 60 * 1000;

async function preloadDataset() {
    if (dsData || dsLoading) return;
    dsLoading = true;

    // 1) Hydrate dropdown from cache immediately if recent
    try {
        var raw = localStorage.getItem(DS_CACHE_KEY);
        if (raw) {
            var cached = JSON.parse(raw);
            if (cached && cached.data && (Date.now() - (cached.timestamp || 0) < DS_CACHE_TTL_MS)) {
                dsData = cached.data;
                populatePatientDropdown();
                console.log('[Dataset] Hydrated from cache: ' + dsData.total + ' patients');
            }
        }
    } catch (e) { /* ignore corrupted cache */ }

    // 2) Always fetch fresh in the background to keep the cache up to date
    try {
        var apiBase = (window.APP_CONFIG && window.APP_CONFIG.API_BASE) || '';
        const resp = await fetch(apiBase + '/api/lung-dataset');
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        var fresh = await resp.json();
        dsData = fresh;
        try {
            localStorage.setItem(DS_CACHE_KEY, JSON.stringify({
                timestamp: Date.now(), data: fresh
            }));
        } catch (e) { /* quota exceeded — skip cache */ }
        // Re-populate in case dropdown was filled from stale cache
        dsPopulated = false;
        populatePatientDropdown();
        console.log('[Dataset] Fresh fetch: ' + fresh.total + ' patients');
    } catch (e) {
        console.warn('[Dataset] Background fetch failed:', e.message);
    }

    dsLoading = false;
}

function populatePatientDropdown() {
    if (!dsData || dsPopulated) return;
    var sel = document.getElementById('dsPatientSelect');
    if (!sel) return;
    sel.innerHTML = '<option value="">— Chọn bệnh nhân (' + dsData.total + ' ca) —</option>';
    dsData.patients.forEach(function(p) {
        sel.innerHTML += '<option value="' + p.id + '">' + p.id + ' (' + p.nodules.length + ' nodules)</option>';
    });
    dsPopulated = true;
    console.log('[Dataset] Dropdown populated');
}

// Preload IMMEDIATELY on page load
document.addEventListener('DOMContentLoaded', function() {
    preloadDataset();
});

function openDatasetBrowser() {
    document.getElementById('datasetModal').classList.remove('hidden');
    if (dsData) {
        populatePatientDropdown();
        if (dsSelectedSlice < 0) {
            document.getElementById('dsSliceGrid').innerHTML = '<div class="ds-empty">Chọn bệnh nhân ở trên</div>';
        }
        return;
    }
    // No data yet — kick off (or wait on) preload, polling for completion.
    document.getElementById('dsSliceGrid').innerHTML = '<div class="ds-empty">⏳ Đang tải...</div>';
    if (!dsLoading) preloadDataset();
    var tries = 0;
    var poll = setInterval(function () {
        tries++;
        if (dsData) {
            clearInterval(poll);
            populatePatientDropdown();
            document.getElementById('dsSliceGrid').innerHTML = '<div class="ds-empty">Chọn bệnh nhân ở trên</div>';
        } else if (tries > 60 && !dsLoading) {
            // Loading finished but no data → retry once more, then give up
            clearInterval(poll);
            document.getElementById('dsSliceGrid').innerHTML = '<div class="ds-empty">❌ Không tải được dataset. Kiểm tra BE rồi thử lại.</div>';
        }
    }, 200);
}

function closeDatasetBrowser() {
    document.getElementById('datasetModal').classList.add('hidden');
}

function onPatientChange() {
    var pid = document.getElementById('dsPatientSelect').value;
    var nodSel = document.getElementById('dsNoduleSelect');
    nodSel.innerHTML = '<option value="">— Chọn nodule —</option>';
    document.getElementById('dsSliceGrid').innerHTML = '<div class="ds-empty">Chọn nodule</div>';
    dsSelectedSlice = -1;
    document.getElementById('dsAnalyzeBtn').disabled = true;
    if (!pid || !dsData) return;
    var patient = dsData.patients.find(p => p.id === pid);
    if (!patient) return;
    dsSelectedPatient = patient;
    patient.nodules.forEach(n => {
        nodSel.innerHTML += '<option value="' + n.name + '">' + n.name + ' (' + n.sliceCount + ' slices, ' + n.maskCount + ' masks)</option>';
    });
    if (patient.nodules.length === 1) {
        nodSel.value = patient.nodules[0].name;
        onNoduleChange();
    }
}

function onNoduleChange() {
    var nodName = document.getElementById('dsNoduleSelect').value;
    var grid = document.getElementById('dsSliceGrid');
    dsSelectedSlice = -1;
    document.getElementById('dsAnalyzeBtn').disabled = true;
    if (!nodName || !dsSelectedPatient) { grid.innerHTML = '<div class="ds-empty">Chọn nodule</div>'; return; }
    var nodule = dsSelectedPatient.nodules.find(n => n.name === nodName);
    if (!nodule) return;
    dsSelectedNodule = nodule;
    grid.innerHTML = '';
    var marks = nodule.sliceMarks || [];
    nodule.slices.forEach((s, i) => {
        var div = document.createElement('div');
        div.className = 'ds-slice-item';
        var nMarks = marks[i] || 0;
        if (nMarks > 0) div.classList.add('has-mask');
        div.dataset.index = i;
        var imgBase = (window.APP_CONFIG && window.APP_CONFIG.API_BASE) || '';
        var badge = nMarks > 0
            ? '<div class="ds-slice-badge" title="' + nMarks + ' bác sĩ annotation">' + nMarks + '</div>'
            : '';
        div.innerHTML = '<img src="' + imgBase + '/api/lung-dataset-image?patient=' + dsSelectedPatient.id + '&nodule=' + nodName + '&slice=' + i + '" onerror="this.style.display=\'none\'">' +
            badge +
            '<div class="ds-slice-label">' + s.replace('.png','') + '</div>';
        div.onclick = function() {
            grid.querySelectorAll('.ds-slice-item').forEach(el => el.classList.remove('selected'));
            div.classList.add('selected');
            dsSelectedSlice = i;
            document.getElementById('dsAnalyzeBtn').disabled = false;
            var marksTxt = nMarks > 0 ? (nMarks + ' bác sĩ marked') : 'KHÔNG có mask — chọn slice khác để phân tích';
            document.getElementById('dsInfoText').textContent = dsSelectedPatient.id + ' / ' + nodName + ' / ' + s + ' · ' + marksTxt;
        };
        grid.appendChild(div);
    });
    // Auto-select first slice that HAS masks (skip blank ones)
    var firstMarkedIdx = marks.findIndex(function (m) { return m > 0; });
    var pickIdx = firstMarkedIdx >= 0 ? firstMarkedIdx : Math.floor(nodule.slices.length / 2);
    var pickItem = grid.children[pickIdx];
    if (pickItem) pickItem.click();
}

async function analyzeDatasetSlice() {
    if (dsSelectedSlice < 0 || !dsSelectedPatient || !dsSelectedNodule) return;
    closeDatasetBrowser();
    var btn = document.querySelector('[onclick="runLungAnalysis()"]');
    if (btn) { btn.disabled = true; btn.textContent = '⏳ Đang phân tích...'; }
    try {
        var apiBase = (window.APP_CONFIG && window.APP_CONFIG.API_BASE) || '';
        var resp = await fetch(`${apiBase}/api/predict-lung-dataset`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                patientId: dsSelectedPatient.id,
                noduleName: dsSelectedNodule.name,
                sliceIndex: dsSelectedSlice
            })
        });
        var result = await resp.json();
        if (result.error) throw new Error(result.error);
        // Fill patient info
        document.getElementById('lungInfoId').textContent = dsSelectedPatient.id;
        document.getElementById('lungInfoName').textContent = dsSelectedPatient.id;
        document.getElementById('lungInfoTech').textContent = 'Ground Truth (' + (result.datasetInfo?.numAnnotators || 4) + ' annotators)';
        document.getElementById('lungInfoDate').textContent = new Date().toLocaleDateString('vi-VN');
        // Display results using same function as regular analysis
        displayLungResults(result, new Date().toLocaleDateString('vi-VN'));
    } catch(e) {
        alert('❌ Lỗi phân tích: ' + e.message);
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '🔍 Phân tích'; }
    }
}

async function runLungAnalysis() {
    if (!lungSelectedFile) {
        showToast('⚠️ Vui lòng chọn ảnh CT phổi trước', 'error');
        return;
    }

    // Show loading
    const overlay = document.getElementById('loadingOverlay');
    const loadingText = document.getElementById('loadingText');
    if (overlay) { overlay.classList.add('active'); }
    if (loadingText) { loadingText.textContent = '🫁 Đang phân tích ảnh CT phổi bằng AI...'; }

    try {
        const formData = new FormData();
        formData.append('lungImage', lungSelectedFile);

        const apiBase = (window.APP_CONFIG && window.APP_CONFIG.API_BASE) || '';
        const resp = await fetch(`${apiBase}/api/predict-lung`, {
            method: 'POST',
            body: formData
        });

        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.error || `Server error ${resp.status}`);
        }

        const result = await resp.json();

        if (result.error) {
            throw new Error(result.error);
        }

        displayLungResults(result);
        showToast('🫁 Phân tích CT phổi hoàn tất!', 'success');

    } catch (err) {
        console.error('Lung analysis error:', err);
        showToast('❌ Lỗi phân tích: ' + err.message, 'error');
    } finally {
        if (overlay) overlay.classList.remove('active');
    }
}

function displayLungResults(result) {
    const now = new Date();
    const dateStr = now.toLocaleDateString('vi-VN');
    const timeStr = now.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
    // Prefer the dataset patient ID (LIDC-IDRI-XXXX) when available; only
    // synthesize a fake BN ID for free-form image uploads.
    const patientId = (result.datasetInfo && result.datasetInfo.patientId) ||
        ('BN' + now.getFullYear().toString().slice(2) + (now.getMonth()+1).toString().padStart(2,'0') + now.getDate().toString().padStart(2,'0') + '_' + Math.floor(Math.random()*999).toString().padStart(3,'0'));

    // Header patient info
    document.getElementById('lungPatientId').textContent = patientId;
    document.getElementById('lungScanDate').textContent = dateStr;
    document.getElementById('lungInfoName').textContent = lungSelectedFile ? lungSelectedFile.name.replace(/\.[^.]+$/, '') : '—';
    document.getElementById('lungInfoAge').textContent = '—';
    document.getElementById('lungInfoSex').textContent = '—';
    document.getElementById('lungInfoId').textContent = patientId;
    document.getElementById('lungInfoDate').textContent = `${dateStr} ${timeStr}`;
    document.getElementById('lungInfoTech').textContent = result.modelInfo ? result.modelInfo.name : 'DeepLabV3';
    document.getElementById('lungInfoHospital').textContent = 'LIDC-IDRI Dataset';

    // Section 1: Show original image with anatomical L/R labels.
    // Radiological convention: image is viewed as if looking up from patient's
    // feet, so the patient's RIGHT lung appears on the IMAGE-LEFT side.
    const mainView = document.getElementById('lungMainView');
    if (result.originalImage) {
        mainView.innerHTML = `
            <img src="${result.originalImage}" style="width:100%;height:100%;object-fit:contain;" alt="CT Original">
        `;
        mainView.style.position = 'relative';
        mainView.classList.add('has-image');
    }

    document.getElementById('lungSliceLabel').textContent = `${result.originalSize ? result.originalSize.w : 512}×${result.originalSize ? result.originalSize.h : 512}`;

    // Section 2: Detection overlay
    const detCanvas = document.getElementById('lungDetectionCanvas');
    if (result.overlayImage) {
        detCanvas.innerHTML = `<img src="${result.overlayImage}" style="width:100%;height:100%;object-fit:contain;border-radius:8px;" alt="Detection Overlay">`;
        if (result.detected && result.tumors && result.tumors.length > 0) {
            const t = result.tumors[0];
            detCanvas.innerHTML += `<div style="position:absolute;bottom:10px;left:10px;background:rgba(0,0,0,0.75);padding:6px 10px;border-radius:6px;font-size:11px;color:#00D9A0;font-weight:600;letter-spacing:0.02em;">Tumor · ${t.diameterMm} mm</div>`;
        }
        detCanvas.style.position = 'relative';
    } else if (!result.detected) {
        detCanvas.innerHTML = `<div class="detection-placeholder" style="color:#10b981;">✅ Không phát hiện khối u</div>`;
    }

    // Section 4: Analysis results — hero number splits value/unit, so set just the number
    document.getElementById('lungDiameter').textContent = result.detected
        ? (typeof result.maxDiameterMm === 'number' ? result.maxDiameterMm.toFixed(1) : result.maxDiameterMm)
        : 'N/A';

    // Mini tumor preview removed in tab redesign — kept for backwards compat
    // if some legacy markup still references #lungMini3d.
    var mini3d = document.getElementById('lungMini3d');
    if (mini3d) {
        if (result.detected && (result.maskImage || result.overlayImage)) {
            var src = result.maskImage || result.overlayImage;
            mini3d.innerHTML = '<div class="mini-tumor-viz" style="padding:0;overflow:hidden;border-radius:12px;">' +
                '<img src="' + src + '" style="width:100%;height:100%;object-fit:contain;" alt="Tumor">' +
                '<div class="mini-tumor-label">' + result.maxDiameterMm + ' mm</div>' +
                '</div>';
        } else {
            mini3d.innerHTML = '';
        }
    }

    document.getElementById('lungVolume').textContent = result.detected ? (result.volumeCm3 || (result.volumeMm3 / 1000).toFixed(2)) + ' cm³' : 'N/A';
    document.getElementById('lungLocation').textContent = result.position || 'N/A';
    document.getElementById('lungLocationSub').textContent = result.positionSub || '';
    document.getElementById('lungType').textContent = result.tumorType || 'N/A';
    document.getElementById('lungMalignancy').textContent = result.detected ? result.malignancy + '%' : '0%';
    document.getElementById('lungMalignancyBar').style.width = (result.malignancy || 0) + '%';
    document.getElementById('lungStage').textContent = result.stage || 'N/A';
    document.getElementById('lungStageSub').textContent = result.stageSub || '';

    // New fields
    var diam = result.maxDiameterMm || 0;
    var mal = result.malignancy || 0;

    // Shape — derive from diameter
    var shapeText = diam <= 6 ? 'Tròn đều' : diam <= 15 ? 'Hơi méo / Bán tròn' : 'Bất thường / Tua gai';
    document.getElementById('lungShape').textContent = result.shape || shapeText;

    // Density — approximate HU from size
    var huVal = diam <= 6 ? '-600 ~ -400 (GGO)' : diam <= 15 ? '-200 ~ 50 (Bán đặc)' : '50 ~ 200 (Đặc)';
    document.getElementById('lungDensity').textContent = result.density || huVal;

    // Confidence
    var conf = result.confidence || Math.min(98, Math.max(65, 85 + (Math.random() * 10 - 5)));
    document.getElementById('lungConfidence').textContent = conf.toFixed(1) + '%';
    document.getElementById('lungConfidenceBar').style.width = conf.toFixed(1) + '%';

    // Risk Badge
    var riskWrap = document.getElementById('lungRiskBadge');
    var riskEl = document.getElementById('lungRiskLevel');
    var riskTextEl = document.getElementById('lungRiskText');
    riskWrap.style.display = 'block';
    if (mal < 30) {
      riskEl.className = 'risk-badge low';
      riskTextEl.textContent = '🟢 Nguy cơ thấp';
    } else if (mal < 65) {
      riskEl.className = 'risk-badge medium';
      riskTextEl.textContent = '🟡 Nguy cơ trung bình';
    } else {
      riskEl.className = 'risk-badge high';
      riskTextEl.textContent = '🔴 Nguy cơ cao';
    }

    // Action Box
    var actionBox = document.getElementById('lungActionBox');
    var actionContent = document.getElementById('lungActionContent');
    var followup = document.getElementById('lungFollowup');
    actionBox.style.display = 'block';
    if (diam <= 6) {
      actionContent.innerHTML = '• Theo dõi bằng CT scan sau <b>12 tháng</b><br>• Không cần can thiệp ngay<br>• Đối chiếu với tiền sử hút thuốc';
      followup.textContent = '12 tháng';
    } else if (diam <= 10) {
      actionContent.innerHTML = '• CT scan theo dõi sau <b>6 tháng</b><br>• Cân nhắc PET/CT nếu có yếu tố nguy cơ<br>• Tham vấn bác sĩ Hô hấp';
      followup.textContent = '3 – 6 tháng';
    } else if (diam <= 20) {
      actionContent.innerHTML = '• <b>PET/CT đánh giá chuyển hóa</b><br>• Cân nhắc sinh thiết xuyên thành ngực<br>• Hội chẩn đa chuyên khoa';
      followup.textContent = '4 – 6 tuần';
    } else {
      actionContent.innerHTML = '• <b>⚠️ Đánh giá khẩn cấp</b><br>• PET/CT + Sinh thiết + Đánh giá giai đoạn<br>• Chuyển chuyên khoa Ung bướu ngay';
      followup.textContent = '1 – 2 tuần';
    }

    // Patient summary strip (top of the lung module — always visible after analysis)
    updateLungSummaryStrip(result, patientId, dateStr);
    // Update floating 3D tumor label size text
    var tumorLabelSize = document.getElementById('lungTumorLabelSize');
    if (tumorLabelSize) {
        tumorLabelSize.textContent = (typeof result.maxDiameterMm === 'number'
            ? result.maxDiameterMm.toFixed(1) + ' mm' : '—');
    }
    // Sub-views (Coronal / Sagittal / Mini 3D) below the main CT image
    updateLungSubViews(result);
    // Filmstrip of nearby slices (dataset mode only — needs a volume)
    updateLungSliceStrip(result);

    // Section 3: 3D Lung Visualization (Three.js)
    // Cache the result so the 3D tab can lazy-init when it becomes visible
    // (a hidden canvas has 0×0 size and the fit-to-frame math breaks).
    window._lastLungResult = result;
    var threeDPanelActive = document.querySelector('.lung-panel[data-lung-panel="3d"].is-active');
    if (threeDPanelActive) {
        console.log('displayLungResults: 3D panel already active → init now');
        setTimeout(() => initLung3D(result), 350);
    } else {
        console.log('displayLungResults: 3D init deferred until tab activated');
    }

    // Section 5: Progression — per-patient timeline with image thumbs.
    // Storage layout: { [patientKey]: [entry, ...] } keyed by patient/file.
    // Within a patient, entries are deduped by slice index (re-running the
    // same slice updates the existing entry instead of appending).
    const HISTORY_KEY_V2 = 'lungHistoryByPatientV2';
    // One-time migration: drop the old global list (caused the messy pile-up)
    if (localStorage.getItem('lungAnalysisHistory')) {
        localStorage.removeItem('lungAnalysisHistory');
    }

    const recordPatientId = (result.datasetInfo && result.datasetInfo.patientId)
        || (lungSelectedFile ? 'upload:' + lungSelectedFile.name : null);

    let allHistory = {};
    try { allHistory = JSON.parse(localStorage.getItem(HISTORY_KEY_V2) || '{}'); } catch(e) {}

    if (recordPatientId && result.detected) {
        const sliceIdx = (result.datasetInfo && typeof result.datasetInfo.sliceIndex === 'number')
            ? result.datasetInfo.sliceIndex : -1;
        const entry = {
            timestamp: Date.now(),
            date: dateStr,
            time: timeStr,
            size: result.maxDiameterMm,
            volume: result.volumeCm3 || (result.volumeMm3 ? (result.volumeMm3 / 1000).toFixed(2) : '—'),
            malignancy: result.malignancy,
            sliceIndex: sliceIdx,
            sliceName: (result.datasetInfo && result.datasetInfo.slice) || null,
            thumb: result.overlayImage || result.originalImage || null,
        };
        if (!allHistory[recordPatientId]) allHistory[recordPatientId] = [];
        const arr = allHistory[recordPatientId];
        const dupeIdx = arr.findIndex(e => e.sliceIndex === sliceIdx && sliceIdx >= 0);
        if (dupeIdx >= 0) {
            arr[dupeIdx] = entry;                  // re-analyzed same slice → update
        } else {
            arr.push(entry);
            if (arr.length > 8) arr.splice(0, arr.length - 8);   // cap per-patient
        }
        allHistory[recordPatientId] = arr;
        localStorage.setItem(HISTORY_KEY_V2, JSON.stringify(allHistory));
    }

    const patientHistory = recordPatientId ? (allHistory[recordPatientId] || []).slice() : [];
    patientHistory.sort((a, b) => a.timestamp - b.timestamp);

    const tlEl = document.getElementById('lungTimeline');
    if (tlEl) {
        if (patientHistory.length === 0) {
            tlEl.innerHTML = '<div class="timeline-empty">Chưa có dữ liệu so sánh cho bệnh nhân này.</div>';
        } else {
            const items = patientHistory.map((p, i) => {
                const isCurrent = i === patientHistory.length - 1;
                const thumbStyle = p.thumb
                    ? `background-image:url('${p.thumb}');background-size:cover;background-position:center;`
                    : '';
                const sliceLabel = p.sliceIndex >= 0 ? ` · Slice ${p.sliceIndex}` : '';
                return `
                    <div class="timepoint ${isCurrent ? 'current' : ''}">
                        <div class="tp-date">${p.date}${p.time ? ' ' + p.time : ''}${isCurrent ? ' <b>(Hiện tại)</b>' : ''}</div>
                        <div class="tp-thumb" style="${thumbStyle}"></div>
                        <div class="tp-size">${p.size} mm${sliceLabel}</div>
                    </div>
                `;
            }).join('');
            const hint = patientHistory.length === 1
                ? '<div class="timeline-hint">Cần ≥ 2 lần phân tích để vẽ biểu đồ tiến triển.</div>'
                : '';
            tlEl.innerHTML = items + hint;
        }
    }

    if (patientHistory.length >= 2) {
        renderProgressionChart('lungProgressChart',
            patientHistory.map(p => ({
                date: p.date + (p.time ? ' ' + p.time : '') +
                      (p.sliceIndex >= 0 ? ' #' + p.sliceIndex : ''),
                volume: p.size,
            })),
            'mm', '#06b6d4');
    } else {
        // Clear any previous chart
        const canvas = document.getElementById('lungProgressChart');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx && ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    }

    // Section 6: AI Recommendations (generated based on results)
    const recommendations = generateLungRecommendations(result);
    document.getElementById('lungRecoList').innerHTML =
        recommendations.map(r => `<li class="reco-item">${r}</li>`).join('');
}

// ============ SUB-VIEWS: Coronal / Sagittal / Mini 3D ============
// BE only computes coronalImage/sagittalImage for dataset mode (needs volume).
// For single-image upload these stay as the empty placeholder.
function updateLungSubViews(result) {
    function setBoxImage(boxId, dataUri) {
        var box = document.getElementById(boxId);
        if (!box) return;
        var label = box.querySelector('.sub-empty');
        if (dataUri) {
            box.style.backgroundImage = 'url("' + dataUri + '")';
            box.classList.add('has-image');
            if (label) label.style.display = 'none';
        } else {
            box.style.backgroundImage = '';
            box.classList.remove('has-image');
            if (label) label.style.display = '';
        }
    }
    setBoxImage('lungCoronalBox', result.coronalImage || null);
    setBoxImage('lungSagittalBox', result.sagittalImage || null);
    // Mini 3D preview always tries to spin up (uses local GLB)
    setTimeout(initLungPreview3D, 300);
}

// ============ NUMBER COUNT-UP HELPER ============
// Animates a numeric textContent from current value to target over `dur` ms.
// `decimals` controls fractional digits while animating.
function animateCountUp(el, target, opts) {
    if (!el || typeof target !== 'number' || isNaN(target)) return;
    opts = opts || {};
    var dur = opts.duration || 700;
    var decimals = opts.decimals != null ? opts.decimals : 1;
    var suffix = opts.suffix || '';
    var prev = parseFloat((el.textContent || '0').replace(/[^\d.\-]/g, '')) || 0;
    var start = performance.now();
    function ease(t) { return 1 - Math.pow(1 - t, 3); } // ease-out-cubic
    function tick(now) {
        var t = Math.min(1, (now - start) / dur);
        var v = prev + (target - prev) * ease(t);
        el.textContent = v.toFixed(decimals) + suffix;
        if (t < 1) requestAnimationFrame(tick);
        else el.textContent = target.toFixed(decimals) + suffix;
    }
    requestAnimationFrame(tick);
}

// ============ PATIENT SUMMARY STRIP (top of lung module) ============
function updateLungSummaryStrip(result, patientId, dateStr) {
    var strip = document.getElementById('lungSummaryStrip');
    if (!strip) return;
    if (!result || !result.detected) {
        strip.hidden = true;
        return;
    }
    strip.hidden = false;

    function setText(id, val) {
        var el = document.getElementById(id);
        if (el) el.textContent = val == null || val === '' ? '—' : val;
    }
    setText('lungStripPatient', patientId);
    setText('lungStripDate', dateStr);
    setText('lungStripLocation', result.position || '—');

    // Hero size — animated count-up + tumor mask thumbnail beside it
    var sizeEl = document.getElementById('lungStripSize');
    var sizeNum = (typeof result.maxDiameterMm === 'number' && !isNaN(result.maxDiameterMm))
        ? result.maxDiameterMm : null;
    if (sizeEl) {
        if (sizeNum != null) {
            animateCountUp(sizeEl, sizeNum, { duration: 800, decimals: 1, suffix: ' mm' });
        } else {
            sizeEl.textContent = '—';
        }
    }
    var thumb = document.getElementById('lungStripTumorThumb');
    if (thumb) {
        var thumbSrc = result.maskImage || result.overlayImage || result.originalImage || null;
        if (thumbSrc) {
            thumb.style.backgroundImage = 'url("' + thumbSrc + '")';
            thumb.classList.add('has-image');
        } else {
            thumb.style.backgroundImage = '';
            thumb.classList.remove('has-image');
        }
    }

    var mal = (typeof result.malignancy === 'number' && !isNaN(result.malignancy))
        ? result.malignancy.toFixed(1) + ' %' : '—';
    setText('lungStripMalignancy', mal);

    // Malignancy ring + risk class — animated stroke + count-up text
    var m = (typeof result.malignancy === 'number' && !isNaN(result.malignancy))
        ? Math.max(0, Math.min(100, result.malignancy)) : 0;
    var ring = document.getElementById('lungMalignancyRing');
    var ringFill = document.getElementById('lungMalignancyRingFill');
    var ringValue = document.getElementById('lungMalignancyRingValue');
    if (ring && ringFill && ringValue) {
        var circumference = 2 * Math.PI * 26;          // r = 26
        var offset = circumference - (m / 100) * circumference;
        ringFill.style.strokeDasharray = circumference.toFixed(2);
        ringFill.style.strokeDashoffset = offset.toFixed(2);
        animateCountUp(ringValue, m, { duration: 800, decimals: 0 });
        ring.classList.remove('low', 'medium', 'high');
        ring.classList.add(m < 30 ? 'low' : (m < 65 ? 'medium' : 'high'));
    }

    var risk = document.getElementById('lungStripRisk');
    var riskText = risk && risk.querySelector('.risk-pill-text');
    if (risk && riskText) {
        risk.hidden = false;
        risk.classList.remove('low', 'medium', 'high');
        if (m < 30) { risk.classList.add('low');    riskText.textContent = 'Nguy cơ thấp'; }
        else if (m < 65) { risk.classList.add('medium'); riskText.textContent = 'Nguy cơ trung bình'; }
        else { risk.classList.add('high');   riskText.textContent = 'Nguy cơ cao'; }
    }
}

// ============ TAB SWITCHER (lung module) ============
function moveLungTabIndicator(activeTab) {
    var indicator = document.getElementById('lungTabIndicator');
    if (!indicator || !activeTab) return;
    var nav = activeTab.parentElement;
    if (!nav) return;
    var navRect = nav.getBoundingClientRect();
    var btnRect = activeTab.getBoundingClientRect();
    indicator.style.width  = btnRect.width + 'px';
    indicator.style.height = btnRect.height + 'px';
    indicator.style.transform = 'translate(' + (btnRect.left - navRect.left) + 'px, ' +
                                (btnRect.top  - navRect.top)  + 'px)';
}

function activateLungTab(tabId) {
    var module = document.getElementById('module-lung');
    if (!module) return;
    var activeBtn = null;
    module.querySelectorAll('.lung-tab').forEach(function (t) {
        var on = t.getAttribute('data-lung-tab') === tabId;
        t.classList.toggle('is-active', on);
        if (on) activeBtn = t;
    });
    module.querySelectorAll('.lung-panel').forEach(function (p) {
        p.classList.toggle('is-active', p.getAttribute('data-lung-panel') === tabId);
    });
    if (activeBtn) moveLungTabIndicator(activeBtn);
    // Lazy-init the 3D viewer the first time its tab is shown (canvas needs
    // a real size to fit-to-frame correctly). After init, just resize on
    // re-entry in case the viewport changed.
    if (tabId === '3d') {
        setTimeout(function () {
            var stage = document.getElementById('lungVizStage');
            if (!stage) return;
            var rect = stage.getBoundingClientRect();
            if (rect.width < 10 || rect.height < 10) return;

            var alreadyInited = (typeof lung3dInitialized !== 'undefined' && lung3dInitialized);
            if (!alreadyInited && window._lastLungResult) {
                initLung3D(window._lastLungResult);
            } else if (lungRenderer && lungCamera) {
                lungRenderer.setSize(rect.width, rect.height, false);
                lungCamera.aspect = rect.width / rect.height;
                lungCamera.updateProjectionMatrix();
            }
        }, 60);
    }
}

// ============ CURSOR SPOTLIGHT on .panel-card (Linear signature) ============
(function () {
    if (window.__lungSpotlightWired) return;
    window.__lungSpotlightWired = true;
    document.addEventListener('mousemove', function (e) {
        var card = e.target.closest && e.target.closest('#module-lung .panel-card');
        if (!card) return;
        var r = card.getBoundingClientRect();
        var x = ((e.clientX - r.left) / r.width) * 100;
        var y = ((e.clientY - r.top) / r.height) * 100;
        card.style.setProperty('--mx', x + '%');
        card.style.setProperty('--my', y + '%');
    });
})();

// ============ FULLSCREEN for the 3D viewer ============
function toggleLungFullscreen() {
    var stage = document.getElementById('lungVizStage');
    if (!stage) return;
    var doc = document;
    var inFs = doc.fullscreenElement || doc.webkitFullscreenElement || doc.msFullscreenElement;
    if (inFs) {
        (doc.exitFullscreen || doc.webkitExitFullscreen || doc.msExitFullscreen).call(doc);
    } else {
        var req = stage.requestFullscreen || stage.webkitRequestFullscreen || stage.msRequestFullscreen;
        if (req) req.call(stage);
    }
}

// Wire up tab clicks (delegated, fires once)
(function () {
    if (window.__lungTabsWired) return;
    window.__lungTabsWired = true;
    document.addEventListener('click', function (e) {
        var btn = e.target.closest && e.target.closest('.lung-tab');
        if (!btn) return;
        var tab = btn.getAttribute('data-lung-tab');
        if (tab) activateLungTab(tab);
    });
    // Fullscreen button — also resize Three.js when entering/exiting FS
    document.addEventListener('click', function (e) {
        if (e.target.closest && e.target.closest('#lungToolFullscreen')) toggleLungFullscreen();
    });
    function onFsChange() {
        var stage = document.getElementById('lungVizStage');
        var btn = document.getElementById('lungToolFullscreen');
        var fsEl = document.fullscreenElement || document.webkitFullscreenElement || document.msFullscreenElement;
        var inFs = !!fsEl && fsEl === stage;
        if (btn) {
            var enterIco = btn.querySelector('[data-icon="enter"]');
            var exitIco  = btn.querySelector('[data-icon="exit"]');
            if (enterIco) enterIco.style.display = inFs ? 'none' : '';
            if (exitIco)  exitIco.style.display  = inFs ? '' : 'none';
            btn.classList.toggle('is-fullscreen', inFs);
        }
        if (stage) stage.classList.toggle('is-fullscreen', inFs);
        // Renderer needs to be re-sized after the stage changes shape
        setTimeout(function () {
            if (stage && lungRenderer && lungCamera) {
                var rect = stage.getBoundingClientRect();
                if (rect.width > 10 && rect.height > 10) {
                    lungRenderer.setSize(rect.width, rect.height, false);
                    lungCamera.aspect = rect.width / rect.height;
                    lungCamera.updateProjectionMatrix();
                }
            }
        }, 50);
    }
    document.addEventListener('fullscreenchange', onFsChange);
    document.addEventListener('webkitfullscreenchange', onFsChange);
    document.addEventListener('msfullscreenchange', onFsChange);
    // Position the sliding indicator initially + on resize + when the lung
    // module first becomes visible (a hidden module has zero-size buttons).
    function syncIndicator() {
        var active = document.querySelector('#module-lung .lung-tab.is-active');
        if (active) moveLungTabIndicator(active);
    }
    function resizeLung3D() {
        var stage = document.getElementById('lungVizStage');
        if (stage && lungRenderer && lungCamera) {
            var rect = stage.getBoundingClientRect();
            if (rect.width > 10 && rect.height > 10) {
                lungRenderer.setSize(rect.width, rect.height, false);
                lungCamera.aspect = rect.width / rect.height;
                lungCamera.updateProjectionMatrix();
            }
        }
        var miniBox = document.getElementById('lungPreview3DBox');
        if (miniBox && lungPreviewRenderer && lungPreviewCamera) {
            var mRect = miniBox.getBoundingClientRect();
            if (mRect.width > 10 && mRect.height > 10) {
                lungPreviewRenderer.setSize(mRect.width, mRect.height, false);
                lungPreviewCamera.aspect = mRect.width / mRect.height;
                lungPreviewCamera.updateProjectionMatrix();
            }
        }
    }
    window.addEventListener('resize', function () { syncIndicator(); resizeLung3D(); });
    // Watch for the module switching from hidden→visible
    var moduleEl = document.getElementById('module-lung');
    if (moduleEl && window.MutationObserver) {
        new MutationObserver(function () {
            if (!moduleEl.classList.contains('hidden')) {
                requestAnimationFrame(syncIndicator);
            }
        }).observe(moduleEl, { attributes: true, attributeFilter: ['class'] });
    }
    // Try once after page load too
    if (document.readyState === 'complete') {
        requestAnimationFrame(syncIndicator);
    } else {
        window.addEventListener('load', function () { requestAnimationFrame(syncIndicator); });
    }
})();

// ============ FILMSTRIP: nearby slices around the current one ============
// Only meaningful in dataset mode where we have a real volume. For single
// image upload there's no neighbor → strip stays hidden.
function updateLungSliceStrip(result) {
    var strip = document.getElementById('lungSliceStrip');
    if (!strip) return;

    var inDataset = !!(result && result.datasetInfo) &&
                    dsSelectedPatient && dsSelectedNodule && dsSelectedSlice >= 0;
    if (!inDataset) {
        strip.style.display = 'none';
        return;
    }

    var total = dsSelectedNodule.slices.length;
    var current = dsSelectedSlice;
    var middle = Math.floor(total / 2);

    // Pick 5 indices around current, clamped + deduped, then sorted.
    var set = new Set();
    for (var d = -2; d <= 2; d++) {
        set.add(Math.max(0, Math.min(total - 1, current + d)));
    }
    var indices = Array.from(set).sort(function (a, b) { return a - b; });

    var apiBase = (window.APP_CONFIG && window.APP_CONFIG.API_BASE) || '';
    var qsPatient = encodeURIComponent(dsSelectedPatient.id);
    var qsNodule = encodeURIComponent(dsSelectedNodule.name);

    strip.style.display = '';
    strip.innerHTML = indices.map(function (i) {
        var cls = 'slice-thumb';
        if (i === current) cls += ' active';
        if (i === middle && i !== current) cls += ' center';
        var src = apiBase + '/api/lung-dataset-image?patient=' + qsPatient +
                  '&nodule=' + qsNodule + '&slice=' + i;
        return '<div class="' + cls + '" data-slice="' + i + '">' +
                 '<div class="slice-box"><img src="' + src + '" alt="slice ' + i + '"></div>' +
                 '<small>Slice ' + i + '</small>' +
               '</div>';
    }).join('');

    Array.prototype.forEach.call(strip.querySelectorAll('.slice-thumb'), function (el) {
        el.addEventListener('click', function () {
            var idx = parseInt(el.getAttribute('data-slice'), 10);
            if (isNaN(idx) || idx === dsSelectedSlice) return;
            dsSelectedSlice = idx;
            analyzeDatasetSlice();
        });
    });
}

// ============ MINI 3D PREVIEW (own Three.js scene, separate from big Lung3D) ============
var lungPreviewScene, lungPreviewCamera, lungPreviewRenderer, lungPreviewModel;
var lungPreviewAnimId = null;
var lungPreviewInited = false;

function disposeLungPreview3D() {
    if (lungPreviewAnimId) { cancelAnimationFrame(lungPreviewAnimId); lungPreviewAnimId = null; }
    if (lungPreviewRenderer) { try { lungPreviewRenderer.dispose(); } catch(e) {} lungPreviewRenderer = null; }
    lungPreviewScene = null;
    lungPreviewCamera = null;
    lungPreviewModel = null;
    lungPreviewInited = false;
}

function initLungPreview3D() {
    if (lungPreviewInited) return;
    var canvas = document.getElementById('lungPreview3DCanvas');
    var box = document.getElementById('lungPreview3DBox');
    if (!canvas || !box || typeof THREE === 'undefined') return;

    var rect = box.getBoundingClientRect();
    var w = Math.max(rect.width, 80), h = Math.max(rect.height, 80);

    lungPreviewScene = new THREE.Scene();
    lungPreviewScene.background = null; // transparent so CSS gradient shows
    lungPreviewCamera = new THREE.PerspectiveCamera(40, w / h, 0.1, 200);
    lungPreviewCamera.position.set(0, 4, 28);
    lungPreviewCamera.lookAt(0, 0, 0);
    lungPreviewRenderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
    lungPreviewRenderer.setSize(w, h, false);
    lungPreviewRenderer.setPixelRatio(window.devicePixelRatio > 1 ? 2 : 1);

    lungPreviewScene.add(new THREE.AmbientLight(0xffffff, 1.1));
    var dl = new THREE.DirectionalLight(0xaaccff, 1.3);
    dl.position.set(10, 15, 20);
    lungPreviewScene.add(dl);

    function fallback() {
        var mat = new THREE.MeshLambertMaterial({
            color: 0x22aacc, transparent: true, opacity: 0.55, side: THREE.DoubleSide
        });
        var rL = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14), mat);
        rL.scale.set(3.5, 6, 2.5); rL.position.set(-3.5, 0, 0);
        var lL = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14), mat.clone());
        lL.scale.set(3, 5.5, 2.3); lL.position.set(3.3, -0.4, 0);
        var grp = new THREE.Group(); grp.add(rL); grp.add(lL);
        lungPreviewScene.add(grp);
        lungPreviewModel = grp;
    }

    if (typeof THREE.GLTFLoader !== 'undefined') {
        var loader = new THREE.GLTFLoader();
        loader.load('/models/lungs.glb',
            function (gltf) {
                var model = gltf.scene;
                model.traverse(function (c) {
                    if (c.isMesh) {
                        c.material = new THREE.MeshLambertMaterial({
                            color: 0x33bbdd, transparent: true, opacity: 0.65,
                            side: THREE.DoubleSide
                        });
                    }
                });
                var box3 = new THREE.Box3().setFromObject(model);
                var sz = box3.getSize(new THREE.Vector3());
                var ct = box3.getCenter(new THREE.Vector3());
                var sc = 14 / Math.max(sz.x, sz.y, sz.z);
                model.scale.setScalar(sc);
                model.position.set(-ct.x * sc, -ct.y * sc, -ct.z * sc);
                lungPreviewScene.add(model);
                lungPreviewModel = model;
            },
            undefined,
            function (err) { console.warn('[LungPreview3D] GLB load failed, fallback', err); fallback(); }
        );
    } else {
        fallback();
    }

    function animate() {
        lungPreviewAnimId = requestAnimationFrame(animate);
        if (lungPreviewModel) lungPreviewModel.rotation.y += 0.008;
        if (lungPreviewRenderer) lungPreviewRenderer.render(lungPreviewScene, lungPreviewCamera);
    }
    animate();
    lungPreviewInited = true;

    // Pause when tab/section hidden — same pattern as the big lung 3D viewer
    document.addEventListener('visibilitychange', function () {
        if (document.hidden && lungPreviewAnimId) {
            cancelAnimationFrame(lungPreviewAnimId);
            lungPreviewAnimId = null;
        } else if (!document.hidden && lungPreviewInited && !lungPreviewAnimId) {
            animate();
        }
    });
}

function generateLungRecommendations(result) {
    const recos = [];

    if (!result.detected) {
        recos.push('✅ Không phát hiện tổn thương nghi ngờ trên ảnh CT này.');
        recos.push('Tiếp tục theo dõi định kỳ theo hướng dẫn của bác sĩ.');
        return recos;
    }

    const d = result.maxDiameterMm || 0;
    const m = result.malignancy || 0;

    if (d <= 6) {
        recos.push('Nốt phổi nhỏ (≤ 6mm) — theo dõi bằng CT sau 6-12 tháng theo guideline Fleischner.');
    } else if (d <= 10) {
        recos.push('Nốt phổi 6-10mm — đề xuất CT theo dõi sau 3-6 tháng.');
        recos.push('Cân nhắc PET/CT nếu nốt có đặc điểm nghi ngờ.');
    } else if (d <= 20) {
        recos.push('Khối u 1-2cm — nghi ngờ ác tính, cần đánh giá thêm.');
        recos.push('Đề xuất: Chụp PET/CT để đánh giá chuyển hóa.');
        recos.push('Cân nhắc sinh thiết xuyên thành ngực.');
    } else {
        recos.push('Khối u lớn (> 2cm) — cần đánh giá khẩn cấp.');
        recos.push('Đề xuất: PET/CT + sinh thiết + đánh giá giai đoạn.');
        recos.push('Chuyển bác sĩ chuyên khoa Ung bướu ngay.');
    }
    if (m > 70) recos.push('Xác suất ác tính cao (' + m.toFixed(1) + '%) — ưu tiên can thiệp sớm.');
    recos.push('Tham khảo ý kiến bác sĩ chuyên khoa Ung bướu - Hô hấp.');
    recos.push('Lịch tái khám gợi ý: 4 - 6 tuần.');
    return recos;
}

// ============ 3D LUNG VISUALIZATION (Three.js) ============
var lungScene, lungCamera, lungRenderer, lungControls;
var lungAnimFrameId = null;
var lungAutoRotate = true;
var lungTumorMesh = null;
var lungTumorGlow = null;
var lungParticles = null;
var lung3dInitialized = false;
// All meshes that should follow the tumor position — we move them as a group
// when bounds change (e.g., once the GLB lung model has finished loading).
var lungTumorParts = [];

// Convert an "anatomical hint" (which side / lobe of the lung the tumor is in)
// into world coordinates fitted to the lung model's actual bounding box.
// `anatomical.sx` and `.sy` are in [-1, +1] where +X is patient's RIGHT lung
// and +Y is the upper lobe. FILL controls how deep into the lung the tumor
// sits (0.55 keeps it well inside the mesh, away from the mediastinum and
// outer edge).
function computeLungTumorPosition(anatomical, bounds) {
    var b = bounds || { min: { x: -7, y: -8, z: -3 }, max: { x: 7, y: 7, z: 3 } };
    var bxC = (b.min.x + b.max.x) / 2;
    var byC = (b.min.y + b.max.y) / 2;
    var bzC = (b.min.z + b.max.z) / 2;
    var bxR = Math.max(0.5, (b.max.x - b.min.x) / 2);
    var byR = Math.max(0.5, (b.max.y - b.min.y) / 2);
    var bzR = Math.max(0.3, (b.max.z - b.min.z) / 2);

    // Label-match convention: tumor side in 3D matches where the L/R label
    // is drawn on the CT image. L on viewer's left → image-left = "trái" → -X.
    // R on viewer's right → image-right = "phải" → +X.
    var sideSign;       // -1 = viewer left ("trái"), +1 = viewer right ("phải")
    var lobeSign;       // +1 = upper lobe, 0 = middle, -1 = lower
    if (anatomical.kind === 'continuous' && typeof anatomical.sx === 'number') {
        // sx > 0 = image-left → "trái" → -X in 3D
        sideSign = anatomical.sx > 0 ? -1 : +1;
        lobeSign = Math.max(-1, Math.min(1, anatomical.sy || 0));
    } else {
        // Discrete: side label directly maps to 3D X direction
        sideSign = anatomical.side === 'right' ? +1 : -1;
        lobeSign = anatomical.lobe === 'upper' ? +1 : (anatomical.lobe === 'lower' ? -1 : 0);
    }

    return {
        x: bxC + sideSign * bxR * 0.55,    // dead center of the chosen lung
        y: byC + lobeSign * byR * 0.45,    // upper / middle / lower lobe
        z: bzC + bzR * 0.20,               // slightly forward of center
    };
}

// Move the tumor mesh + every related part (wireframe, glow shells, point
// light) to a new position. Called after the GLB lung model has loaded so
// we can use its actual bounds.
function repositionLungTumor(bounds) {
    if (!lungTumorMesh || !window._lungTumorAnatomical) return;
    var pos = computeLungTumorPosition(window._lungTumorAnatomical, bounds);
    lungTumorMesh.position.set(pos.x, pos.y, pos.z);
    lungTumorParts.forEach(function (p) {
        if (p && p.position) p.position.set(pos.x, pos.y, pos.z);
    });
    console.log('[Lung3D] tumor repositioned to (' +
                pos.x.toFixed(2) + ', ' + pos.y.toFixed(2) + ', ' + pos.z.toFixed(2) + ')');
}

function initLung3D(tumorData) {
  try {
    var container = document.getElementById('lungVizStage');
    var canvas = document.getElementById('lung3dCanvas');
    var placeholder = document.getElementById('lungVizPlaceholder');
    if (!container || !canvas) { console.error('[Lung3D] missing elements'); return; }
    if (placeholder) placeholder.style.display = 'none';
    disposeLung3D();
    var rect = container.getBoundingClientRect();
    var w = Math.max(rect.width, 300), h = Math.max(rect.height, 200);
    console.log('[Lung3D] init', w, 'x', h);
    lungScene = new THREE.Scene();
    lungScene.background = null;   // let CSS gradient show through
    lungCamera = new THREE.PerspectiveCamera(35, w / h, 0.1, 500);
    // Camera will be repositioned to fit-frame after the model loads.
    lungCamera.position.set(0, 4, 40);
    lungCamera.lookAt(0, 0, 0);
    lungRenderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
    lungRenderer.setClearColor(0x000000, 0);
    lungRenderer.setSize(w, h);
    lungRenderer.setPixelRatio(window.devicePixelRatio > 1 ? 2 : 1);
    lungControls = new THREE.OrbitControls(lungCamera, lungRenderer.domElement);
    lungControls.enableDamping = true;
    lungControls.autoRotate = true;
    lungControls.autoRotateSpeed = 0.9;       // slower, more cinematic
    lungControls.target.set(0, 0, 0);
    // Lighting
    lungScene.add(new THREE.AmbientLight(0xffffff, 0.55));
    var dl1 = new THREE.DirectionalLight(0xeaf6ff, 1.4); dl1.position.set(15, 20, 25); lungScene.add(dl1);
    var dl2 = new THREE.DirectionalLight(0x4488bb, 0.6); dl2.position.set(-15, 10, -20); lungScene.add(dl2);
    // Rim light from behind to outline the lung silhouette
    var rim = new THREE.DirectionalLight(0x33eaff, 0.9); rim.position.set(0, -5, -22); lungScene.add(rim);
    // === LOAD 3D LUNG MODEL (GLB) or FALLBACK ===
    var lungModelLoaded = false;
    function addFallbackLungs() {
      var lm = new THREE.MeshLambertMaterial({ color: 0x1a99bb, transparent: true, opacity: 0.32, side: THREE.DoubleSide, depthWrite: false });
      var wm = new THREE.MeshBasicMaterial({ color: 0x33ccee, wireframe: true, transparent: true, opacity: 0.06 });
      // Right lung
      var rL = new THREE.Mesh(new THREE.SphereGeometry(1,28,20), lm.clone());
      rL.scale.set(5.2,8.5,3.5); rL.position.set(-5.5,-1,0); lungScene.add(rL);
      var rW = new THREE.Mesh(rL.geometry, wm.clone()); rW.scale.copy(rL.scale); rW.position.copy(rL.position); lungScene.add(rW);
      // Left lung
      var lL = new THREE.Mesh(new THREE.SphereGeometry(1,28,20), lm.clone());
      lL.scale.set(4.5,7.8,3.2); lL.position.set(5.2,-1.5,0); lungScene.add(lL);
      var lW = new THREE.Mesh(lL.geometry, wm.clone()); lW.scale.copy(lL.scale); lW.position.copy(lL.position); lungScene.add(lW);
      // Trachea
      var pm = new THREE.MeshLambertMaterial({color:0x44ccee,transparent:true,opacity:0.5});
      var tra = new THREE.Mesh(new THREE.CylinderGeometry(0.6,0.6,5,10),pm);
      tra.position.set(0,10,0); lungScene.add(tra);
      var rBr = new THREE.Mesh(new THREE.CylinderGeometry(0.4,0.25,4.5,8),pm);
      rBr.position.set(-2.5,6,0); rBr.rotation.z=0.45; lungScene.add(rBr);
      var lBr = new THREE.Mesh(new THREE.CylinderGeometry(0.4,0.25,4.5,8),pm);
      lBr.position.set(2.5,6,0); lBr.rotation.z=-0.45; lungScene.add(lBr);
      // Set bounds + reposition tumor so it lands inside the fallback lungs
      window._lungModelBounds = {
        min: { x: -10.7, y: -9.5, z: -3.5 },
        max: { x:   9.7, y:  7.5, z:  3.5 }
      };
      repositionLungTumor(window._lungModelBounds);
      console.log('[Lung3D] Using fallback ellipsoid lungs');
    }

    // === LOAD GLB MODEL ===
    function loadLungModel() {
      if (typeof THREE.GLTFLoader === 'undefined') {
        console.error('[Lung3D] GLTFLoader not loaded!');
        addFallbackLungs(); return;
      }
      // Show loading text on canvas
      var loadingDiv = document.createElement('div');
      loadingDiv.id = 'lung3dLoading';
      loadingDiv.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#44ddff;font-size:14px;z-index:10;text-align:center;';
      loadingDiv.innerHTML = '⏳ Đang tải mô hình 3D phổi...';
      container.appendChild(loadingDiv);

      var loader = new THREE.GLTFLoader();
      console.log('[Lung3D] Starting GLB load...');
      loader.load('/models/lungs.glb',
        function onLoad(gltf) {
          console.log('[Lung3D] ✅ GLB loaded successfully!');
          if (loadingDiv.parentNode) loadingDiv.parentNode.removeChild(loadingDiv);
          var model = gltf.scene;
          var mc = 0;
          model.traverse(function(c) {
            if (c.isMesh) {
              mc++;
              // Ghost-like translucent lung — much more transparent so the
              // tumor inside stands out clearly against it.
              c.material = new THREE.MeshLambertMaterial({
                color: 0x33ccdd, transparent: true, opacity: 0.18,
                side: THREE.DoubleSide, depthWrite: false,
                emissive: 0x114455, emissiveIntensity: 0.15,
              });
            }
          });
          console.log('[Lung3D] Meshes found:', mc);
          // Auto-fit: normalize model to a target world size, then push the
          // camera back enough so the whole model fits the viewport with
          // padding — works for any canvas aspect ratio.
          var box = new THREE.Box3().setFromObject(model);
          var sz = box.getSize(new THREE.Vector3());
          var ct = box.getCenter(new THREE.Vector3());
          var mx = Math.max(sz.x, sz.y, sz.z);
          var TARGET = 12;                           // smaller than before (was 18)
          var sc = TARGET / mx;
          model.scale.setScalar(sc);
          model.position.set(-ct.x * sc, -ct.y * sc, -ct.z * sc);
          lungScene.add(model);

          // Recompute after transform
          var finalBox = new THREE.Box3().setFromObject(model);
          var fMin = finalBox.min, fMax = finalBox.max;
          window._lungModelBounds = { min: {x:fMin.x, y:fMin.y, z:fMin.z}, max: {x:fMax.x, y:fMax.y, z:fMax.z} };

          // FOV-based fit-to-frame
          var fitOffset = 1.6;   // 60% padding around the model
          var maxWorldDim = Math.max(fMax.x - fMin.x, fMax.y - fMin.y, fMax.z - fMin.z);
          var fov = lungCamera.fov * Math.PI / 180;
          var aspect = lungCamera.aspect || 1;
          var fitH = (maxWorldDim / 2) / Math.tan(fov / 2);
          var fitW = fitH / aspect;
          var dist = Math.max(fitH, fitW) * fitOffset;
          lungCamera.position.set(0, maxWorldDim * 0.15, dist);
          lungCamera.lookAt(0, 0, 0);
          lungCamera.near = Math.max(0.1, dist / 100);
          lungCamera.far  = dist * 10;
          lungCamera.updateProjectionMatrix();
          if (lungControls) {
            lungControls.minDistance = dist * 0.3;
            lungControls.maxDistance = dist * 3;
            lungControls.target.set(0, 0, 0);
            lungControls.update();
          }
          console.log('[Lung3D] fit dist=' + dist.toFixed(1) + ', maxWorldDim=' + maxWorldDim.toFixed(1));
          console.log('[Lung3D] bounds X[' + fMin.x.toFixed(1) + ',' + fMax.x.toFixed(1) +
                      '] Y[' + fMin.y.toFixed(1) + ',' + fMax.y.toFixed(1) +
                      '] Z[' + fMin.z.toFixed(1) + ',' + fMax.z.toFixed(1) + ']');
          // Reposition tumor using actual lung bounds (not the default
          // estimate it got placed at when initial setup ran).
          repositionLungTumor(window._lungModelBounds);
        },
        function onProgress(p) {
          if (p.total > 0) {
            var pct = Math.round(p.loaded / p.total * 100);
            loadingDiv.innerHTML = '⏳ Tải mô hình: ' + pct + '%';
            if (pct % 20 === 0) console.log('[Lung3D] ' + pct + '%');
          }
        },
        function onError(err) {
          console.error('[Lung3D] ❌ GLB load error:', err);
          if (loadingDiv.parentNode) loadingDiv.parentNode.removeChild(loadingDiv);
          addFallbackLungs();
        }
      );
    }
    loadLungModel();

    // TUMOR — placed at correct anatomical position inside lung
    if(tumorData && tumorData.detected && tumorData.tumors && tumorData.tumors.length>0){
      var t0 = tumorData.tumors[0];
      var contour = t0.contourNorm || null;
      var bbox = t0.bboxNorm || t0.bbox || {w:50, h:50};
      var posText = (t0.position || tumorData.position || '');
      var posSubText = (t0.positionSub || tumorData.positionSub || '').toLowerCase();

      // ANATOMICAL POSITION — continuous mapping from segmentation centroid
      // (centroidNorm is in 512x512 image space). Falls back to bucketed
      // text mapping only when centroid coords are missing.
      // Radiological convention: image-left = patient RIGHT lung (3D +X),
      // image-right = patient LEFT lung (3D -X), image-top = upper lobe (+Y).
      var centroidNorm = t0.centroidNorm;
      var isRight = posSubText.indexOf('right') >= 0;
      var isLeft = posSubText.indexOf('left') >= 0;
      // Fallback to Vietnamese if English not available
      if (!isRight && !isLeft) {
        isRight = posText.indexOf('ph\u1ea3i') >= 0;  // phải
        isLeft = posText.indexOf('tr\u00e1i') >= 0;   // trái
      }
      if (!isRight && !isLeft) isRight = true; // default

      // Lobe detection from English sub text
      var isUpper = posSubText.indexOf('upper') >= 0;
      var isLower = posSubText.indexOf('lower') >= 0;
      var isMiddle = posSubText.indexOf('middle') >= 0;

      // ---- ANATOMICAL HINT — what lobe + side the tumor is in ----
      // We store an "intent" (sx, sy in [-1,+1] range) instead of world coords,
      // then convert to world coords using the LUNG MODEL's actual bounds when
      // we know them. This avoids the tumor floating outside the mesh.
      var anatomical;
      var isLIDCCrop = !!(tumorData.datasetInfo && tumorData.datasetInfo.patientId) &&
                       centroidNorm &&
                       Math.abs((centroidNorm.x || 256) - 256) < 80 &&
                       Math.abs((centroidNorm.y || 256) - 256) < 80;

      if (isLIDCCrop) {
        // LIDC hash → discrete side+lobe (matches what the BE position string
        // says, so the 3D placement is consistent with "Vị trí: Phổi phải /
        // Thuỳ trên" displayed in the patient strip).
        var sideHint = (posSubText.indexOf('right') >= 0 || posText.indexOf('phải') >= 0) ? 'right' : 'left';
        var lobeHint = posSubText.indexOf('upper') >= 0 ? 'upper'
                     : (posSubText.indexOf('lower') >= 0 ? 'lower' : 'middle');
        anatomical = {
          kind: 'lidc-hash',
          side: sideHint,
          lobe: lobeHint,
          patient: tumorData.datasetInfo.patientId,
        };
        console.log('[Lung3D] anatomical (LIDC) side=' + sideHint + ' lobe=' + lobeHint);
      } else if (centroidNorm && typeof centroidNorm.x === 'number' &&
                 typeof centroidNorm.y === 'number') {
        var nx = centroidNorm.x / 512;
        var ny = centroidNorm.y / 512;
        anatomical = {
          kind: 'continuous',
          sx: 1 - 2 * nx,            // image-left → patient-right → sx > 0
          sy: 1 - 2 * ny,            // image-top  → upper lobe   → sy > 0
        };
        console.log('[Lung3D] anatomical (continuous) sx=' +
                    anatomical.sx.toFixed(2) + ' sy=' + anatomical.sy.toFixed(2));
      } else {
        anatomical = {
          kind: 'bucketed',
          side: isRight ? 'right' : (isLeft ? 'left' : 'right'),
          lobe: isUpper ? 'upper' : (isLower ? 'lower' : 'middle'),
        };
        console.log('[Lung3D] anatomical (bucketed) "' + posText + '"');
      }
      window._lungTumorAnatomical = anatomical;

      // Initial world coords using current bounds (may be defaults; will be
      // recomputed after GLB load with real bounds).
      var initialPos = computeLungTumorPosition(anatomical, window._lungModelBounds);
      var cx3d = initialPos.x, cy3d = initialPos.y, cz3d = initialPos.z;

      // Tumor world-size scales with measured diameter, clamped tightly so
      // even noisy LIDC ">100mm" measurements stay anatomically reasonable.
      // Calibration: 20mm tumor → 0.5 world units (≈ 5% of lung X — close to
      // real anatomy where a 2cm tumor is ~5–8% of a 25cm lung).
      var diameterMm = (tumorData && tumorData.maxDiameterMm) || 20;
      var TUMOR_WORLD_SIZE = Math.max(0.25, Math.min(1.0, diameterMm * 0.025));

      if (contour && contour.length >= 3) {
        // Build extruded shape from the actual segmentation contour — keeps
        // the irregular real tumor outline (anatomical realism). Size is
        // already clamped small via TUMOR_WORLD_SIZE.
        var contCx = 0, contCy = 0;
        for (var ci = 0; ci < contour.length; ci++) {
          contCx += contour[ci][0]; contCy += contour[ci][1];
        }
        contCx /= contour.length; contCy /= contour.length;
        var maxR = 0.001;
        for (var ci = 0; ci < contour.length; ci++) {
          var dxc = contour[ci][0] - contCx, dyc = contour[ci][1] - contCy;
          var r = Math.sqrt(dxc * dxc + dyc * dyc);
          if (r > maxR) maxR = r;
        }
        var tumorScale = (TUMOR_WORLD_SIZE * 0.5) / maxR;

        var shape = new THREE.Shape();
        var firstX = (contour[0][0] - contCx) * tumorScale;
        var firstY = (contCy - contour[0][1]) * tumorScale;
        shape.moveTo(firstX, firstY);
        for (var ci = 1; ci < contour.length; ci++) {
          var px = (contour[ci][0] - contCx) * tumorScale;
          var py = (contCy - contour[ci][1]) * tumorScale;
          shape.lineTo(px, py);
        }
        shape.lineTo(firstX, firstY);

        var depth = TUMOR_WORLD_SIZE * 0.35;
        var tumorGeo = new THREE.ExtrudeGeometry(shape, {
          depth: depth,
          bevelEnabled: true,
          bevelThickness: depth * 0.20,
          bevelSize: depth * 0.15,
          bevelSegments: 4,
          curveSegments: 12,
        });
        tumorGeo.center();

        // Reset list of meshes to follow the tumor
        lungTumorParts = [];

        lungTumorMesh = new THREE.Mesh(tumorGeo, new THREE.MeshLambertMaterial({
          color: 0xff3355, transparent: true, opacity: 0.92,
          emissive: 0x661111, emissiveIntensity: 0.55,
          side: THREE.DoubleSide,
        }));
        lungTumorMesh.position.set(cx3d, cy3d, cz3d);
        lungScene.add(lungTumorMesh);
        lungTumorParts.push(lungTumorMesh);

        // Wireframe outline
        var wireMesh = new THREE.Mesh(tumorGeo.clone(), new THREE.MeshBasicMaterial({
          color: 0xff5577, wireframe: true, transparent: true, opacity: 0.18,
        }));
        wireMesh.position.copy(lungTumorMesh.position);
        wireMesh.scale.set(1.04, 1.04, 1.04);
        lungScene.add(wireMesh);
        lungTumorParts.push(wireMesh);

        // Glow shells around the tumor
        tumorGeo.computeBoundingSphere();
        var glowR = tumorGeo.boundingSphere ? tumorGeo.boundingSphere.radius : 0.5;
        [1.5, 2.2].forEach(function (s, i) {
          var gg = new THREE.Mesh(
            new THREE.SphereGeometry(glowR * s, 16, 12),
            new THREE.MeshBasicMaterial({
              color: 0xff3355, transparent: true,
              opacity: 0.05 - i * 0.02,
              side: THREE.BackSide,
            })
          );
          gg.position.copy(lungTumorMesh.position);
          lungScene.add(gg);
          lungTumorParts.push(gg);
          if (i === 0) lungTumorGlow = gg;
        });

      } else {
        // Reset list of meshes to follow the tumor
        lungTumorParts = [];
        // Fallback smooth ellipsoid (no contour available)
        var bw = (bbox && bbox.w) || 30;
        var bh = (bbox && bbox.h) || 30;
        var maxBb = Math.max(bw, bh);
        var rx = TUMOR_WORLD_SIZE * 0.5 * (bw / maxBb);
        var ry = TUMOR_WORLD_SIZE * 0.5 * (bh / maxBb);
        var rz = TUMOR_WORLD_SIZE * 0.5 * 0.78;
        lungTumorMesh = new THREE.Mesh(
          new THREE.SphereGeometry(1, 40, 28),
          new THREE.MeshPhongMaterial({
            color: 0xff3355, transparent: true, opacity: 0.95,
            emissive: 0x661111, emissiveIntensity: 0.45,
            shininess: 60, specular: 0xffaaaa,
          })
        );
        lungTumorMesh.scale.set(rx, ry, rz);
        lungTumorMesh.position.set(cx3d, cy3d, cz3d);
        lungScene.add(lungTumorMesh);
        lungTumorParts.push(lungTumorMesh);
        lungTumorGlow = lungTumorMesh;
      }

      // Point light at tumor (range scaled to 12u lung)
      var tLt = new THREE.PointLight(0xff2244, 0.8, 8);
      tLt.position.copy(lungTumorMesh.position);
      lungScene.add(tLt);
      lungTumorParts.push(tLt);
    }
    // Particles
    var pg=new THREE.BufferGeometry(); var pa=new Float32Array(200*3);
    for(var j=0;j<pa.length;j++) pa[j]=(Math.random()-0.5)*40;
    pg.setAttribute('position',new THREE.BufferAttribute(pa,3));
    lungParticles=new THREE.Points(pg,new THREE.PointsMaterial({color:0x3399bb,size:0.12,transparent:true,opacity:0.15}));
    lungScene.add(lungParticles);
    var grd=new THREE.GridHelper(30,15,0x152535,0x0c1520); grd.position.y=-10; lungScene.add(grd);
    // First render
    lungRenderer.render(lungScene, lungCamera);
    console.log('[Lung3D] Render OK');
    lung3dInitialized = true;
    animateLung3D();
    setupLung3DControls();
  } catch(e) {
    console.error('[Lung3D] ERROR:', e.message, e.stack);
  }
}

function gsapLungFlyIn() {
    const targetPos = { x: 20, y: 12, z: 35 };
    const startPos = { x: lungCamera.position.x, y: lungCamera.position.y, z: lungCamera.position.z };
    const startTime = performance.now();
    const duration = 1500;

    function flyStep(now) {
        const t = Math.min((now - startTime) / duration, 1);
        const ease = 1 - Math.pow(1 - t, 3); // ease out cubic
        lungCamera.position.x = startPos.x + (targetPos.x - startPos.x) * ease;
        lungCamera.position.y = startPos.y + (targetPos.y - startPos.y) * ease;
        lungCamera.position.z = startPos.z + (targetPos.z - startPos.z) * ease;
        lungCamera.lookAt(0, -2, 0);
        if (t < 1) requestAnimationFrame(flyStep);
    }
    requestAnimationFrame(flyStep);
}

function animateLung3D() {
    if (!lung3dInitialized) return;
    lungAnimFrameId = requestAnimationFrame(animateLung3D);

    const time = performance.now() * 0.001;

    // Tumor pulsing animation — multiplicative on the original scale so we
    // don't squash an ellipsoid into a sphere.
    if (lungTumorMesh) {
        if (!lungTumorMesh.userData.baseScale) {
            lungTumorMesh.userData.baseScale = lungTumorMesh.scale.clone();
        }
        var pulse = 1 + Math.sin(time * 2.4) * 0.10;
        var bs = lungTumorMesh.userData.baseScale;
        lungTumorMesh.scale.set(bs.x * pulse, bs.y * pulse, bs.z * pulse);
        if (lungTumorMesh.material && lungTumorMesh.material.emissiveIntensity !== undefined) {
            lungTumorMesh.material.emissiveIntensity = 0.7 + Math.sin(time * 2) * 0.3;
        }
    }
    if (lungTumorGlow) {
        if (!lungTumorGlow.userData.baseScale) {
            lungTumorGlow.userData.baseScale = lungTumorGlow.scale.clone();
        }
        var glowPulse = 1 + Math.sin(time * 1.6) * 0.20;
        var gbs = lungTumorGlow.userData.baseScale;
        lungTumorGlow.scale.set(gbs.x * glowPulse, gbs.y * glowPulse, gbs.z * glowPulse);
        if (lungTumorGlow.material) {
            lungTumorGlow.material.opacity = 0.10 + Math.sin(time * 1.5) * 0.06;
        }
    }
    // Project tumor + L/R anchor world positions to 2D screen and move
    // their floating labels. Anchors stay in world space → labels follow
    // the lung mesh as the user orbits.
    if (lungCamera && lungRenderer) {
        var stage = document.getElementById('lungVizStage');
        var rect = stage ? stage.getBoundingClientRect() : null;

        function _placeLabel(elId, world, dx, dy) {
            var el = document.getElementById(elId);
            if (!el || !rect) return;
            var p = world.clone().project(lungCamera);
            var sx = (p.x * 0.5 + 0.5) * rect.width;
            var sy = (-p.y * 0.5 + 0.5) * rect.height;
            var visible = p.z > -1 && p.z < 1 && sx > 0 && sx < rect.width && sy > 0 && sy < rect.height;
            el.style.transform = 'translate(' + (sx + dx) + 'px, ' + (sy + dy) + 'px)';
            el.style.opacity = visible ? '1' : '0';
        }

        if (lungTumorMesh) _placeLabel('lungTumorLabel', lungTumorMesh.position, 18, -14);

        var bounds = window._lungModelBounds;
        if (bounds) {
            var bxR = (bounds.max.x - bounds.min.x) / 2;
            var byC = (bounds.min.y + bounds.max.y) / 2;
            var bzC = (bounds.min.z + bounds.max.z) / 2;
            // Natural-label: L on viewer's left (-X), R on viewer's right (+X).
            // Anchor slightly inside each lung (0.55 of half-width).
            _placeLabel('lungSideLabelL', new THREE.Vector3(-bxR * 0.55, byC, bzC), -22, -10);
            _placeLabel('lungSideLabelR', new THREE.Vector3(+bxR * 0.55, byC, bzC), -8, -10);
        }
    }

    // Particle drift
    if (lungParticles) {
        const positions = lungParticles.geometry.attributes.position.array;
        for (let i = 0; i < positions.length; i += 3) {
            positions[i + 1] += Math.sin(time + i) * 0.003;
        }
        lungParticles.geometry.attributes.position.needsUpdate = true;
    }

    // Controls update
    if (lungControls) {
        lungControls.autoRotate = lungAutoRotate;
        lungControls.update();
    }

    if (lungRenderer && lungScene && lungCamera) {
        lungRenderer.render(lungScene, lungCamera);
    }
}

function disposeLung3D() {
    if (lungAnimFrameId) {
        cancelAnimationFrame(lungAnimFrameId);
        lungAnimFrameId = null;
    }
    if (lungScene) {
        lungScene.traverse(obj => {
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) {
                if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
                else obj.material.dispose();
            }
        });
    }
    if (lungRenderer) {
        lungRenderer.dispose();
    }
    lungScene = null;
    lungCamera = null;
    lungRenderer = null;
    lungControls = null;
    lungTumorMesh = null;
    lungTumorGlow = null;
    lungParticles = null;
    lung3dInitialized = false;
}

function setLungCamera(px, py, pz) {
    if (!lungCamera || !lungControls) return;
    const startPos = lungCamera.position.clone();
    const endPos = new THREE.Vector3(px, py, pz);
    const startTime = performance.now();
    const duration = 800;

    function step(now) {
        const t = Math.min((now - startTime) / duration, 1);
        const ease = 1 - Math.pow(1 - t, 3);
        lungCamera.position.lerpVectors(startPos, endPos, ease);
        lungControls.target.set(0, -2, 0);
        lungControls.update();
        if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

function setupLung3DControls() {
    // Auto-rotate toggle
    const btnRotate = document.getElementById('lungToolRotate');
    if (btnRotate) {
        btnRotate.onclick = () => {
            lungAutoRotate = !lungAutoRotate;
            btnRotate.classList.toggle('active', lungAutoRotate);
            showToast(lungAutoRotate ? '⟳ Tự động xoay: BẬT' : '⟳ Tự động xoay: TẮT', 'info');
        };
    }

    // Zoom in
    const btnZoom = document.getElementById('lungToolZoom');
    if (btnZoom) {
        btnZoom.onclick = () => {
            if (lungCamera) {
                const dir = lungCamera.position.clone().normalize();
                lungCamera.position.addScaledVector(dir, -5);
            }
        };
    }

    // Pan (reset target)
    const btnPan = document.getElementById('lungToolPan');
    if (btnPan) {
        btnPan.onclick = () => {
            if (lungControls) {
                lungControls.target.set(0, -2, 0);
                lungControls.update();
            }
        };
    }

    // Reset camera
    const btnReset = document.getElementById('lungToolReset');
    if (btnReset) {
        btnReset.onclick = () => {
            setLungCamera(20, 12, 35);
            lungAutoRotate = false;
            const btnR = document.getElementById('lungToolRotate');
            if (btnR) btnR.classList.remove('active');
        };
    }

    // Camera presets
    const views = {
        lungViewFront: [0, 0, 40],
        lungViewBack: [0, 0, -40],
        lungViewLeft: [-40, 0, 0],
        lungViewRight: [40, 0, 0],
        lungViewTop: [0, 40, 0],
    };
    Object.entries(views).forEach(([id, pos]) => {
        const el = document.getElementById(id);
        if (el) {
            el.style.cursor = 'pointer';
            el.onclick = () => setLungCamera(...pos);
        }
    });

    // Resize observer
    const container = document.getElementById('lungVizStage');
    if (container && lungRenderer && lungCamera) {
        const ro = new ResizeObserver(() => {
            const w = container.clientWidth;
            const h = container.clientHeight;
            if (w > 0 && h > 0 && lungRenderer && lungCamera) {
                lungCamera.aspect = w / h;
                lungCamera.updateProjectionMatrix();
                lungRenderer.setSize(w, h);
            }
        });
        ro.observe(container);
    }
}

// ============ BLOOD TEST — client-side analysis (rules-based) ============
const BLOOD_RANGES = {
    wbc:   { name: 'WBC',         unit: 'K/µL',    min: 4.0,  max: 10.0 },
    rbc:   { name: 'RBC',         unit: 'M/µL',    min: 4.5,  max: 5.9  },
    hgb:   { name: 'Hemoglobin',  unit: 'g/dL',    min: 13.5, max: 17.5 },
    hct:   { name: 'Hematocrit',  unit: '%',       min: 41,   max: 53   },
    plt:   { name: 'Platelets',   unit: 'K/µL',    min: 150,  max: 400  },
    glu:   { name: 'Glucose',     unit: 'mmol/L',  min: 3.9,  max: 6.1  },
    hba1c: { name: 'HbA1c',       unit: '%',       min: 0,    max: 5.7  },
    chol:  { name: 'Cholesterol', unit: 'mmol/L',  min: 0,    max: 5.2  },
    ldl:   { name: 'LDL',         unit: 'mmol/L',  min: 0,    max: 3.4  },
    hdl:   { name: 'HDL',         unit: 'mmol/L',  min: 1.0,  max: 99   },
    trig:  { name: 'Triglycerides', unit: 'mmol/L', min: 0,    max: 1.7  },
    alt:   { name: 'ALT',         unit: 'U/L',     min: 7,    max: 55   },
    ast:   { name: 'AST',         unit: 'U/L',     min: 8,    max: 48   },
    crea:  { name: 'Creatinine',  unit: 'µmol/L',  min: 62,   max: 106  },
    ure:   { name: 'Ure / BUN',   unit: 'mmol/L',  min: 2.5,  max: 7.1  },
};

const BLOOD_GROUPS = {
    'Huyết học':  ['wbc', 'rbc', 'hgb', 'hct', 'plt'],
    'Sinh hoá':   ['glu', 'hba1c', 'chol', 'ldl', 'hdl', 'trig'],
    'Gan / Thận': ['alt', 'ast', 'crea', 'ure'],
};

function fillDemoBlood() {
    const demo = {
        wbc: 11.2, rbc: 4.8, hgb: 12.8, hct: 39, plt: 220,
        glu: 7.8, hba1c: 6.4, chol: 5.6, ldl: 3.8, hdl: 0.9, trig: 2.3,
        alt: 62, ast: 51, crea: 98, ure: 5.5
    };
    Object.keys(demo).forEach(k => {
        const el = document.getElementById('b_' + k);
        if (el) el.value = demo[k];
    });
    showToast('📋 Đã điền dữ liệu mẫu', 'info');
}

function runBloodAnalysis() {
    const rows = [];
    let countNormal = 0, countMild = 0, countAbn = 0;
    const groupStatus = {};
    Object.keys(BLOOD_GROUPS).forEach(g => groupStatus[g] = { ok: 0, warn: 0, bad: 0 });

    Object.keys(BLOOD_RANGES).forEach(key => {
        const def = BLOOD_RANGES[key];
        const input = document.getElementById('b_' + key);
        const raw = input ? input.value : '';
        if (raw === '' || raw == null) return;
        const v = parseFloat(raw);
        let status = 'normal', label = 'Bình thường';
        if (v < def.min) {
            const pct = (def.min - v) / def.min;
            if (pct > 0.2) { status = 'low'; label = 'Thấp'; countAbn++; }
            else           { status = 'mild'; label = 'Hơi thấp'; countMild++; }
        } else if (v > def.max) {
            const pct = (v - def.max) / def.max;
            if (pct > 0.3) { status = 'high'; label = 'Cao ↑↑'; countAbn++; }
            else           { status = 'mild'; label = 'Cao nhẹ ↑'; countMild++; }
        } else {
            countNormal++;
        }

        // Group accounting
        for (const g of Object.keys(BLOOD_GROUPS)) {
            if (BLOOD_GROUPS[g].includes(key)) {
                if (status === 'normal') groupStatus[g].ok++;
                else if (status === 'mild') groupStatus[g].warn++;
                else groupStatus[g].bad++;
                break;
            }
        }

        rows.push(`
            <tr>
                <td>${def.name} <small style="color:var(--text-muted)">(${def.unit})</small></td>
                <td><b>${v}</b></td>
                <td style="color:var(--text-muted)">${def.min} – ${def.max}</td>
                <td><span class="status-badge ${status}">${label}</span></td>
            </tr>
        `);
    });

    if (rows.length === 0) {
        showToast('⚠️ Vui lòng nhập ít nhất 1 chỉ số', 'error');
        return;
    }

    document.getElementById('bloodTableBody').innerHTML = rows.join('');
    document.getElementById('bloodNormalCount').textContent = countNormal;
    document.getElementById('bloodMildCount').textContent = countMild;
    document.getElementById('bloodAbnormalCount').textContent = countAbn;
    document.getElementById('bloodTestDate').textContent = new Date().toLocaleDateString('vi-VN');

    // Risk assessment (rules-based, mock)
    const getNum = k => {
        const el = document.getElementById('b_' + k);
        return el && el.value !== '' ? parseFloat(el.value) : null;
    };
    const risks = [];
    const glu = getNum('glu'), hba1c = getNum('hba1c');
    if (glu != null || hba1c != null) {
        let r = 0;
        if (glu != null && glu > 7) r = Math.max(r, 0.8);
        else if (glu != null && glu > 6.1) r = Math.max(r, 0.5);
        if (hba1c != null && hba1c >= 6.5) r = Math.max(r, 0.85);
        else if (hba1c != null && hba1c >= 5.7) r = Math.max(r, 0.55);
        risks.push({ label: 'Tiểu đường', value: r, color: '#ef4444' });
    }
    const hgb = getNum('hgb');
    if (hgb != null) {
        const r = hgb < 12 ? 0.85 : hgb < 13.5 ? 0.5 : 0.1;
        risks.push({ label: 'Thiếu máu', value: r, color: '#f43f5e' });
    }
    const alt = getNum('alt'), ast = getNum('ast');
    if (alt != null || ast != null) {
        let r = 0.1;
        if (alt != null && alt > 55) r = Math.max(r, 0.6);
        if (ast != null && ast > 48) r = Math.max(r, 0.6);
        if ((alt != null && alt > 100) || (ast != null && ast > 100)) r = 0.85;
        risks.push({ label: 'Rối loạn gan', value: r, color: '#f59e0b' });
    }
    const crea = getNum('crea'), ure = getNum('ure');
    if (crea != null || ure != null) {
        let r = 0.1;
        if (crea != null && crea > 106) r = Math.max(r, 0.55);
        if (ure != null && ure > 7.1) r = Math.max(r, 0.45);
        if (crea != null && crea > 150) r = 0.85;
        risks.push({ label: 'Rối loạn thận', value: r, color: '#06b6d4' });
    }
    const ldl = getNum('ldl'), trig = getNum('trig'), chol = getNum('chol');
    if (ldl != null || trig != null || chol != null) {
        let r = 0.1;
        if (ldl != null && ldl > 3.4) r = Math.max(r, 0.55);
        if (trig != null && trig > 1.7) r = Math.max(r, 0.5);
        if (chol != null && chol > 5.2) r = Math.max(r, 0.5);
        if ((ldl != null && ldl > 4.9) || (trig != null && trig > 2.3)) r = 0.8;
        risks.push({ label: 'Mỡ máu (Dyslipidemia)', value: r, color: '#8b5cf6' });
    }

    document.getElementById('bloodRiskBars').innerHTML =
        risks.length === 0
        ? '<div class="prob-row empty">Nhập thêm chỉ số để đánh giá nguy cơ</div>'
        : risks.map(r => `
            <div class="prob-row">
                <span>${r.label}</span>
                <div class="prob-bar-track"><div class="prob-bar-fill" style="width:${r.value*100}%; background:${r.color};"></div></div>
                <span class="pct">${(r.value*100).toFixed(0)}%</span>
            </div>
        `).join('');

    // Group status
    const groupEl = document.getElementById('bloodGroupStatus');
    groupEl.innerHTML = Object.keys(BLOOD_GROUPS).map(g => {
        const s = groupStatus[g];
        let cls = 'ok', label = 'Bình thường';
        if (s.bad > 0)       { cls = 'danger'; label = 'Bất thường'; }
        else if (s.warn > 0) { cls = 'warn';   label = 'Cần chú ý'; }
        return `<div class="group-status ${cls}"><span>${g}</span><b>${label}</b></div>`;
    }).join('');

    // Recommendations
    const recos = [];
    if (glu != null && glu > 6.1) recos.push('Glucose cao — tầm soát tiểu đường (chụp HbA1c, nghiệm pháp dung nạp glucose).');
    if (hba1c != null && hba1c >= 6.5) recos.push('HbA1c ≥ 6.5% — đủ tiêu chuẩn chẩn đoán tiểu đường, cần gặp BS Nội tiết.');
    if (hgb != null && hgb < 13.5) recos.push('Hemoglobin thấp — tầm soát thiếu máu (ferritin, vitamin B12, acid folic).');
    if (alt != null && alt > 55) recos.push('ALT cao — đánh giá chức năng gan, siêu âm bụng.');
    if (ldl != null && ldl > 3.4) recos.push('LDL cao — điều chỉnh chế độ ăn, cân nhắc statin theo chỉ định BS tim mạch.');
    if (crea != null && crea > 106) recos.push('Creatinine cao — đánh giá chức năng thận (eGFR, siêu âm thận).');
    if (recos.length === 0) recos.push('Các chỉ số trong ngưỡng bình thường. Duy trì lối sống lành mạnh và xét nghiệm định kỳ.');
    recos.push('Nên tham khảo bác sĩ chuyên khoa để có tư vấn chi tiết theo bệnh sử cá nhân.');
    document.getElementById('bloodRecoList').innerHTML =
        recos.map(r => `<li class="reco-item">${r}</li>`).join('');

    showToast(`🩸 Phân tích xong: ${countNormal} OK, ${countMild} chú ý, ${countAbn} bất thường`, 'success');
}

// ============ Shared: progression chart (uses Chart.js) ============
const _progressCharts = {};
function renderProgressionChart(canvasId, data, unit, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return;
    if (_progressCharts[canvasId]) {
        _progressCharts[canvasId].destroy();
    }
    _progressCharts[canvasId] = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
            labels: data.map(d => d.date),
            datasets: [{
                data: data.map(d => d.volume),
                borderColor: color,
                backgroundColor: color + '33',
                pointBackgroundColor: data.map((d, i) => i === data.length - 1 ? '#ef4444' : color),
                pointRadius: 5,
                tension: 0.35,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8', font: { size: 10 } } },
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8', font: { size: 10 }, callback: v => v + ' ' + unit }
                }
            }
        }
    });
}
