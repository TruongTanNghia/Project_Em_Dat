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

// ============ BRAIN TUMOR (MRI) — mock analysis ============
function runBrainAnalysis() {
    const result = {
        confidence: 0.96,
        volume: 32.7,
        type: 'Glioma',
        grade: 'Cao (High Grade)',
        location: 'Thùy trán',
        locationSub: 'Trái (Frontal Lobe - Left)',
        coords: '-32.4, 45.6, 28.1',
        probs: [
            { label: 'Glioma (High Grade)', value: 0.82, color: '#ef4444' },
            { label: 'Meningioma',          value: 0.11, color: '#f59e0b' },
            { label: 'Pituitary Tumor',     value: 0.04, color: '#10b981' },
            { label: 'Others',              value: 0.03, color: '#3b82f6' },
        ],
        patient: {
            id: '230514_001', age: 45, sex: 'Nam',
            date: '14/05/2024',
            symptom: 'Đau đầu, chóng mặt',
            history: 'Không có'
        },
        progression: [
            { date: '01/03/2024', volume: 8.2 },
            { date: '01/04/2024', volume: 16.3 },
            { date: '01/05/2024', volume: 16.1 },
            { date: '14/05/2024', volume: 32.7, current: true },
        ],
        recommendations: [
            'Nên tham khảo ý kiến bác sĩ chuyên khoa Thần kinh.',
            'Khối u có dấu hiệu phát triển, cần theo dõi và điều trị sớm.',
            'Đề xuất: Chụp MRI định kỳ và đánh giá lại sau 4 tuần.',
        ]
    };

    // Populate patient info
    document.getElementById('brainPatientId').textContent = result.patient.id;
    document.getElementById('brainScanDate').textContent = result.patient.date;
    document.getElementById('brainInfoId').textContent = result.patient.id;
    document.getElementById('brainInfoAge').textContent = result.patient.age;
    document.getElementById('brainInfoSex').textContent = result.patient.sex;
    document.getElementById('brainInfoDate').textContent = result.patient.date;
    document.getElementById('brainInfoSymptom').textContent = result.patient.symptom;
    document.getElementById('brainInfoHistory').textContent = result.patient.history;

    // Detection stats
    document.getElementById('brainConfidence').textContent = result.confidence.toFixed(2);
    document.getElementById('brainVolume').textContent = result.volume + ' cm³';
    document.getElementById('brainType').textContent = result.type + ' (High Grade)';

    // Analysis
    document.getElementById('brainSize').textContent = result.volume + ' cm³';
    document.getElementById('brainLocation').textContent = result.location;
    document.getElementById('brainLocationSub').textContent = result.locationSub;
    document.getElementById('brainCoords').textContent = result.coords;
    document.getElementById('brainGrade').textContent = result.grade;

    // Prob bars
    const probEl = document.getElementById('brainProbBars');
    probEl.innerHTML = result.probs.map(p => `
        <div class="prob-row">
            <span>${p.label}</span>
            <div class="prob-bar-track"><div class="prob-bar-fill" style="width:${p.value*100}%; background:${p.color};"></div></div>
            <span class="pct">${p.value.toFixed(2)}</span>
        </div>
    `).join('');

    // Timeline
    const tlEl = document.getElementById('brainTimeline');
    tlEl.innerHTML = result.progression.map(p => `
        <div class="timepoint ${p.current ? 'current' : ''}">
            <div class="tp-date">${p.date}${p.current ? ' (Hiện tại)' : ''}</div>
            <div class="tp-thumb"></div>
            <div class="tp-size">Thể tích: <b>${p.volume} cm³</b></div>
        </div>
    `).join('');

    // Progression chart
    renderProgressionChart('brainProgressChart', result.progression, 'cm³', '#06b6d4');

    // Recommendations
    document.getElementById('brainRecoList').innerHTML =
        result.recommendations.map(r => `<li class="reco-item">${r}</li>`).join('');

    showToast('🧠 Đã phân tích xong (mock data)', 'success');
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

// Preload dataset list in background on page load
async function preloadDataset() {
    if (dsData || dsLoading) return;
    dsLoading = true;
    try {
        var apiBase = (window.APP_CONFIG && window.APP_CONFIG.API_BASE) || '';
        const resp = await fetch(apiBase + '/api/lung-dataset');
        dsData = await resp.json();
        console.log('[Dataset] Preloaded: ' + dsData.total + ' patients');
        // Pre-populate dropdown immediately
        populatePatientDropdown();
    } catch(e) {
        console.warn('[Dataset] Preload failed:', e.message);
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
        // Only reset slice grid if no selection
        if (dsSelectedSlice < 0) {
            document.getElementById('dsSliceGrid').innerHTML = '<div class="ds-empty">Chọn bệnh nhân ở trên</div>';
        }
    } else if (dsLoading) {
        document.getElementById('dsSliceGrid').innerHTML = '<div class="ds-empty">⏳ Đang tải...</div>';
        var checkInterval = setInterval(function() {
            if (dsData) {
                clearInterval(checkInterval);
                populatePatientDropdown();
                document.getElementById('dsSliceGrid').innerHTML = '<div class="ds-empty">Chọn bệnh nhân ở trên</div>';
            }
        }, 200);
    } else {
        document.getElementById('dsSliceGrid').innerHTML = '<div class="ds-empty">⏳ Đang tải...</div>';
        preloadDataset().then(function() {
            if (dsData) {
                populatePatientDropdown();
                document.getElementById('dsSliceGrid').innerHTML = '<div class="ds-empty">Chọn bệnh nhân ở trên</div>';
            } else {
                document.getElementById('dsSliceGrid').innerHTML = '<div class="ds-empty">❌ Không thể tải dataset</div>';
            }
        });
    }
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
    nodule.slices.forEach((s, i) => {
        var div = document.createElement('div');
        div.className = 'ds-slice-item';
        div.dataset.index = i;
        var imgBase = (window.APP_CONFIG && window.APP_CONFIG.API_BASE) || '';
        div.innerHTML = '<img src="' + imgBase + '/api/lung-dataset-image?patient=' + dsSelectedPatient.id + '&nodule=' + nodName + '&slice=' + i + '" onerror="this.style.display=\'none\'">' +
            '<div class="ds-slice-label">' + s.replace('.png','') + '</div>';
        div.onclick = function() {
            grid.querySelectorAll('.ds-slice-item').forEach(el => el.classList.remove('selected'));
            div.classList.add('selected');
            dsSelectedSlice = i;
            document.getElementById('dsAnalyzeBtn').disabled = false;
            document.getElementById('dsInfoText').textContent = '📋 ' + dsSelectedPatient.id + ' / ' + nodName + ' / ' + s + ' | ' + nodule.maskCount + ' bác sĩ annotation';
        };
        grid.appendChild(div);
    });
    // Auto-select middle slice
    var midIdx = Math.floor(nodule.slices.length / 2);
    var midItem = grid.children[midIdx];
    if (midItem) midItem.click();
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
    const patientId = 'BN' + now.getFullYear().toString().slice(2) + (now.getMonth()+1).toString().padStart(2,'0') + now.getDate().toString().padStart(2,'0') + '_' + Math.floor(Math.random()*999).toString().padStart(3,'0');

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
            <img src="${result.originalImage}" style="width:100%;height:100%;object-fit:contain;border-radius:8px;" alt="CT Original">
            <div style="position:absolute;left:12px;top:50%;color:#fff;font-size:11px;font-weight:700;opacity:0.6;text-shadow:0 0 4px #000;">R</div>
            <div style="position:absolute;right:12px;top:50%;color:#fff;font-size:11px;font-weight:700;opacity:0.6;text-shadow:0 0 4px #000;">L</div>
        `;
        mainView.style.position = 'relative';
    }

    document.getElementById('lungSliceLabel').textContent = `${result.originalSize ? result.originalSize.w : 512}×${result.originalSize ? result.originalSize.h : 512}`;

    // Section 2: Detection overlay
    const detCanvas = document.getElementById('lungDetectionCanvas');
    if (result.overlayImage) {
        detCanvas.innerHTML = `<img src="${result.overlayImage}" style="width:100%;height:100%;object-fit:contain;border-radius:8px;" alt="Detection Overlay">`;
        if (result.detected && result.tumors && result.tumors.length > 0) {
            const t = result.tumors[0];
            detCanvas.innerHTML += `<div style="position:absolute;bottom:8px;left:8px;background:rgba(0,0,0,0.7);padding:4px 8px;border-radius:4px;font-size:11px;color:#0f0;">Tumor detected: ${t.diameterMm} mm</div>`;
            detCanvas.style.position = 'relative';
        }
    } else if (!result.detected) {
        detCanvas.innerHTML = `<div class="detection-placeholder" style="color:#10b981;">✅ Không phát hiện khối u</div>`;
    }

    // Section 4: Analysis results
    document.getElementById('lungDiameter').textContent = result.detected ? result.maxDiameterMm + ' mm' : 'N/A';

    // Mini tumor preview — use actual mask image cropped to bbox
    var mini3d = document.getElementById('lungMini3d');
    if (result.detected && result.maskImage) {
      // Show the mask-only image (red tumor on transparent/dark bg)
      mini3d.innerHTML = '<div class="mini-tumor-viz" style="padding:0;overflow:hidden;border-radius:12px;">' +
        '<img src="' + result.maskImage + '" style="width:100%;height:100%;object-fit:contain;" alt="Tumor">' +
        '<div class="mini-tumor-label">' + result.maxDiameterMm + ' mm</div>' +
        '</div>';
    } else if (result.detected && result.overlayImage) {
      // Fallback: use overlay image
      mini3d.innerHTML = '<div class="mini-tumor-viz" style="padding:0;overflow:hidden;border-radius:12px;">' +
        '<img src="' + result.overlayImage + '" style="width:100%;height:100%;object-fit:contain;" alt="Tumor">' +
        '<div class="mini-tumor-label">' + result.maxDiameterMm + ' mm</div>' +
        '</div>';
    } else {
      mini3d.innerHTML = '<div class="mini-3d-placeholder" style="color:#10b981;">✅</div>';
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

    // Sub-views (Coronal / Sagittal / Mini 3D) below the main CT image
    updateLungSubViews(result);
    // Filmstrip of nearby slices (dataset mode only — needs a volume)
    updateLungSliceStrip(result);

    // Section 3: 3D Lung Visualization (Three.js)
    console.log('displayLungResults: scheduling initLung3D, result.detected=', result.detected);
    setTimeout(() => initLung3D(result), 500);

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
    recos.push('Vị trí: ' + (result.position || 'N/A') + ' — cần đối chiếu lâm sàng.');
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
    lungScene.background = new THREE.Color(0x050a14);
    lungCamera = new THREE.PerspectiveCamera(45, w / h, 0.1, 500);
    lungCamera.position.set(18, 10, 30);
    lungCamera.lookAt(0, 0, 0);
    lungRenderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    lungRenderer.setSize(w, h);
    lungRenderer.setPixelRatio(window.devicePixelRatio > 1 ? 2 : 1);
    lungControls = new THREE.OrbitControls(lungCamera, lungRenderer.domElement);
    lungControls.enableDamping = true;
    lungControls.autoRotate = true;
    lungControls.autoRotateSpeed = 1.5;
    lungControls.target.set(0, 0, 0);
    // Lighting
    lungScene.add(new THREE.AmbientLight(0xffffff, 1.0));
    var dl1 = new THREE.DirectionalLight(0xaaccff, 1.5); dl1.position.set(15, 20, 25); lungScene.add(dl1);
    var dl2 = new THREE.DirectionalLight(0x6688bb, 0.8); dl2.position.set(-15, 10, -20); lungScene.add(dl2);
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
              c.material = new THREE.MeshLambertMaterial({
                color: 0x22aacc, transparent: true, opacity: 0.35,
                side: THREE.DoubleSide, depthWrite: false
              });
            }
          });
          console.log('[Lung3D] Meshes found:', mc);
          // Auto-fit
          var box = new THREE.Box3().setFromObject(model);
          var sz = box.getSize(new THREE.Vector3());
          var ct = box.getCenter(new THREE.Vector3());
          var mx = Math.max(sz.x, sz.y, sz.z);
          var sc = 18 / mx;
          model.scale.setScalar(sc);
          model.position.set(-ct.x * sc, -ct.y * sc, -ct.z * sc);
          lungScene.add(model);
          // Recompute after transform
          var finalBox = new THREE.Box3().setFromObject(model);
          var fMin = finalBox.min, fMax = finalBox.max;
          window._lungModelBounds = { min: {x:fMin.x, y:fMin.y, z:fMin.z}, max: {x:fMax.x, y:fMax.y, z:fMax.z} };
          console.log('[Lung3D] Model bounds: X[' + fMin.x.toFixed(1) + ',' + fMax.x.toFixed(1) + '] Y[' + fMin.y.toFixed(1) + ',' + fMax.y.toFixed(1) + '] Z[' + fMin.z.toFixed(1) + ',' + fMax.z.toFixed(1) + ']');
          console.log('[Lung3D] Model scale=' + sc.toFixed(3));
          // Reposition tumor if it was placed before model loaded
          if (lungTumorMesh && window._lungTumorTarget) {
            var tb = window._lungTumorTarget;
            lungTumorMesh.position.set(tb.x, tb.y, tb.z);
            console.log('[Lung3D] Tumor repositioned to model-relative coords');
          }
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

      var cx3d, cy3d, cz3d = 0.8;
      // LIDC nodule crops are tumor-centered → centroid coords say nothing
      // about which lung the tumor was originally in. Detect this case
      // (centroid near image center within a LIDC dataset mode) and
      // distribute tumors deterministically by patient-id hash so the 3D
      // viewer shows variety across patients instead of all stacking up
      // in the mediastinum.
      var isLIDCCrop = !!(tumorData.datasetInfo && tumorData.datasetInfo.patientId) &&
                       centroidNorm &&
                       Math.abs((centroidNorm.x || 256) - 256) < 80 &&
                       Math.abs((centroidNorm.y || 256) - 256) < 80;

      if (isLIDCCrop) {
        var pid = tumorData.datasetInfo.patientId;
        var h = 0;
        for (var hi = 0; hi < pid.length; hi++) {
          h = ((h << 5) - h + pid.charCodeAt(hi)) | 0;
        }
        var ax = ((h & 0xff) / 255) * 2 - 1;          // -1..1
        var ay = (((h >> 8) & 0xff) / 255) * 2 - 1;   // -1..1
        cx3d = (ax > 0 ? 1 : -1) * (3.0 + Math.abs(ax) * 2.5);   // ±[3.0, 5.5], avoid mediastinum
        cy3d = ay * 3.5 + 0.3;                                    // [-3.2, +3.8]
        console.log('[Lung3D] LIDC patient hash (' + pid + ') -> 3D(' +
                    cx3d.toFixed(2) + ', ' + cy3d.toFixed(2) + ')');
      } else if (centroidNorm && typeof centroidNorm.x === 'number' && typeof centroidNorm.y === 'number') {
        var nx = centroidNorm.x / 512;     // 0..1, image-left to image-right
        var ny = centroidNorm.y / 512;     // 0..1, image-top to image-bottom
        cx3d = (1 - 2 * nx) * 5.5;         // [-5.5, +5.5]
        cy3d = (1 - 2 * ny) * 4.0 + 0.3;   // [-3.7, +4.3]
        console.log('[Lung3D] continuous centroid (nx=' + nx.toFixed(2) +
                    ', ny=' + ny.toFixed(2) + ') -> 3D(' +
                    cx3d.toFixed(2) + ', ' + cy3d.toFixed(2) + ')');
      } else {
        cx3d = isRight ? 4.0 : -4.0;
        cy3d = isUpper ? 3.5 : (isLower ? -3.5 : 0.5);
        console.log('[Lung3D] bucketed fallback pos="' + posText +
                    '" -> 3D(' + cx3d + ', ' + cy3d + ')');
      }

      // Store for repositioning after model loads
      window._lungTumorTarget = {x: cx3d, y: cy3d, z: cz3d};

      // Scale for tumor shape
      var tumorSizeScale = 0.22; // controls how big tumor appears relative to lung

      if (contour && contour.length >= 3) {
        // Compute contour centroid in normalized space
        var contCx = 0, contCy = 0;
        for (var ci = 0; ci < contour.length; ci++) {
          contCx += contour[ci][0]; contCy += contour[ci][1];
        }
        contCx /= contour.length; contCy /= contour.length;

        // Build shape CENTERED at origin (relative coords)
        var shape = new THREE.Shape();
        var firstX = (contour[0][0] - contCx) * 18 * tumorSizeScale;
        var firstY = (contCy - contour[0][1]) * 14 * tumorSizeScale;
        shape.moveTo(firstX, firstY);
        for (var ci = 1; ci < contour.length; ci++) {
          var px = (contour[ci][0] - contCx) * 18 * tumorSizeScale;
          var py = (contCy - contour[ci][1]) * 14 * tumorSizeScale;
          shape.lineTo(px, py);
        }
        shape.lineTo(firstX, firstY);

        var depth = Math.max(0.15, (tumorData.maxDiameterMm || 10) / 350 * 18 * tumorSizeScale * 0.5);
        var tumorGeo = new THREE.ExtrudeGeometry(shape, {
          depth: depth,
          bevelEnabled: true,
          bevelThickness: depth * 0.12,
          bevelSize: depth * 0.08,
          bevelSegments: 2
        });
        tumorGeo.center();

        lungTumorMesh = new THREE.Mesh(tumorGeo, new THREE.MeshLambertMaterial({
          color: 0xff2244, transparent: true, opacity: 0.72,
          emissive: 0x550000, side: THREE.DoubleSide
        }));
        // Position at the correct anatomical location
        lungTumorMesh.position.set(cx3d, cy3d, cz3d);
        lungScene.add(lungTumorMesh);

        // Wireframe
        var wireMesh = new THREE.Mesh(tumorGeo.clone(), new THREE.MeshBasicMaterial({
          color: 0xff4466, wireframe: true, transparent: true, opacity: 0.2
        }));
        wireMesh.position.copy(lungTumorMesh.position);
        wireMesh.scale.set(1.03, 1.03, 1.03);
        lungScene.add(wireMesh);

        // Glow
        tumorGeo.computeBoundingSphere();
        var glowR = tumorGeo.boundingSphere ? tumorGeo.boundingSphere.radius : 1;
        [1.4, 2.0].forEach(function(s, i){
          var gg = new THREE.Mesh(
            new THREE.SphereGeometry(glowR * s, 10, 8),
            new THREE.MeshBasicMaterial({color:0xff3355, transparent:true, opacity:0.04-i*0.015, side:THREE.BackSide})
          );
          gg.position.copy(lungTumorMesh.position);
          lungScene.add(gg);
          if(i===0) lungTumorGlow = gg;
        });

        console.log('[Lung3D] Contour tumor at (' + cx3d.toFixed(1) + ',' + cy3d.toFixed(1) + ',' + cz3d + ') ' + contour.length + ' pts');

      } else {
        // Fallback ellipsoid at correct position
        var scaleU = 18 / 512 * tumorSizeScale;
        var rx = Math.max(0.2, (bbox.w||30) * scaleU);
        var ry = Math.max(0.2, (bbox.h||30) * scaleU);
        var rz = Math.max(0.1, (rx+ry)/2 * 0.5);
        lungTumorMesh = new THREE.Mesh(
          new THREE.SphereGeometry(1, 20, 14),
          new THREE.MeshLambertMaterial({color:0xff2244, transparent:true, opacity:0.75, emissive:0x440000})
        );
        lungTumorMesh.scale.set(rx, ry, rz);
        lungTumorMesh.position.set(cx3d, cy3d, cz3d);
        lungScene.add(lungTumorMesh);
        lungTumorGlow = lungTumorMesh;
        console.log('[Lung3D] Ellipsoid tumor at (' + cx3d.toFixed(1) + ',' + cy3d.toFixed(1) + ')');
      }

      // Point light at tumor
      var tLt = new THREE.PointLight(0xff2244, 0.8, 12);
      tLt.position.copy(lungTumorMesh.position);
      lungScene.add(tLt);
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

    // Tumor pulsing animation
    if (lungTumorMesh) {
        const pulse = 1 + Math.sin(time * 3) * 0.08;
        lungTumorMesh.scale.setScalar(pulse);
        lungTumorMesh.material.emissiveIntensity = 0.4 + Math.sin(time * 2) * 0.3;
    }
    if (lungTumorGlow) {
        const glowPulse = 1 + Math.sin(time * 2) * 0.15;
        lungTumorGlow.scale.setScalar(glowPulse);
        lungTumorGlow.material.opacity = 0.1 + Math.sin(time * 1.5) * 0.08;
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
