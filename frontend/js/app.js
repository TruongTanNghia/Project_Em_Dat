// ===== GLOBAL STATE =====
let selectedFile = null;
let currentAnalysis = null;
let chatSessionId = null;
let charts = {};
let uploadMode = 'edf'; // 'edf' or 'image'

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

        const response = await fetch('/api/predict-edf', {
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

        const response = await fetch('/api/analyze', {
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

        const response = await fetch('/api/chat', {
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
        const response = await fetch('/api/mcp/status');
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
                const histRes = await fetch('/api/history');
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

        const response = await fetch('/api/mcp/execute', {
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
        const response = await fetch('/api/model-status');
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
    abnormalRegions.forEach(region => {
        const xStart = (region.start / duration) * 50;
        const xEnd = (region.end / duration) * 50;
        const xMid = (xStart + xEnd) / 2;

        const sphereGeo = new THREE.SphereGeometry(0.8, 16, 16);
        const sphereMat = new THREE.MeshPhongMaterial({
            color: 0xff4444,
            emissive: 0xff2222,
            emissiveIntensity: 0.8,
            transparent: true,
            opacity: 0.7
        });
        const sphere = new THREE.Mesh(sphereGeo, sphereMat);
        sphere.position.set(xMid, 16, (nCh - 1) * 3);
        sphere.userData.isEEG = true;
        sphere.userData.isAbnormalMarker = true;
        scene3d.add(sphere);
        glowSpheres3d.push(sphere);
        abnormalMarkers3d.push(sphere);
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
    showToast(eeg3dAbnormalVisible ? '🔴 Hiển thị vùng bất thường' : '⚪ Ẩn vùng bất thường', 'info');
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
