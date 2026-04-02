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
    
    const isNormal = data.overall === 'BÌNH THƯỜNG';
    document.getElementById('predictionIcon').textContent = isNormal ? '✅' : '🔴';
    document.getElementById('predictionLabel').textContent = data.overall;
    document.getElementById('predictionLabel').style.color = isNormal ? '#10b981' : '#ef4444';
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
    badge.className = `assessment-badge ${isNormal ? 'normal' : data.severity}`;
    badge.textContent = data.overall;
    
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
    
    // Window results as findings
    let findingsHtml = '';
    const abnormalWindows = (data.windowResults || []).filter(w => w.prediction === 'Bất thường');
    if (abnormalWindows.length > 0) {
        abnormalWindows.slice(0, 10).forEach(w => {
            findingsHtml += `
                <div class="finding-item">
                    <div class="finding-header">
                        <span class="finding-title">⚠️ Bất thường tại ${w.startSec}s - ${w.endSec}s</span>
                        <span class="severity-tag severe">${w.probability}%</span>
                    </div>
                    <div class="finding-location">📊 Xác suất bất thường: ${w.probability}%</div>
                </div>`;
        });
    } else {
        findingsHtml = '<p style="color:var(--text-muted)">Không phát hiện đoạn bất thường nào.</p>';
    }
    document.getElementById('findingsList').innerHTML = findingsHtml;
    
    document.getElementById('abnormalitiesList').innerHTML = isNormal 
        ? '<p style="color:var(--text-muted)">Không phát hiện bất thường.</p>'
        : `<div class="finding-item"><div class="finding-header"><span class="finding-title">Phát hiện ${data.abnormalWindows} đoạn EEG có dấu hiệu bất thường</span></div><p style="font-size:13px; color:var(--text-secondary); margin-top:4px;">Tỉ lệ: ${data.abnormalRatio}% windows. Cần đối chiếu lâm sàng.</p></div>`;
    
    document.getElementById('recommendationsList').innerHTML = `
        <div class="recommendation-item"><span class="num">1</span><span>Đối chiếu kết quả với triệu chứng lâm sàng</span></div>
        <div class="recommendation-item"><span class="num">2</span><span>${isNormal ? 'EEG trong giới hạn bình thường, theo dõi định kỳ' : 'Cần tham khảo ý kiến chuyên gia thần kinh'}</span></div>
        <div class="recommendation-item"><span class="num">3</span><span>Sử dụng chatbot để được giải thích chi tiết kết quả</span></div>`;
    
    document.getElementById('detailedText').textContent = 
        `File EDF: ${selectedFile.name}, Thời lượng: ${data.duration}s, ` +
        `Kênh: ${(data.channelNames || []).join(', ')}, ` +
        `Tần số lấy mẫu: ${data.samplingRate}Hz. ` +
        `Model ${data.modelInfo?.name} sử dụng ${data.modelInfo?.features} đặc trưng qEEG ` +
        `với threshold ${data.threshold} để phân loại.`;
    
    document.getElementById('clinicalText').textContent = 
        isNormal 
            ? 'Không phát hiện dấu hiệu bất thường trên EEG. Hoạt động não trong giới hạn bình thường.'
            : 'Phát hiện các đoạn EEG bất thường có thể liên quan đến hoạt động co giật (seizure). Cần đối chiếu với tiền sử bệnh và khám lâm sàng.';
    
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
        const chColors = ['#3b82f6', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6', '#06b6d4'];
        
        // Create datasets - offset each channel vertically
        const datasets = [];
        channelNames.forEach((chName, idx) => {
            const rawData = waveform.channels[chName];
            // Normalize and offset each channel
            const mean = rawData.reduce((a,b) => a+b, 0) / rawData.length;
            const std = Math.sqrt(rawData.reduce((a,b) => a + (b-mean)**2, 0) / rawData.length) || 1;
            const normalized = rawData.map(v => ((v - mean) / std) + (channelNames.length - idx) * 4);
            
            datasets.push({
                label: chName,
                data: normalized,
                borderColor: chColors[idx % chColors.length],
                borderWidth: 1,
                pointRadius: 0,
                fill: false,
                tension: 0.1
            });
        });
        
        // Create annotation boxes for abnormal regions
        const annotations = {};
        (data.abnormalRegions || []).forEach((region, idx) => {
            const duration = data.duration || 3600;
            const totalPoints = waveform.time.length;
            const xMin = (region.start / duration) * totalPoints;
            const xMax = (region.end / duration) * totalPoints;
            
            annotations[`box${idx}`] = {
                type: 'box',
                xMin: Math.floor(xMin),
                xMax: Math.ceil(xMax),
                backgroundColor: `rgba(239, 68, 68, ${Math.min(region.prob / 200, 0.4)})`,
                borderColor: 'rgba(239, 68, 68, 0.6)',
                borderWidth: 1,
                label: {
                    display: region.prob > 80,
                    content: `${region.prob}%`,
                    position: 'start',
                    font: { size: 9, weight: 'bold' },
                    color: '#fff'
                }
            };
        });
        
        // Time labels (show every Nth)
        const timeLabels = waveform.time.map((t, i) => {
            const mins = Math.floor(t / 60);
            const secs = Math.floor(t % 60);
            return `${mins}:${String(secs).padStart(2, '0')}`;
        });
        
        charts.waveform = new Chart(document.getElementById('eegWaveformChart'), {
            type: 'line',
            data: { labels: timeLabels, datasets },
            options: {
                ...commonOpts,
                animation: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { 
                        position: 'top', 
                        labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 }, boxWidth: 12, padding: 8 } 
                    },
                    annotation: { annotations },
                    tooltip: {
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
                        title: { display: true, text: 'Kênh EEG (chuẩn hóa)', color: '#94a3b8', font: { size: 12 } }
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
        const winColors = winProbs.map(p => p >= (data.threshold || 0.5) * 100 ? 'rgba(239,68,68,0.8)' : 'rgba(16,185,129,0.5)');
        
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
