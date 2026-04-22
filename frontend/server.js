const path = require('path');
const PROJECT_ROOT = path.resolve(__dirname, '..');
require('dotenv').config({ path: path.join(PROJECT_ROOT, '.env') });
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs');
const http = require('http');
const OpenAI = require('openai');
const { v4: uuidv4 } = require('uuid');

// Python API URL (Flask server)
const PYTHON_API = process.env.PYTHON_API || 'http://localhost:5000';

const app = express();
const expressWs = require('express-ws')(app);

// OpenAI client
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
// Whitelist static asset directories — server.js sits in the same dir as
// index.html so serving `__dirname` would also expose server.js / package.json
// over HTTP. Only /css and /js are public; index.html is routed explicitly.
app.use('/css', express.static(path.join(__dirname, 'css')));
app.use('/js', express.static(path.join(__dirname, 'js')));
app.get('/', (_req, res) => res.sendFile(path.join(__dirname, 'index.html')));

// Upload config
const uploadDir = path.join(PROJECT_ROOT, 'uploads');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });

const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, uploadDir),
    filename: (req, file, cb) => {
        const ext = path.extname(file.originalname);
        cb(null, `eeg_${Date.now()}_${uuidv4().slice(0, 8)}${ext}`);
    }
});
const upload = multer({
    storage,
    limits: { fileSize: 100 * 1024 * 1024 },
    fileFilter: (req, file, cb) => {
        const allowed = /jpeg|jpg|png|gif|bmp|webp|tiff|edf/;
        const ext = allowed.test(path.extname(file.originalname).toLowerCase());
        const mime = allowed.test(file.mimetype);
        cb(null, ext || mime || file.originalname.endsWith('.edf'));
    }
});
const uploadEdf = multer({
    storage,
    limits: { fileSize: 500 * 1024 * 1024 },
    fileFilter: (req, file, cb) => {
        cb(null, file.originalname.toLowerCase().endsWith('.edf'));
    }
});

// Store analysis history and chat sessions
const analysisHistory = [];
const chatSessions = {};

// Serve uploaded files
app.use('/uploads', express.static(uploadDir));

// ==================== EDF PREDICT (PROXY TO PYTHON) ====================
app.post('/api/predict-edf', uploadEdf.single('edfFile'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'Không có file .edf được upload.' });
    }

    try {
        const FormData = require('form-data');
        const formData = new FormData();
        formData.append('edfFile', fs.createReadStream(req.file.path), {
            filename: req.file.originalname,
            contentType: 'application/octet-stream'
        });

        // Forward to Python API using form-data's submit (avoids Node 22 fetch conflict)
        const result = await new Promise((resolve, reject) => {
            formData.submit(`${PYTHON_API}/api/predict-edf`, (err, response) => {
                if (err) return reject(err);
                let body = '';
                response.on('data', chunk => body += chunk);
                response.on('end', () => {
                    try {
                        resolve(JSON.parse(body));
                    } catch (e) {
                        reject(new Error(`Invalid response: ${body.substring(0, 200)}`));
                    }
                });
                response.on('error', reject);
            });
        });

        // Store in history
        if (result.success) {
            const record = {
                id: uuidv4(),
                timestamp: new Date().toISOString(),
                type: 'edf',
                originalName: req.file.originalname,
                prediction: result.overall,
                analysis: result
            };
            analysisHistory.push(record);
            result.id = record.id;
        }

        res.json(result);
    } catch (error) {
        console.error('EDF predict error:', error);
        res.status(500).json({
            error: 'Không thể kết nối Python API. Hãy chắc chắn đã chạy: python python_api.py',
            details: error.message
        });
    }
});

// Python API health check
app.get('/api/model-status', async (req, res) => {
    try {
        const response = await fetch(`${PYTHON_API}/health`);
        const data = await response.json();
        res.json({ ...data, pythonApi: PYTHON_API });
    } catch {
        res.json({ status: 'offline', pythonApi: PYTHON_API });
    }
});

// ==================== EEG ANALYSIS ENDPOINT ====================
app.post('/api/analyze', upload.single('eegImage'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'Không có file ảnh được upload.' });
        }

        const imagePath = req.file.path;
        const imageBuffer = fs.readFileSync(imagePath);
        const base64Image = imageBuffer.toString('base64');
        const mimeType = req.file.mimetype || 'image/png';

        const systemPrompt = `Bạn là một chuyên gia thần kinh học (neurologist) hàng đầu với hơn 20 năm kinh nghiệm đọc và phân tích điện não đồ (EEG). 
Hãy phân tích ảnh điện não đồ được cung cấp một cách chi tiết và chuyên nghiệp.

BẮT BUỘC trả về JSON với cấu trúc sau (KHÔNG thêm markdown code blocks, CHỈ JSON thuần):
{
    "patientSummary": "Tóm tắt tổng quan về bệnh nhân dựa trên EEG",
    "overallAssessment": "Đánh giá tổng thể: Bình thường / Bất thường nhẹ / Bất thường trung bình / Bất thường nặng",
    "findings": [
        {
            "finding": "Mô tả phát hiện",
            "severity": "normal/mild/moderate/severe",
            "location": "Vùng não liên quan"
        }
    ],
    "frequencyBands": {
        "delta": { "power": 15, "status": "normal/abnormal", "note": "Ghi chú về sóng Delta (0.5-4 Hz)" },
        "theta": { "power": 20, "status": "normal/abnormal", "note": "Ghi chú về sóng Theta (4-8 Hz)" },
        "alpha": { "power": 35, "status": "normal/abnormal", "note": "Ghi chú về sóng Alpha (8-13 Hz)" },
        "beta": { "power": 22, "status": "normal/abnormal", "note": "Ghi chú về sóng Beta (13-30 Hz)" },
        "gamma": { "power": 8, "status": "normal/abnormal", "note": "Ghi chú về sóng Gamma (30-100 Hz)" }
    },
    "brainRegions": {
        "frontal": { "activity": 70, "status": "normal/abnormal", "note": "Ghi chú" },
        "temporal": { "activity": 65, "status": "normal/abnormal", "note": "Ghi chú" },
        "parietal": { "activity": 60, "status": "normal/abnormal", "note": "Ghi chú" },
        "occipital": { "activity": 75, "status": "normal/abnormal", "note": "Ghi chú" },
        "central": { "activity": 68, "status": "normal/abnormal", "note": "Ghi chú" }
    },
    "abnormalities": [
        {
            "type": "Loại bất thường",
            "description": "Mô tả chi tiết",
            "clinicalSignificance": "Ý nghĩa lâm sàng"
        }
    ],
    "recommendations": [
        "Khuyến nghị 1",
        "Khuyến nghị 2"
    ],
    "detailedAnalysis": "Phân tích chi tiết dạng văn bản dài, chuyên sâu về toàn bộ EEG",
    "clinicalCorrelation": "Liên hệ lâm sàng và các bệnh lý có thể liên quan"
}

Lưu ý: Các giá trị power/activity là phần trăm (0-100). Hãy phân tích thật chi tiết và chuyên nghiệp.`;

        const response = await openai.chat.completions.create({
            model: 'gpt-4o',
            messages: [
                { role: 'system', content: systemPrompt },
                {
                    role: 'user',
                    content: [
                        { type: 'text', text: 'Hãy phân tích chi tiết ảnh điện não đồ (EEG) này như một chuyên gia thần kinh học:' },
                        {
                            type: 'image_url',
                            image_url: {
                                url: `data:${mimeType};base64,${base64Image}`,
                                detail: 'high'
                            }
                        }
                    ]
                }
            ],
            max_tokens: 4096,
            temperature: 0.3
        });

        let analysisText = response.choices[0].message.content;
        
        // Clean markdown code blocks if present
        analysisText = analysisText.replace(/```json\s*/gi, '').replace(/```\s*/g, '').trim();

        let analysis;
        try {
            analysis = JSON.parse(analysisText);
        } catch (parseError) {
            // If JSON parsing fails, create structured response from text
            analysis = {
                patientSummary: analysisText.substring(0, 200),
                overallAssessment: "Cần đánh giá thêm",
                findings: [{ finding: analysisText, severity: "moderate", location: "Toàn bộ" }],
                frequencyBands: {
                    delta: { power: 15, status: "normal", note: "Đang phân tích" },
                    theta: { power: 20, status: "normal", note: "Đang phân tích" },
                    alpha: { power: 35, status: "normal", note: "Đang phân tích" },
                    beta: { power: 22, status: "normal", note: "Đang phân tích" },
                    gamma: { power: 8, status: "normal", note: "Đang phân tích" }
                },
                brainRegions: {
                    frontal: { activity: 70, status: "normal", note: "" },
                    temporal: { activity: 65, status: "normal", note: "" },
                    parietal: { activity: 60, status: "normal", note: "" },
                    occipital: { activity: 75, status: "normal", note: "" },
                    central: { activity: 68, status: "normal", note: "" }
                },
                abnormalities: [],
                recommendations: ["Cần phân tích lại với ảnh chất lượng cao hơn"],
                detailedAnalysis: analysisText,
                clinicalCorrelation: "Cần thêm thông tin lâm sàng để đánh giá"
            };
        }

        // Store in history
        const record = {
            id: uuidv4(),
            timestamp: new Date().toISOString(),
            imagePath: `/uploads/${req.file.filename}`,
            originalName: req.file.originalname,
            analysis
        };
        analysisHistory.push(record);

        res.json({
            success: true,
            id: record.id,
            imagePath: record.imagePath,
            analysis
        });

    } catch (error) {
        console.error('Analysis error:', error);
        res.status(500).json({ 
            error: 'Lỗi phân tích ảnh điện não.', 
            details: error.message 
        });
    }
});

// ==================== CHAT ENDPOINT ====================
app.post('/api/chat', async (req, res) => {
    try {
        const { message, sessionId, analysisContext } = req.body;
        
        if (!message) {
            return res.status(400).json({ error: 'Tin nhắn không được để trống.' });
        }

        const sid = sessionId || uuidv4();
        if (!chatSessions[sid]) {
            chatSessions[sid] = {
                messages: [
                    {
                        role: 'system',
                        content: `Bạn là một chuyên gia thần kinh học AI, chuyên phân tích và tư vấn về điện não đồ (EEG). 
Bạn có kiến thức sâu rộng về:
- Đọc và phân tích EEG (các sóng Delta, Theta, Alpha, Beta, Gamma)
- Các bệnh lý thần kinh (động kinh, tổn thương não, rối loạn giấc ngủ, v.v.)
- Các kỹ thuật ghi điện não và artifacts
- Lâm sàng thần kinh học

Hãy trả lời bằng tiếng Việt, chuyên nghiệp nhưng dễ hiểu. Sử dụng emoji phù hợp.
Nếu được cung cấp context phân tích EEG, hãy sử dụng thông tin đó để trả lời chính xác hơn.`
                    }
                ],
                analysisContext: null
            };
        }

        // Update analysis context if provided
        if (analysisContext) {
            chatSessions[sid].analysisContext = analysisContext;
            chatSessions[sid].messages.push({
                role: 'system',
                content: `Context phân tích EEG mới nhất: ${JSON.stringify(analysisContext)}`
            });
        }

        chatSessions[sid].messages.push({ role: 'user', content: message });

        // Keep conversation manageable (last 20 messages + system prompt)
        const msgs = chatSessions[sid].messages;
        const systemMsgs = msgs.filter(m => m.role === 'system');
        const convMsgs = msgs.filter(m => m.role !== 'system').slice(-20);
        const truncated = [...systemMsgs, ...convMsgs];

        const response = await openai.chat.completions.create({
            model: 'gpt-4o',
            messages: truncated,
            max_tokens: 2048,
            temperature: 0.7
        });

        const reply = response.choices[0].message.content;
        chatSessions[sid].messages.push({ role: 'assistant', content: reply });

        res.json({
            success: true,
            sessionId: sid,
            reply
        });

    } catch (error) {
        console.error('Chat error:', error);
        res.status(500).json({ 
            error: 'Lỗi chatbot.', 
            details: error.message 
        });
    }
});

// ==================== MCP SERVER ENDPOINTS ====================
// MCP Status
app.get('/api/mcp/status', (req, res) => {
    res.json({
        status: 'active',
        serverName: 'EEG Analysis MCP Server',
        version: '1.0.0',
        tools: [
            {
                name: 'analyze_eeg',
                description: 'Phân tích ảnh điện não đồ',
                inputSchema: {
                    type: 'object',
                    properties: {
                        imageId: { type: 'string', description: 'ID của ảnh đã upload' }
                    }
                }
            },
            {
                name: 'generate_report',
                description: 'Tạo báo cáo phân tích EEG',
                inputSchema: {
                    type: 'object',
                    properties: {
                        analysisId: { type: 'string', description: 'ID của phân tích' }
                    }
                }
            },
            {
                name: 'get_analysis_history',
                description: 'Lấy lịch sử phân tích',
                inputSchema: {
                    type: 'object',
                    properties: {
                        limit: { type: 'number', description: 'Số lượng kết quả' }
                    }
                }
            }
        ],
        uptime: process.uptime(),
        analysisCount: analysisHistory.length
    });
});

// MCP Tool Execution
app.post('/api/mcp/execute', async (req, res) => {
    try {
        const { tool, params } = req.body;

        switch (tool) {
            case 'analyze_eeg': {
                const record = analysisHistory.find(a => a.id === params.imageId);
                if (!record) return res.status(404).json({ error: 'Không tìm thấy bản phân tích.' });
                res.json({ success: true, result: record });
                break;
            }
            case 'generate_report': {
                const record = analysisHistory.find(a => a.id === params.analysisId);
                if (!record) return res.status(404).json({ error: 'Không tìm thấy bản phân tích.' });

                const reportPrompt = `Dựa trên kết quả phân tích EEG sau, hãy viết một báo cáo y khoa chuyên nghiệp bằng tiếng Việt:
${JSON.stringify(record.analysis, null, 2)}

Báo cáo cần bao gồm:
1. Thông tin chung
2. Kỹ thuật ghi EEG
3. Mô tả chi tiết các sóng não
4. Phát hiện bất thường
5. Kết luận và khuyến nghị`;

                const response = await openai.chat.completions.create({
                    model: 'gpt-4o',
                    messages: [
                        { role: 'system', content: 'Bạn là bác sĩ thần kinh viết báo cáo y khoa.' },
                        { role: 'user', content: reportPrompt }
                    ],
                    max_tokens: 3000
                });

                res.json({
                    success: true,
                    report: response.choices[0].message.content,
                    generatedAt: new Date().toISOString()
                });
                break;
            }
            case 'get_analysis_history': {
                const limit = params?.limit || 10;
                res.json({
                    success: true,
                    history: analysisHistory.slice(-limit).reverse()
                });
                break;
            }
            default:
                res.status(400).json({ error: `Tool không hợp lệ: ${tool}` });
        }
    } catch (error) {
        console.error('MCP error:', error);
        res.status(500).json({ error: 'Lỗi thực thi MCP tool.', details: error.message });
    }
});

// Analysis history
app.get('/api/history', (req, res) => {
    res.json({
        success: true,
        history: analysisHistory.slice().reverse()
    });
});

// ==================== WEBSOCKET CHAT ====================
app.ws('/ws/chat', (ws, req) => {
    const sid = uuidv4();
    chatSessions[sid] = {
        messages: [
            {
                role: 'system',
                content: `Bạn là chuyên gia thần kinh học AI tư vấn về EEG. Trả lời bằng tiếng Việt, chuyên nghiệp và dễ hiểu. Sử dụng emoji phù hợp.`
            }
        ]
    };

    ws.send(JSON.stringify({ type: 'connected', sessionId: sid }));

    ws.on('message', async (msg) => {
        try {
            const data = JSON.parse(msg);
            chatSessions[sid].messages.push({ role: 'user', content: data.message });

            const response = await openai.chat.completions.create({
                model: 'gpt-4o',
                messages: chatSessions[sid].messages.slice(-22),
                max_tokens: 2048,
                temperature: 0.7
            });

            const reply = response.choices[0].message.content;
            chatSessions[sid].messages.push({ role: 'assistant', content: reply });

            ws.send(JSON.stringify({ type: 'reply', content: reply }));
        } catch (error) {
            ws.send(JSON.stringify({ type: 'error', content: 'Lỗi xử lý tin nhắn.' }));
        }
    });

    ws.on('close', () => {
        delete chatSessions[sid];
    });
});

// SPA fallback
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`🧠 EEG Analysis Server running on http://localhost:${PORT}`);
    console.log(`📡 MCP Server active`);
    console.log(`🔌 WebSocket ready on ws://localhost:${PORT}/ws/chat`);
});
