/**
 * MCP (Model Context Protocol) Server for EEG Analysis
 * Standalone MCP server that can be connected to by MCP clients
 */
const readline = require('readline');
const fs = require('fs');
const path = require('path');

const SERVER_INFO = {
    name: 'eeg-analysis-mcp',
    version: '1.0.0',
    description: 'MCP Server for EEG Brain Wave Analysis'
};

const TOOLS = [
    {
        name: 'analyze_eeg',
        description: 'Phân tích ảnh điện não đồ EEG sử dụng AI',
        inputSchema: {
            type: 'object',
            properties: {
                imagePath: {
                    type: 'string',
                    description: 'Đường dẫn tới file ảnh EEG'
                }
            },
            required: ['imagePath']
        }
    },
    {
        name: 'generate_report',
        description: 'Tạo báo cáo phân tích EEG chi tiết',
        inputSchema: {
            type: 'object',
            properties: {
                analysisId: {
                    type: 'string',
                    description: 'ID của bản phân tích'
                },
                format: {
                    type: 'string',
                    enum: ['text', 'html', 'markdown'],
                    description: 'Định dạng báo cáo'
                }
            },
            required: ['analysisId']
        }
    },
    {
        name: 'get_analysis_history',
        description: 'Lấy danh sách lịch sử phân tích EEG',
        inputSchema: {
            type: 'object',
            properties: {
                limit: {
                    type: 'number',
                    description: 'Số lượng kết quả tối đa'
                }
            }
        }
    },
    {
        name: 'compare_analyses',
        description: 'So sánh hai hoặc nhiều bản phân tích EEG',
        inputSchema: {
            type: 'object',
            properties: {
                analysisIds: {
                    type: 'array',
                    items: { type: 'string' },
                    description: 'Danh sách ID các bản phân tích cần so sánh'
                }
            },
            required: ['analysisIds']
        }
    }
];

const RESOURCES = [
    {
        uri: 'eeg://knowledge/frequency-bands',
        name: 'EEG Frequency Bands Reference',
        description: 'Tài liệu tham khảo về các dải tần số sóng não',
        mimeType: 'application/json'
    },
    {
        uri: 'eeg://knowledge/abnormalities',
        name: 'Common EEG Abnormalities',
        description: 'Danh sách các bất thường EEG phổ biến',
        mimeType: 'application/json'
    }
];

// Knowledge base data
const FREQUENCY_BANDS = {
    delta: { range: '0.5-4 Hz', normalState: 'Giấc ngủ sâu', abnormal: 'Tổn thương não, viêm não' },
    theta: { range: '4-8 Hz', normalState: 'Buồn ngủ, thiền', abnormal: 'Rối loạn chú ý, tổn thương' },
    alpha: { range: '8-13 Hz', normalState: 'Thư giãn, nhắm mắt', abnormal: 'Lo âu, mất ngủ' },
    beta: { range: '13-30 Hz', normalState: 'Tập trung, tư duy', abnormal: 'Căng thẳng, lo âu' },
    gamma: { range: '30-100 Hz', normalState: 'Xử lý nhận thức cao', abnormal: 'Động kinh' }
};

const ABNORMALITIES = [
    { name: 'Spike waves', description: 'Sóng nhọn - liên quan đến động kinh', severity: 'high' },
    { name: 'Sharp waves', description: 'Sóng sắc - bất thường kịch phát', severity: 'high' },
    { name: 'Slow waves', description: 'Sóng chậm lan tỏa - tổn thương não', severity: 'moderate' },
    { name: 'Burst suppression', description: 'Bùng nổ-ức chế - hôn mê sâu', severity: 'critical' },
    { name: 'Periodic discharges', description: 'Phóng điện chu kỳ', severity: 'high' },
    { name: 'Asymmetry', description: 'Bất đối xứng hai bán cầu', severity: 'moderate' }
];

// MCP Protocol handler (JSON-RPC over stdio)
class MCPServer {
    constructor() {
        this.rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
            terminal: false
        });
    }

    start() {
        console.error(`🧠 ${SERVER_INFO.name} v${SERVER_INFO.version} started`);
        
        let buffer = '';
        
        this.rl.on('line', (line) => {
            buffer += line;
            try {
                const request = JSON.parse(buffer);
                buffer = '';
                this.handleRequest(request);
            } catch (e) {
                // Incomplete JSON, continue accumulating
            }
        });

        this.rl.on('close', () => {
            console.error('MCP Server shutting down');
            process.exit(0);
        });
    }

    handleRequest(request) {
        const { id, method, params } = request;

        switch (method) {
            case 'initialize':
                this.sendResponse(id, {
                    protocolVersion: '2024-11-05',
                    capabilities: {
                        tools: {},
                        resources: {}
                    },
                    serverInfo: SERVER_INFO
                });
                break;

            case 'initialized':
                // Notification, no response needed
                break;

            case 'tools/list':
                this.sendResponse(id, { tools: TOOLS });
                break;

            case 'tools/call':
                this.handleToolCall(id, params);
                break;

            case 'resources/list':
                this.sendResponse(id, { resources: RESOURCES });
                break;

            case 'resources/read':
                this.handleResourceRead(id, params);
                break;

            default:
                this.sendError(id, -32601, `Method not found: ${method}`);
        }
    }

    async handleToolCall(id, params) {
        const { name, arguments: args } = params;

        switch (name) {
            case 'analyze_eeg':
                this.sendResponse(id, {
                    content: [{
                        type: 'text',
                        text: JSON.stringify({
                            message: 'Phân tích EEG cần được thực hiện qua API endpoint /api/analyze',
                            endpoint: 'POST /api/analyze',
                            imagePath: args.imagePath
                        })
                    }]
                });
                break;

            case 'generate_report':
                this.sendResponse(id, {
                    content: [{
                        type: 'text',
                        text: JSON.stringify({
                            message: 'Báo cáo EEG',
                            analysisId: args.analysisId,
                            format: args.format || 'markdown',
                            endpoint: 'POST /api/mcp/execute'
                        })
                    }]
                });
                break;

            case 'get_analysis_history':
                this.sendResponse(id, {
                    content: [{
                        type: 'text',
                        text: JSON.stringify({
                            message: 'Lịch sử phân tích có thể truy cập qua API',
                            endpoint: 'GET /api/history',
                            limit: args.limit || 10
                        })
                    }]
                });
                break;

            case 'compare_analyses':
                this.sendResponse(id, {
                    content: [{
                        type: 'text',
                        text: JSON.stringify({
                            message: 'So sánh phân tích EEG',
                            analysisIds: args.analysisIds,
                            note: 'Cần ít nhất 2 bản phân tích để so sánh'
                        })
                    }]
                });
                break;

            default:
                this.sendError(id, -32602, `Unknown tool: ${name}`);
        }
    }

    handleResourceRead(id, params) {
        const { uri } = params;

        switch (uri) {
            case 'eeg://knowledge/frequency-bands':
                this.sendResponse(id, {
                    contents: [{
                        uri,
                        mimeType: 'application/json',
                        text: JSON.stringify(FREQUENCY_BANDS, null, 2)
                    }]
                });
                break;

            case 'eeg://knowledge/abnormalities':
                this.sendResponse(id, {
                    contents: [{
                        uri,
                        mimeType: 'application/json',
                        text: JSON.stringify(ABNORMALITIES, null, 2)
                    }]
                });
                break;

            default:
                this.sendError(id, -32602, `Unknown resource: ${uri}`);
        }
    }

    sendResponse(id, result) {
        const response = { jsonrpc: '2.0', id, result };
        process.stdout.write(JSON.stringify(response) + '\n');
    }

    sendError(id, code, message) {
        const response = { jsonrpc: '2.0', id, error: { code, message } };
        process.stdout.write(JSON.stringify(response) + '\n');
    }
}

const server = new MCPServer();
server.start();
