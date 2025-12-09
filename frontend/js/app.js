class NetworkMonitor {
    constructor() {
        // ⚠️ IMPORTANTE: Usar URL dinâmica do próprio frontend
        this.API_BASE = window.location.origin;
        
        this.autoRefreshInterval = null;
        this.currentView = 'all';
        this.currentData = null;
        
        console.log('🔗 Frontend URL:', window.location.origin);
        console.log('🔗 API Base:', this.API_BASE);
        
        this.initializeEventListeners();
        this.checkAPIStatus();
        this.loadData();
    }
    
    initializeEventListeners() {
        document.getElementById('refreshBtn').addEventListener('click', () => this.loadData());
        document.getElementById('autoRefreshBtn').addEventListener('click', () => this.toggleAutoRefresh());
        
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.changeView(e.target.dataset.view));
        });
        
        document.querySelector('.close-btn').addEventListener('click', () => this.closeModal());
        document.getElementById('flowModal').addEventListener('click', (e) => {
            if (e.target.id === 'flowModal') this.closeModal();
        });
    }

    async checkAPIStatus() {
    try {
        console.log('🔍 Verificando status da API...');
        const testUrl = `${this.API_BASE}/`;
        console.log('📍 Testando URL:', testUrl);
        
        const response = await fetch(testUrl);
        console.log('📊 Status do teste:', response.status);
        
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        
        if (response.ok) {
            statusDot.className = 'status-dot connected';
            statusText.textContent = 'Conectado';
            console.log('✅ API conectada');
        } else {
            throw new Error(`API retornou status: ${response.status}`);
        }
    } catch (error) {
        console.error('❌ Erro ao conectar com a API:', error);
        this.updateStatus('error', 'Erro de conexão');
    }
}

    updateStatus(type, message) {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        
        statusDot.className = 'status-dot';
        statusText.textContent = message;
        
        if (type === 'error') {
            statusDot.style.background = '#ff6b6b';
        }
    }

    async loadData() {
    this.showLoading();
    
    try {
        console.log('🔍 ============ INICIANDO DEBUG ============');
        
        // 1. Verificar URL da API
        const apiUrl = `${this.API_BASE}/predict_latest`;
        console.log('📍 URL da API:', apiUrl);
        
        // 2. Fazer a requisição
        console.log('📡 Fazendo requisição...');
        const response = await fetch(apiUrl);
        console.log('📊 Status HTTP:', response.status, response.statusText);
        
        // 3. Verificar headers
        console.log('📋 Headers:');
        for (const [key, value] of response.headers.entries()) {
            console.log(`   ${key}: ${value}`);
        }
        
        // 4. Ler resposta como texto primeiro
        const responseText = await response.text();
        console.log('📄 Tamanho da resposta:', responseText.length, 'caracteres');
        console.log('📝 Primeiros 500 caracteres:', responseText.substring(0, 500));
        
        // 5. Verificar se é JSON válido
        let data;
        try {
            data = JSON.parse(responseText);
            console.log('✅ Resposta é JSON válido');
            console.log('📦 Estrutura do JSON:', Object.keys(data));
        } catch (jsonError) {
            console.error('❌ NÃO É JSON VÁLIDO:', jsonError);
            
            // Verificar se é HTML
            if (responseText.includes('<!DOCTYPE') || responseText.includes('<html')) {
                console.error('🚨 A API está retornando HTML em vez de JSON!');
                console.error('💡 Possíveis causas:');
                console.error('   - URL errada (está apontando para o frontend)');
                console.error('   - Problema de roteamento no ngrok');
                console.error('   - Backend não está respondendo na porta correta');
            }
            
            throw new Error(`A API retornou HTML em vez de JSON. Verifique se o backend está rodando corretamente.`);
        }
        
        // 6. VALIDAÇÃO DOS DADOS
        if (!data) {
            throw new Error('Resposta vazia da API');
        }
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        console.log('🎯 Dados recebidos:', {
            total_flows: data.total_flows,
            attacks: data.attacks, 
            normal: data.normal,
            attack_rate: data.attack_rate,
            data_length: data.data ? data.data.length : 0
        });
        
        // 7. Processar dados
        const safeData = {
            total_flows: this.getSafeNumber(data.total_flows),
            attacks: this.getSafeNumber(data.attacks),
            normal: this.getSafeNumber(data.normal),
            attack_rate: this.getSafeNumber(data.attack_rate),
            attack_stats: data.attack_stats || {},
            data: Array.isArray(data.data) ? data.data : []
        };
        
        console.log('🛡️ Dados validados:', safeData);
        console.log('✅ ============ DEBUG CONCLUÍDO ============');
        
        this.currentData = safeData;
        this.updateDashboard(safeData);
        this.updateFlowsTable(safeData.data);
        this.updateAttackStats(safeData.attack_stats);
        this.updateLastUpdate();
        
    } catch (error) {
        console.error('❌ ============ ERRO DETALHADO ============');
        console.error('Erro:', error);
        console.error('Stack:', error.stack);
        console.error('❌ =======================================');
        
        this.showError(`Erro: ${error.message}`);
        this.showDemoData(); // Fallback para dados de demonstração
    } finally {
        this.hideLoading();
    }
}

    // ✅ FUNÇÃO AUXILIAR PARA VALIDAÇÃO
    getSafeNumber(value) {
        if (value === undefined || value === null || isNaN(value)) {
            return 0;
        }
        return Number(value);
    }

    updateDashboard(data) {
        console.log('📊 Atualizando dashboard com:', data);
        
        // ✅ EXTRA SEGURANÇA: Verifica cada valor individualmente
        const totalFlows = this.getSafeNumber(data.total_flows);
        const attacks = this.getSafeNumber(data.attacks);
        const normal = this.getSafeNumber(data.normal);
        const attackRate = this.getSafeNumber(data.attack_rate);
        
        console.log('🎯 Valores calculados:', { totalFlows, attacks, normal, attackRate });
        
        document.getElementById('totalFlows').textContent = totalFlows;
        document.getElementById('attackCount').textContent = attacks;
        document.getElementById('normalCount').textContent = normal;
        document.getElementById('attackRate').textContent = attackRate.toFixed(1) + '%';
    }

    updateFlowsTable(flows) {
        const tbody = document.getElementById('flowsTableBody');
        const tableCount = document.getElementById('tableCount');
        
        if (!Array.isArray(flows) || flows.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="9" style="text-align: center; padding: 40px; color: #adb5bd;">
                        <i class="fas fa-inbox" style="font-size: 2rem; margin-bottom: 10px; display: block;"></i>
                        Nenhum fluxo encontrado
                    </td>
                </tr>
            `;
            tableCount.textContent = 'Mostrando 0 fluxos';
            return;
        }
        
        const filteredFlows = this.filterFlows(flows);
        tbody.innerHTML = '';
        
        filteredFlows.forEach(flow => {
            const row = this.createFlowRow(flow);
            tbody.appendChild(row);
        });
        
        tableCount.textContent = `Mostrando ${filteredFlows.length} de ${flows.length} fluxos`;
    }

    filterFlows(flows) {
        if (!Array.isArray(flows)) return [];
        
        switch (this.currentView) {
            case 'attacks':
                return flows.filter(flow => flow.prediction === 1);
            case 'normal':
                return flows.filter(flow => flow.prediction === 0);
            default:
                return flows;
        }
    }

    createFlowRow(flow) {
        const row = document.createElement('tr');
        
        const timestamp = this.formatTimestamp(flow.timestamp);
        const duration = this.formatDuration(flow.dur);
        const packets = `${flow.spkts || 0} → ${flow.dpkts || 0}`;
        const bytes = this.formatBytes((flow.sbytes || 0) + (flow.dbytes || 0));
        const status = flow.prediction === 1 ? 'Ataque' : 'Normal';
        const statusClass = flow.prediction === 1 ? 'status-attack' : 'status-normal';
        
        row.innerHTML = `
            <td>${timestamp}</td>
            <td>
                <div style="font-weight: 600;">${flow.src_ip || 'N/A'}</div>
                <div style="font-size: 0.8rem; color: #adb5bd;">Porta: ${flow.sport || 'N/A'}</div>
            </td>
            <td>
                <div style="font-weight: 600;">${flow.dst_ip || 'N/A'}</div>
                <div style="font-size: 0.8rem; color: #adb5bd;">Porta: ${flow.dport || 'N/A'}</div>
            </td>
            <td>
                <span style="padding: 4px 8px; background: rgba(76, 201, 240, 0.2); border-radius: 6px; font-size: 0.8rem;">
                    ${this.getProtocolName(flow.proto)}
                </span>
            </td>
            <td>${duration}</td>
            <td>${packets}</td>
            <td>${bytes}</td>
            <td><span class="status-badge ${statusClass}">${status}</span></td>
            <td>
                <button class="action-btn view-details" data-flow='${JSON.stringify(flow).replace(/'/g, "\\'")}'>
                    <i class="fas fa-search"></i> Detalhes
                </button>
            </td>
        `;
        
        row.querySelector('.view-details').addEventListener('click', (e) => {
            const flowData = JSON.parse(e.target.dataset.flow);
            this.showFlowDetails(flowData);
        });
        
        return row;
    }

    showDemoData() {
        console.log('🔄 Carregando dados de demonstração...');
        
        const demoData = {
            total_flows: 15,
            attacks: 2,
            normal: 13,
            attack_rate: 13.3,
            attack_stats: {
                top_protocols: { "tcp": 2 },
                top_source_ips: { "192.168.1.100": 1, "10.0.0.5": 1 }
            },
            data: [
                {
                    timestamp: new Date().toLocaleString('pt-BR'),
                    src_ip: "192.168.1.3",
                    sport: 51864,
                    dst_ip: "8.8.8.8",
                    dport: 53,
                    proto: "udp",
                    dur: 1500,
                    spkts: 1,
                    dpkts: 1,
                    sbytes: 64,
                    dbytes: 128,
                    rate: 51200,
                    prediction: 0,
                    label: "Normal"
                },
                {
                    timestamp: new Date().toLocaleString('pt-BR'),
                    src_ip: "192.168.1.100",
                    sport: 445,
                    dst_ip: "192.168.1.3",
                    dport: 49152,
                    proto: "tcp",
                    dur: 5000,
                    spkts: 100,
                    dpkts: 0,
                    sbytes: 5120,
                    dbytes: 0,
                    rate: 102400,
                    prediction: 1,
                    label: "Ataque"
                }
            ]
        };
        
        this.currentData = demoData;
        this.updateDashboard(demoData);
        this.updateFlowsTable(demoData.data);
        this.updateAttackStats(demoData.attack_stats);
        this.updateLastUpdate();
    }

    // ... (mantenha as outras funções como formatTimestamp, formatDuration, etc.)

    formatTimestamp(timestamp) {
        if (!timestamp) return 'N/A';
        try {
            const date = new Date(timestamp);
            return isNaN(date.getTime()) ? timestamp : date.toLocaleString('pt-BR');
        } catch {
            return timestamp;
        }
    }

    formatDuration(milliseconds) {
        if (!milliseconds) return '0s';
        const seconds = Math.floor(milliseconds / 1000);
        if (seconds < 60) return `${seconds}s`;
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}m ${remainingSeconds}s`;
    }

    formatBytes(bytes) {
        if (!bytes) return '0 B';
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }

    getProtocolName(proto) {
        const protocols = {
            '6': 'TCP', '17': 'UDP', '1': 'ICMP', 'tcp': 'TCP', 
            'udp': 'UDP', 'icmp': 'ICMP', 'missing': 'Desconhecido'
        };
        return protocols[String(proto).toLowerCase()] || String(proto);
    }

    updateAttackStats(stats) {
        const statsSection = document.getElementById('attackStatsSection');
        const statsContainer = document.getElementById('attackStats');
        
        if (!stats || Object.keys(stats).length === 0) {
            statsSection.style.display = 'none';
            return;
        }
        
        statsSection.style.display = 'block';
        statsContainer.innerHTML = '';
        
        if (stats.top_protocols) {
            const protoStats = this.createStatsItem('Protocolos em Ataques', stats.top_protocols);
            statsContainer.appendChild(protoStats);
        }
        
        if (stats.top_source_ips) {
            const ipStats = this.createStatsItem('IPs Fonte em Ataques', stats.top_source_ips);
            statsContainer.appendChild(ipStats);
        }
    }

    createStatsItem(title, data) {
        const item = document.createElement('div');
        item.className = 'stat-item';
        let html = `<h4>${title}</h4><ul class="stat-list">`;
        Object.entries(data).forEach(([key, value]) => {
            html += `<li><span>${key}</span><span class="stat-count">${value}</span></li>`;
        });
        html += '</ul>';
        item.innerHTML = html;
        return item;
    }

    changeView(view) {
        this.currentView = view;
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === view);
        });
        if (this.currentData) {
            this.updateFlowsTable(this.currentData.data);
        }
    }

    showFlowDetails(flow) {
        const modalBody = document.getElementById('modalBody');
        const details = `
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                <div><h4 style="color: #4cc9f0;">Origem</h4><p><strong>IP:</strong> ${flow.src_ip}</p><p><strong>Porta:</strong> ${flow.sport}</p></div>
                <div><h4 style="color: #4cc9f0;">Destino</h4><p><strong>IP:</strong> ${flow.dst_ip}</p><p><strong>Porta:</strong> ${flow.dport}</p></div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div><h4 style="color: #4cc9f0;">Informações</h4><p><strong>Protocolo:</strong> ${this.getProtocolName(flow.proto)}</p><p><strong>Duração:</strong> ${this.formatDuration(flow.dur)}</p><p><strong>Timestamp:</strong> ${this.formatTimestamp(flow.timestamp)}</p><p><strong>Status:</strong> <span class="status-badge ${flow.prediction === 1 ? 'status-attack' : 'status-normal'}">${flow.prediction === 1 ? 'Ataque' : 'Normal'}</span></p></div>
                <div><h4 style="color: #4cc9f0;">Estatísticas</h4><p><strong>Pacotes:</strong> ${flow.spkts} → ${flow.dpkts || 0}</p><p><strong>Bytes:</strong> ${this.formatBytes((flow.sbytes || 0) + (flow.dbytes || 0))}</p><p><strong>Taxa:</strong> ${flow.rate ? Math.round(flow.rate) : 0} B/s</p></div>
            </div>
            ${flow.prediction === 1 ? `<div style="margin-top: 20px; padding: 15px; background: rgba(255, 107, 107, 0.1); border-radius: 8px; border-left: 4px solid #ff6b6b;"><h4 style="color: #ff6b6b;"><i class="fas fa-exclamation-triangle"></i> Alerta de Segurança</h4><p>Este fluxo foi classificado como potencial ataque.</p></div>` : ''}
        `;
        modalBody.innerHTML = details;
        document.getElementById('flowModal').style.display = 'flex';
    }

    closeModal() {
        document.getElementById('flowModal').style.display = 'none';
    }

    toggleAutoRefresh() {
        const btn = document.getElementById('autoRefreshBtn');
        if (this.autoRefreshInterval) {
            clearInterval(this.autoRefreshInterval);
            this.autoRefreshInterval = null;
            btn.innerHTML = '<i class="fas fa-play"></i> Auto-Refresh (10s)';
        } else {
            this.autoRefreshInterval = setInterval(() => this.loadData(), 10000);
            btn.innerHTML = '<i class="fas fa-stop"></i> Parar Auto-Refresh';
        }
    }

    updateLastUpdate() {
        const now = new Date();
        document.getElementById('lastUpdate').textContent = `Última atualização: ${now.toLocaleString('pt-BR')}`;
    }

    showLoading() {
        document.getElementById('loadingOverlay').style.display = 'flex';
    }

    hideLoading() {
        document.getElementById('loadingOverlay').style.display = 'none';
    }

    showError(message) {
        alert(message);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new NetworkMonitor();
});