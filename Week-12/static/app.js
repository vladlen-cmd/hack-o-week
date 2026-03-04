const API = 'http://localhost:5011/api';
const L = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: 'Inter,sans-serif', color: '#e5e5e5', size: 12 }, xaxis: { gridcolor: 'rgba(255,255,255,.06)' }, yaxis: { gridcolor: 'rgba(255,255,255,.06)' }, margin: { t: 30, r: 30, b: 50, l: 60 }, hovermode: 'closest' };
const C = { responsive: true, displayModeBar: false };
const AC = 'rgb(244,114,182)';

const socket = io('http://localhost:5011');
let hrData = [], hrTimes = [], MX = 60;

socket.on('connect', () => { document.getElementById('wsBadge').innerHTML = '<span class="status-dot"></span> Live (WebSocket)'; });
socket.on('disconnect', () => { document.getElementById('wsBadge').innerHTML = '<span class="status-dot off"></span> Disconnected'; });

socket.on('ack', (d) => {
    document.getElementById('wsMsg').textContent = `Encrypted & stored (${d.encrypted_size} bytes)`;
    document.getElementById('wsMsg').style.color = AC;
});

socket.on('new_reading', (d) => {
    hrTimes.push(d.timestamp); hrData.push(d.heart_rate || 0);
    if (hrTimes.length > MX) { hrTimes.shift(); hrData.shift(); }
    Plotly.react('hrChart', [{ x: hrTimes, y: hrData, type: 'scatter', mode: 'lines+markers', line: { color: AC, width: 2 }, marker: { size: 4 } }],
        { ...L, xaxis: { ...L.xaxis, title: 'Time' }, yaxis: { ...L.yaxis, title: 'BPM', range: [40, 140] } }, C);
    loadStats(); loadReadings();
});

function sendWS() {
    socket.emit('wearable_data', {
        heart_rate: parseInt(document.getElementById('wHR').value),
        steps: parseInt(document.getElementById('wSteps').value),
        spo2: parseInt(document.getElementById('wSpO2').value),
        user_id: document.getElementById('wUser').value, device: 'WebPortal'
    });
}

async function loadStats() {
    try {
        const r = await fetch(`${API}/stats`); const d = await r.json(); if (!d.success) return;
        document.getElementById('kpiCards').innerHTML = `
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-database"></i></div><div class="kv">${d.total_records}</div><div class="kl">Encrypted Records</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-users"></i></div><div class="kv">${d.unique_users}</div><div class="kl">Users</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-lock"></i></div><div class="kv">Fernet</div><div class="kl">Encryption</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-wifi"></i></div><div class="kv">WS</div><div class="kl">Transport</div></div>`;
    } catch (e) { }
}

async function loadReadings() {
    try {
        const r = await fetch(`${API}/readings`); const d = await r.json(); if (!d.success) return;
        let html = '<table class="anom-table"><thead><tr><th>Time</th><th>User</th><th>HR</th><th>Steps</th><th>SpO2</th><th>Device</th></tr></thead><tbody>';
        d.data.slice(0, 15).forEach(r => { html += `<tr><td>${new Date(r.timestamp).toLocaleString()}</td><td>${r.user_id}</td><td>${r.heart_rate || '-'}</td><td>${r.steps || '-'}</td><td>${r.spo2 || '-'}</td><td>${r.device}</td></tr>` });
        html += '</tbody></table>'; document.getElementById('readingsTable').innerHTML = html;
    } catch (e) { }
}

document.addEventListener('DOMContentLoaded', () => {
    loadStats(); loadReadings();
    Plotly.newPlot('hrChart', [{ x: [], y: [], type: 'scatter', mode: 'lines', line: { color: AC } }], { ...L, yaxis: { ...L.yaxis, title: 'BPM', range: [40, 140] } }, C);
});
