const API = 'http://localhost:5014/api';
const L = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: 'Inter,sans-serif', color: '#e5e5e5', size: 12 }, xaxis: { gridcolor: 'rgba(255,255,255,.06)' }, yaxis: { gridcolor: 'rgba(255,255,255,.06)' }, margin: { t: 30, r: 30, b: 50, l: 60 }, hovermode: 'closest' };
const C = { responsive: true, displayModeBar: false };
async function f(ep) { try { const r = await fetch(`${API}/${ep}`); const d = await r.json(); if (d.success) return d; throw new Error(d.error) } catch (e) { console.error(ep, e); return null } }

function uKPI(d) {
    const el = document.getElementById('kpiCards'); if (!d) return; const m = d.metrics, s = d.summary;
    el.innerHTML = `
    <div class="card kpi-card"><div class="ki"><i class="fa-solid fa-bullseye"></i></div><div class="kv" style="color:hsl(25,90%,55%)">${m.f1_score}</div><div class="kl">F1 Score</div></div>
    <div class="card kpi-card"><div class="ki"><i class="fa-solid fa-magnifying-glass"></i></div><div class="kv" style="color:hsl(140,65%,50%)">${m.precision}</div><div class="kl">Precision</div></div>
    <div class="card kpi-card"><div class="ki"><i class="fa-solid fa-satellite-dish"></i></div><div class="kv" style="color:hsl(200,70%,55%)">${m.recall}</div><div class="kl">Recall</div></div>
    <div class="card kpi-card"><div class="ki"><i class="fa-solid fa-triangle-exclamation"></i></div><div class="kv" style="color:hsl(0,70%,55%)">${m.total_anomalies_detected}</div><div class="kl">Detected / ${m.actual_anomalies} Actual</div></div>`
}

function cSensor(data, chartId, field, label, color) {
    const norm = data.filter(d => !d.predicted_anomaly), anom = data.filter(d => d.predicted_anomaly);
    Plotly.newPlot(chartId, [
        { x: norm.map(d => d.timestamp), y: norm.map(d => d[field]), type: 'scatter', mode: 'lines', name: 'Normal', line: { color, width: 1 } },
        { x: anom.map(d => d.timestamp), y: anom.map(d => d[field]), type: 'scatter', mode: 'markers', name: 'Anomaly', marker: { color: 'rgb(239,68,68)', size: 6, symbol: 'x', line: { width: 1, color: 'white' } } }
    ], { ...L, xaxis: { ...L.xaxis, title: 'Time' }, yaxis: { ...L.yaxis, title: label }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C)
}

function cScores(data) {
    const scores = data.map(d => d.anomaly_score);
    Plotly.newPlot('scoreChart', [{ x: scores, type: 'histogram', nbinsx: 60, marker: { color: 'rgba(249,115,22,.6)', line: { color: 'rgba(249,115,22,.9)', width: 1 } } }], { ...L, xaxis: { ...L.xaxis, title: 'Anomaly Score' }, yaxis: { ...L.yaxis, title: 'Frequency' }, shapes: [{ type: 'line', x0: 0, x1: 0, y0: 0, y1: 1, yref: 'paper', line: { color: 'rgb(239,68,68)', width: 2, dash: 'dash' } }], annotations: [{ x: 0, y: 1, yref: 'paper', text: 'Threshold', showarrow: false, font: { color: 'rgb(239,68,68)', size: 11 }, yanchor: 'bottom' }] }, C)
}

function cConfusion(m) {
    const z = [[m.tn, m.fp], [m.fn, m.tp]];
    const labels = [['TN', 'FP'], ['FN', 'TP']];
    Plotly.newPlot('confChart', [{ z, x: ['Predicted Normal', 'Predicted Anomaly'], y: ['Actual Normal', 'Actual Anomaly'], type: 'heatmap', colorscale: [[0, 'rgb(20,15,10)'], [1, 'rgb(249,115,22)']], text: z.map((r, i) => r.map((v, j) => `${labels[i][j]}: ${v}`)), texttemplate: '%{text}', showscale: false }], { ...L, margin: { ...L.margin, l: 130 } }, C)
}

function uTable(d) {
    const el = document.getElementById('anomTable'); if (!d || !d.anomalies.length) { el.innerHTML = '<p style="color:var(--text-muted)">No anomalies detected</p>'; return }
    let html = '<table class="anom-table"><thead><tr><th>Time</th><th>Power</th><th>Temp</th><th>Network</th><th>Score</th><th>Type</th><th>Real?</th></tr></thead><tbody>';
    d.anomalies.slice(-20).reverse().forEach(a => {
        const tc = a.anomaly_type.includes('power') ? 'power' : a.anomaly_type.includes('temp') ? 'temp' : a.anomaly_type.includes('network') ? 'network' : 'random';
        html += `<tr><td>${new Date(a.timestamp).toLocaleString()}</td><td>${a.power_kwh}</td><td>${a.server_temp_c}°C</td><td>${a.network_mbps}</td><td>${a.anomaly_score?.toFixed(3) || '—'}</td><td><span class="type-badge ${tc}">${a.anomaly_type}</span></td><td>${a.is_anomaly ? '✅' : '—'}</td></tr>`
    });
    html += '</tbody></table>'; el.innerHTML = html
}

async function load() {
    const [st, dt, an] = await Promise.all([f('stats'), f('data'), f('anomalies')]);
    if (st) { uKPI(st); cConfusion(st.metrics) }
    if (dt) { cSensor(dt.data, 'powerChart', 'power_kwh', 'Power (kWh)', 'rgb(249,115,22)'); cSensor(dt.data, 'tempChart', 'server_temp_c', 'Temp (°C)', 'rgb(239,68,68)'); cSensor(dt.data, 'netChart', 'network_mbps', 'Bandwidth (Mbps)', 'rgb(56,189,248)'); cScores(dt.data) }
    if (an) uTable(an);
    document.getElementById('lastUpdate').textContent = `Updated: ${new Date().toLocaleTimeString()}`
}
document.addEventListener('DOMContentLoaded', () => { load(); setInterval(load, 30000) });
