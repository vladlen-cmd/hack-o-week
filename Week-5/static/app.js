const API = 'http://localhost:5004/api';
const L = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: 'Inter,sans-serif', color: '#e5e5e5', size: 12 }, xaxis: { gridcolor: 'rgba(255,255,255,.06)', zerolinecolor: 'rgba(255,255,255,.08)' }, yaxis: { gridcolor: 'rgba(255,255,255,.06)', zerolinecolor: 'rgba(255,255,255,.08)' }, margin: { t: 30, r: 30, b: 50, l: 60 }, hovermode: 'closest' };
const C = { responsive: true, displayModeBar: false };
async function f(ep) { try { const r = await fetch(`${API}/${ep}`); const d = await r.json(); if (d.success) return d; throw new Error(d.error) } catch (e) { console.error(ep, e); return null } }

function updatePrediction(d) {
    const el = document.getElementById('predictionContent');
    if (!d) { el.innerHTML = '<p style="color:var(--text-muted)">No data</p>'; return }
    const { prediction: p, current: c } = d;
    el.innerHTML = `<div class="prediction-label">Forecasted HVAC Load</div>
        <div class="prediction-value">${p.prediction} <span style="font-size:1.2rem">kWh</span></div>
        <div style="font-size:.8rem;color:var(--text-muted);margin-top:.25rem"><i class="fa-solid fa-temperature-half"></i> Outside: ${c.temp_outside}°C · Setpoint: ${c.setpoint}°C · <i class="fa-solid fa-users"></i> ${c.occupancy} ppl</div>
        <div class="confidence-bar"><div class="ci-item"><div class="ci-label">Lower</div><div class="ci-value">${p.lower_bound}</div></div><div class="ci-item"><div class="ci-label">Confidence</div><div class="ci-value">${p.confidence_level}%</div></div><div class="ci-item"><div class="ci-label">Upper</div><div class="ci-value">${p.upper_bound}</div></div></div>`;
}

function updateInsights(d) {
    const el = document.getElementById('insightsContent');
    if (!d) { el.innerHTML = '<p style="color:var(--text-muted)">No data</p>'; return }
    const i = d.insights;
    el.innerHTML = `<div class="insight-grid">
        <div class="insight-item"><div class="il">Active Avg</div><div class="iv">${i.active_avg}<span class="metric-unit">kWh</span></div></div>
        <div class="insight-item"><div class="il">Idle Avg</div><div class="iv">${i.idle_avg}<span class="metric-unit">kWh</span></div></div>
        <div class="insight-item"><div class="il">Savings Potential</div><div class="iv green">${i.potential_savings_pct}%</div></div>
        <div class="insight-item"><div class="il">Peak Hour</div><div class="iv">${i.peak_hour}:00</div></div>
    </div>`;
}

function updateMetrics(d) {
    const el = document.getElementById('modelMetrics');
    if (!d) return; const m = d.metrics;
    el.innerHTML = `<div class="metrics-grid">
        <div class="metric-item"><div class="metric-label">R² Score</div><div class="metric-value good">${m.r2}</div></div>
        <div class="metric-item"><div class="metric-label">RMSE</div><div class="metric-value">${m.rmse}<span class="metric-unit">kWh</span></div></div>
        <div class="metric-item"><div class="metric-label">MAE</div><div class="metric-value">${m.mae}<span class="metric-unit">kWh</span></div></div>
        <div class="metric-item"><div class="metric-label">MAPE</div><div class="metric-value">${m.mape}<span class="metric-unit">%</span></div></div>
    </div>`;
}

function chartTimeSeries(data) {
    Plotly.newPlot('timeSeriesChart', [{ x: data.map(d => d.timestamp), y: data.map(d => d.hvac_kwh), type: 'scatter', mode: 'lines', name: 'HVAC Energy', line: { color: 'rgb(56,189,248)', width: 1.5 } }, { x: data.map(d => d.timestamp), y: data.map(d => d.temp_outside), type: 'scatter', mode: 'lines', name: 'Temp Outside', yaxis: 'y2', line: { color: 'rgb(251,146,60)', width: 1, dash: 'dot' } }], { ...L, yaxis: { ...L.yaxis, title: 'HVAC (kWh)' }, yaxis2: { title: 'Temp (°C)', overlaying: 'y', side: 'right', gridcolor: 'rgba(0,0,0,0)', titlefont: { color: 'rgb(251,146,60)' }, tickfont: { color: 'rgb(251,146,60)' } }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C);
}

function chartHourly(data) {
    const wd = {}, we = {};
    data.forEach(d => { const h = new Date(d.timestamp).getHours(); const b = d.is_weekend ? we : wd; if (!b[h]) b[h] = { t: 0, c: 0 }; b[h].t += d.hvac_kwh; b[h].c++ });
    const hrs = Array.from({ length: 24 }, (_, i) => i);
    Plotly.newPlot('hourlyChart', [
        { x: hrs, y: hrs.map(h => wd[h] ? wd[h].t / wd[h].c : 0), type: 'bar', name: 'Weekday', marker: { color: 'rgba(56,189,248,.7)' } },
        { x: hrs, y: hrs.map(h => we[h] ? we[h].t / we[h].c : 0), type: 'bar', name: 'Weekend', marker: { color: 'rgba(168,85,247,.7)' } }
    ], { ...L, barmode: 'group', xaxis: { ...L.xaxis, title: 'Hour', tickmode: 'linear', dtick: 2 }, yaxis: { ...L.yaxis, title: 'Avg HVAC (kWh)' }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C);
}

function chartFeatures(fd) {
    const s = Object.entries(fd).sort((a, b) => b[1] - a[1]);
    Plotly.newPlot('featureChart', [{ x: s.map(([, v]) => v), y: s.map(([n]) => n.replace(/_/g, ' ')), type: 'bar', orientation: 'h', marker: { color: s.map(([, v]) => v), colorscale: [[0, 'rgb(56,189,248)'], [1, 'rgb(59,130,246)']], line: { color: 'rgba(255,255,255,.1)', width: 1 } } }], { ...L, margin: { ...L.margin, l: 180 }, xaxis: { ...L.xaxis, title: 'Importance' } }, C);
}

function chartTemp(data) {
    Plotly.newPlot('tempChart', [{ x: data.map(d => d.temp_outside), y: data.map(d => d.hvac_kwh), type: 'scatter', mode: 'markers', marker: { color: data.map(d => d.occupancy), colorscale: 'Viridis', size: 4, opacity: .5, colorbar: { title: 'Occ', thickness: 10 } } }], { ...L, xaxis: { ...L.xaxis, title: 'Outside Temp (°C)' }, yaxis: { ...L.yaxis, title: 'HVAC (kWh)' } }, C);
}

function chartHeatmap(data) {
    const dn = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], g = Array.from({ length: 7 }, () => Array(24).fill(0)), c = Array.from({ length: 7 }, () => Array(24).fill(0));
    data.forEach(d => { const dt = new Date(d.timestamp), dw = dt.getDay() === 0 ? 6 : dt.getDay() - 1, h = dt.getHours(); g[dw][h] += d.hvac_kwh; c[dw][h]++ });
    Plotly.newPlot('heatmapChart', [{ z: g.map((r, i) => r.map((v, j) => c[i][j] ? v / c[i][j] : 0)), x: Array.from({ length: 24 }, (_, i) => `${i}:00`), y: dn, type: 'heatmap', colorscale: [[0, 'rgb(15,23,42)'], [.5, 'rgb(56,189,248)'], [1, 'rgb(239,68,68)']], colorbar: { title: 'kWh', thickness: 10 } }], { ...L }, C);
}

async function load() {
    const [h, p, s, o] = await Promise.all([f('historical'), f('predict'), f('stats'), f('optimization')]);
    if (p) updatePrediction(p); if (o) updateInsights(o);
    if (s) { updateMetrics(s); chartFeatures(s.feature_importance) }
    if (h) { chartTimeSeries(h.data); chartHourly(h.data); chartTemp(h.data); chartHeatmap(h.data) }
    document.getElementById('lastUpdate').textContent = `Updated: ${new Date().toLocaleTimeString()}`;
}
document.addEventListener('DOMContentLoaded', () => { load(); setInterval(load, 30000) });
