const API = 'http://localhost:5006/api';
const L = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: 'Inter,sans-serif', color: '#e5e5e5', size: 12 }, xaxis: { gridcolor: 'rgba(255,255,255,.06)', zerolinecolor: 'rgba(255,255,255,.08)' }, yaxis: { gridcolor: 'rgba(255,255,255,.06)', zerolinecolor: 'rgba(255,255,255,.08)' }, margin: { t: 30, r: 30, b: 50, l: 60 }, hovermode: 'closest' };
const C = { responsive: true, displayModeBar: false };
async function f(ep) { try { const r = await fetch(`${API}/${ep}`); const d = await r.json(); if (d.success) return d; throw new Error(d.error) } catch (e) { console.error(ep, e); return null } }

function updatePrediction(d) {
    const el = document.getElementById('predictionContent'); if (!d) return; const { prediction: p, current: c } = d;
    el.innerHTML = `<div class="prediction-label">Forecasted Load</div><div class="prediction-value">${p.prediction} <span style="font-size:1.2rem">kWh</span></div>
    <div style="font-size:.8rem;color:var(--text-muted);margin-top:.25rem">${c.is_weekend ? '<i class="fa-solid fa-calendar"></i> Weekend' : '<i class="fa-solid fa-clipboard"></i> Weekday'} · <i class="fa-solid fa-users"></i> ${c.occupancy}</div>
    <div class="confidence-bar"><div class="ci-item"><div class="ci-label">Lower</div><div class="ci-value">${p.lower_bound}</div></div><div class="ci-item"><div class="ci-label">Conf</div><div class="ci-value">${p.confidence_level}%</div></div><div class="ci-item"><div class="ci-label">Upper</div><div class="ci-value">${p.upper_bound}</div></div></div>`
}

function updateDip(d) {
    const el = document.getElementById('dipContent'); if (!d) return; const dp = d.dip;
    el.innerHTML = `<div class="insight-grid">
    <div class="insight-item"><div class="il">Weekday Avg</div><div class="iv">${dp.weekday_avg} kWh</div></div>
    <div class="insight-item"><div class="il">Weekend Avg</div><div class="iv">${dp.weekend_avg} kWh</div></div>
    <div class="insight-item"><div class="il">Weekend Dip</div><div class="iv good">↓${dp.dip_pct}%</div></div>
    <div class="insight-item"><div class="il">Mon. Surge</div><div class="iv">${dp.monday_surge_avg} kWh</div></div></div>`
}

function updateMetrics(d) {
    const el = document.getElementById('modelMetrics'); if (!d) return; const m = d.metrics;
    el.innerHTML = `<div class="metrics-grid"><div class="metric-item"><div class="metric-label">R²</div><div class="metric-value good">${m.r2}</div></div><div class="metric-item"><div class="metric-label">RMSE</div><div class="metric-value">${m.rmse}<span class="metric-unit">kWh</span></div></div><div class="metric-item"><div class="metric-label">MAE</div><div class="metric-value">${m.mae}<span class="metric-unit">kWh</span></div></div><div class="metric-item"><div class="metric-label">MAPE</div><div class="metric-value">${m.mape}<span class="metric-unit">%</span></div></div></div>`
}

function chartTS(data) {
    const wk = data.filter(d => !d.is_weekend), we = data.filter(d => d.is_weekend);
    Plotly.newPlot('timeSeriesChart', [{ x: wk.map(d => d.timestamp), y: wk.map(d => d.electricity_kwh), type: 'scatter', mode: 'lines', name: 'Weekday', line: { color: 'rgb(96,165,250)', width: 1.5 } }, { x: we.map(d => d.timestamp), y: we.map(d => d.electricity_kwh), type: 'scatter', mode: 'lines', name: 'Weekend', line: { color: 'rgb(52,211,153)', width: 2 } }], { ...L, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' }, xaxis: { ...L.xaxis, title: 'Date' }, yaxis: { ...L.yaxis, title: 'kWh' } }, C)
}

function chartHourly(data) {
    const wd = {}, we = {}; data.forEach(d => { const h = new Date(d.timestamp).getHours(), b = d.is_weekend ? we : wd; if (!b[h]) b[h] = { t: 0, c: 0 }; b[h].t += d.electricity_kwh; b[h].c++ });
    const hrs = Array.from({ length: 24 }, (_, i) => i);
    Plotly.newPlot('hourlyChart', [{ x: hrs, y: hrs.map(h => wd[h] ? wd[h].t / wd[h].c : 0), type: 'bar', name: 'Weekday', marker: { color: 'rgba(96,165,250,.7)' } }, { x: hrs, y: hrs.map(h => we[h] ? we[h].t / we[h].c : 0), type: 'bar', name: 'Weekend', marker: { color: 'rgba(52,211,153,.7)' } }], { ...L, barmode: 'group', xaxis: { ...L.xaxis, title: 'Hour', tickmode: 'linear', dtick: 2 }, yaxis: { ...L.yaxis, title: 'Avg kWh' }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C)
}

function chartFeatures(fd) { const s = Object.entries(fd).sort((a, b) => b[1] - a[1]); Plotly.newPlot('featureChart', [{ x: s.map(([, v]) => v), y: s.map(([n]) => n.replace(/_/g, ' ')), type: 'bar', orientation: 'h', marker: { color: s.map(([, v]) => v), colorscale: [[0, 'rgb(96,165,250)'], [1, 'rgb(59,130,246)']], line: { width: 1, color: 'rgba(255,255,255,.1)' } } }], { ...L, margin: { ...L.margin, l: 160 }, xaxis: { ...L.xaxis, title: 'Importance' } }, C) }

function chartDow(data) {
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], avgs = days.map((_, i) => { const sub = data.filter(d => new Date(d.timestamp).getDay() === (i === 6 ? 0 : i + 1)); return sub.length ? sub.reduce((a, d) => a + d.electricity_kwh, 0) / sub.length : 0 });
    Plotly.newPlot('dowChart', [{ x: days, y: avgs, type: 'bar', marker: { color: avgs.map(v => v > 15 ? 'rgb(96,165,250)' : 'rgb(52,211,153)'), line: { color: 'rgba(255,255,255,.15)', width: 1 } } }], { ...L, xaxis: { ...L.xaxis }, yaxis: { ...L.yaxis, title: 'Avg kWh' } }, C)
}

function chartHeatmap(data) { const dn = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], g = Array.from({ length: 7 }, () => Array(24).fill(0)), c = Array.from({ length: 7 }, () => Array(24).fill(0)); data.forEach(d => { const dt = new Date(d.timestamp), dw = dt.getDay() === 0 ? 6 : dt.getDay() - 1, h = dt.getHours(); g[dw][h] += d.electricity_kwh; c[dw][h]++ }); Plotly.newPlot('heatmapChart', [{ z: g.map((r, i) => r.map((v, j) => c[i][j] ? v / c[i][j] : 0)), x: Array.from({ length: 24 }, (_, i) => `${i}:00`), y: dn, type: 'heatmap', colorscale: [[0, 'rgb(15,23,42)'], [.5, 'rgb(96,165,250)'], [1, 'rgb(239,68,68)']], colorbar: { title: 'kWh', thickness: 10 } }], L, C) }

async function load() { const [h, p, s, dp] = await Promise.all([f('historical'), f('predict'), f('stats'), f('weekend-dip')]); if (p) updatePrediction(p); if (dp) updateDip(dp); if (s) { updateMetrics(s); chartFeatures(s.feature_importance) } if (h) { chartTS(h.data); chartHourly(h.data); chartDow(h.data); chartHeatmap(h.data) } document.getElementById('lastUpdate').textContent = `Updated: ${new Date().toLocaleTimeString()}` }
document.addEventListener('DOMContentLoaded', () => { load(); setInterval(load, 30000) });
