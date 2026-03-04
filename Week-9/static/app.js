const API = 'http://localhost:5008/api';
const L = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: 'Inter,sans-serif', color: '#e5e5e5', size: 12 }, xaxis: { gridcolor: 'rgba(255,255,255,.06)' }, yaxis: { gridcolor: 'rgba(255,255,255,.06)' }, margin: { t: 30, r: 30, b: 50, l: 60 }, hovermode: 'closest' };
const C = { responsive: true, displayModeBar: false };
async function f(ep) { try { const r = await fetch(`${API}/${ep}`); const d = await r.json(); if (d.success) return d; throw new Error(d.error) } catch (e) { console.error(ep, e); return null } }

function uPred(d) {
    const el = document.getElementById('pred'); if (!d) return; const { prediction: p, current: c } = d;
    el.innerHTML = `<div class="prediction-label">Forecasted Laundry Load</div><div class="prediction-value">${p.prediction} <span style="font-size:1.2rem">kWh</span></div>
<div style="font-size:.8rem;color:var(--text-muted);margin-top:.25rem"><i class="fa-solid fa-basket-shopping"></i> ${c.machines} machines active</div>
<div class="confidence-bar"><div class="ci-item"><div class="ci-label">Lower</div><div class="ci-value">${p.lower_bound}</div></div><div class="ci-item"><div class="ci-label">Conf</div><div class="ci-value">${p.confidence_level}%</div></div><div class="ci-item"><div class="ci-label">Upper</div><div class="ci-value">${p.upper_bound}</div></div></div>`
}

function uPeak(d) {
    const el = document.getElementById('peakContent'); if (!d) return; const a = d.analysis;
    el.innerHTML = `<div class="insight-grid">
<div class="insight-item"><div class="il">Peak Hour</div><div class="iv teal">${a.peak_hour}:00</div></div>
<div class="insight-item"><div class="il">Peak Avg</div><div class="iv">${a.peak_avg} kWh</div></div>
<div class="insight-item"><div class="il">Sunday Avg</div><div class="iv teal">${a.sunday_avg} kWh</div></div>
<div class="insight-item"><div class="il">Max Machines</div><div class="iv">${a.max_machines}</div></div></div>`
}

function uMetrics(d) {
    const el = document.getElementById('metrics'); if (!d) return; const m = d.metrics;
    el.innerHTML = `<div class="metrics-grid"><div class="metric-item"><div class="metric-label">R²</div><div class="metric-value good">${m.r2}</div></div><div class="metric-item"><div class="metric-label">RMSE</div><div class="metric-value">${m.rmse}<span class="metric-unit">kWh</span></div></div><div class="metric-item"><div class="metric-label">MAE</div><div class="metric-value">${m.mae}<span class="metric-unit">kWh</span></div></div><div class="metric-item"><div class="metric-label">MAPE</div><div class="metric-value">${m.mape}<span class="metric-unit">%</span></div></div></div>`
}

function cTS(data) { Plotly.newPlot('tsChart', [{ x: data.map(d => d.timestamp), y: data.map(d => d.electricity_kwh), type: 'scatter', mode: 'lines', name: 'kWh', line: { color: 'rgb(45,212,191)', width: 1.5 } }, { x: data.map(d => d.timestamp), y: data.map(d => d.machines_active), type: 'scatter', mode: 'lines', name: 'Machines', yaxis: 'y2', line: { color: 'rgb(251,146,60)', width: 1, dash: 'dot' } }], { ...L, yaxis: { ...L.yaxis, title: 'kWh' }, yaxis2: { title: 'Machines', overlaying: 'y', side: 'right', gridcolor: 'rgba(0,0,0,0)', titlefont: { color: 'rgb(251,146,60)' }, tickfont: { color: 'rgb(251,146,60)' } }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C) }

function cHourly(data) {
    const hr = {}; data.forEach(d => { const h = new Date(d.timestamp).getHours(); if (!hr[h]) hr[h] = { t: 0, c: 0 }; hr[h].t += d.electricity_kwh; hr[h].c++ });
    const hrs = Array.from({ length: 24 }, (_, i) => i), vals = hrs.map(h => hr[h] ? hr[h].t / hr[h].c : 0);
    Plotly.newPlot('hourlyChart', [{ x: hrs, y: vals, type: 'bar', marker: { color: vals.map(v => v > 6 ? 'rgb(45,212,191)' : 'rgba(45,212,191,.4)'), line: { color: 'rgba(255,255,255,.15)', width: 1 } } }], { ...L, xaxis: { ...L.xaxis, title: 'Hour', tickmode: 'linear', dtick: 2 }, yaxis: { ...L.yaxis, title: 'Avg kWh' } }, C)
}

function cFeat(fd) { const s = Object.entries(fd).sort((a, b) => b[1] - a[1]); Plotly.newPlot('featChart', [{ x: s.map(([, v]) => v), y: s.map(([n]) => n.replace(/_/g, ' ')), type: 'bar', orientation: 'h', marker: { color: s.map(([, v]) => v), colorscale: [[0, 'rgb(45,212,191)'], [1, 'rgb(6,182,212)']], line: { width: 1, color: 'rgba(255,255,255,.1)' } } }], { ...L, margin: { ...L.margin, l: 140 }, xaxis: { ...L.xaxis, title: 'Importance' } }, C) }

function cCorr(data) { Plotly.newPlot('corrChart', [{ x: data.map(d => d.machines_active), y: data.map(d => d.electricity_kwh), mode: 'markers', marker: { color: 'rgba(45,212,191,.5)', size: 4 } }], { ...L, xaxis: { ...L.xaxis, title: 'Machines Active' }, yaxis: { ...L.yaxis, title: 'kWh' } }, C) }

function cHeat(data) { const dn = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], g = Array.from({ length: 7 }, () => Array(24).fill(0)), c = Array.from({ length: 7 }, () => Array(24).fill(0)); data.forEach(d => { const dt = new Date(d.timestamp), dw = dt.getDay() === 0 ? 6 : dt.getDay() - 1, h = dt.getHours(); g[dw][h] += d.electricity_kwh; c[dw][h]++ }); Plotly.newPlot('heatChart', [{ z: g.map((r, i) => r.map((v, j) => c[i][j] ? v / c[i][j] : 0)), x: Array.from({ length: 24 }, (_, i) => `${i}:00`), y: dn, type: 'heatmap', colorscale: [[0, 'rgb(10,20,20)'], [.5, 'rgb(45,212,191)'], [1, 'rgb(239,68,68)']], colorbar: { title: 'kWh', thickness: 10 } }], L, C) }

async function load() { const [h, p, s, pk] = await Promise.all([f('historical'), f('predict'), f('stats'), f('peak-analysis')]); if (p) uPred(p); if (pk) uPeak(pk); if (s) { uMetrics(s); cFeat(s.feature_importance) } if (h) { cTS(h.data); cHourly(h.data); cCorr(h.data); cHeat(h.data) } document.getElementById('lastUpdate').textContent = `Updated: ${new Date().toLocaleTimeString()}` }
document.addEventListener('DOMContentLoaded', () => { load(); setInterval(load, 30000) });
