const API = 'http://localhost:5007/api';
const L = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: 'Inter,sans-serif', color: '#e5e5e5', size: 12 }, xaxis: { gridcolor: 'rgba(255,255,255,.06)' }, yaxis: { gridcolor: 'rgba(255,255,255,.06)' }, margin: { t: 30, r: 30, b: 50, l: 60 }, hovermode: 'closest' };
const C = { responsive: true, displayModeBar: false };
async function f(ep) { try { const r = await fetch(`${API}/${ep}`); const d = await r.json(); if (d.success) return d; throw new Error(d.error) } catch (e) { console.error(ep, e); return null } }

function uPred(d) {
    const el = document.getElementById('pred'); if (!d) return; const { prediction: p, current: c } = d;
    el.innerHTML = `<div class="prediction-label">Forecasted Lighting Load</div><div class="prediction-value">${p.prediction} <span style="font-size:1.2rem">kWh</span></div>
    <div style="font-size:.8rem;color:var(--text-muted);margin-top:.25rem">${c.is_dark ? '<i class="fa-solid fa-moon"></i> Dark' : '<i class="fa-solid fa-sun"></i> Light'} · <i class="fa-solid fa-car"></i> ${c.vehicles} vehicles${c.has_event ? ' · <i class="fa-solid fa-party-horn"></i> Event' : ''}</div>
    <div class="confidence-bar"><div class="ci-item"><div class="ci-label">Lower</div><div class="ci-value">${p.lower_bound}</div></div><div class="ci-item"><div class="ci-label">Conf</div><div class="ci-value">${p.confidence_level}%</div></div><div class="ci-item"><div class="ci-label">Upper</div><div class="ci-value">${p.upper_bound}</div></div></div>`
}

function uLight(d) {
    const el = document.getElementById('lightAnalysis'); if (!d) return; const a = d.analysis;
    el.innerHTML = `<div class="insight-grid">
    <div class="insight-item"><div class="il"><i class="fa-solid fa-moon"></i> Dark Avg</div><div class="iv gold">${a.dark_avg} kWh</div></div>
    <div class="insight-item"><div class="il"><i class="fa-solid fa-sun"></i> Light Avg</div><div class="iv">${a.light_avg} kWh</div></div>
    <div class="insight-item"><div class="il">Dark/Light Ratio</div><div class="iv gold">${a.ratio}×</div></div>
    <div class="insight-item"><div class="il">Dark Hours</div><div class="iv">${a.dark_hours_pct}%</div></div></div>`
}

function uMetrics(d) {
    const el = document.getElementById('metrics'); if (!d) return; const m = d.metrics;
    el.innerHTML = `<div class="metrics-grid"><div class="metric-item"><div class="metric-label">R²</div><div class="metric-value good">${m.r2}</div></div><div class="metric-item"><div class="metric-label">RMSE</div><div class="metric-value">${m.rmse}<span class="metric-unit">kWh</span></div></div><div class="metric-item"><div class="metric-label">MAE</div><div class="metric-value">${m.mae}<span class="metric-unit">kWh</span></div></div><div class="metric-item"><div class="metric-label">MAPE</div><div class="metric-value">${m.mape}<span class="metric-unit">%</span></div></div></div>`
}

function cTS(data) {
    const dk = data.filter(d => d.is_dark), lt = data.filter(d => !d.is_dark);
    Plotly.newPlot('tsChart', [{ x: lt.map(d => d.timestamp), y: lt.map(d => d.electricity_kwh), type: 'scatter', mode: 'lines', name: 'Light', line: { color: 'rgb(250,204,21)', width: 1.5 } }, { x: dk.map(d => d.timestamp), y: dk.map(d => d.electricity_kwh), type: 'scatter', mode: 'lines', name: 'Dark', line: { color: 'rgb(99,102,241)', width: 2 } }], { ...L, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' }, xaxis: { ...L.xaxis, title: 'Date' }, yaxis: { ...L.yaxis, title: 'kWh' } }, C)
}

function cHourly(data) {
    const hr = {}; data.forEach(d => { const h = new Date(d.timestamp).getHours(); if (!hr[h]) hr[h] = { t: 0, c: 0 }; hr[h].t += d.electricity_kwh; hr[h].c++ });
    const hrs = Array.from({ length: 24 }, (_, i) => i), vals = hrs.map(h => hr[h] ? hr[h].t / hr[h].c : 0);
    Plotly.newPlot('hourlyChart', [{ x: hrs, y: vals, type: 'bar', marker: { color: vals.map(v => v > 8 ? 'rgb(250,204,21)' : 'rgb(99,102,241)'), line: { color: 'rgba(255,255,255,.15)', width: 1 } } }], { ...L, xaxis: { ...L.xaxis, title: 'Hour', tickmode: 'linear', dtick: 2 }, yaxis: { ...L.yaxis, title: 'Avg kWh' } }, C)
}

function cFeat(fd) { const s = Object.entries(fd).sort((a, b) => b[1] - a[1]); Plotly.newPlot('featChart', [{ x: s.map(([, v]) => v), y: s.map(([n]) => n.replace(/_/g, ' ')), type: 'bar', orientation: 'h', marker: { color: s.map(([, v]) => v), colorscale: [[0, 'rgb(250,204,21)'], [1, 'rgb(245,158,11)']], line: { width: 1, color: 'rgba(255,255,255,.1)' } } }], { ...L, margin: { ...L.margin, l: 140 }, xaxis: { ...L.xaxis, title: 'Importance' } }, C) }

function cCorr(data) { const dk = data.filter(d => d.is_dark), lt = data.filter(d => !d.is_dark); Plotly.newPlot('corrChart', [{ x: lt.map(d => d.vehicles), y: lt.map(d => d.electricity_kwh), mode: 'markers', name: 'Light', marker: { color: 'rgba(250,204,21,.5)', size: 4 } }, { x: dk.map(d => d.vehicles), y: dk.map(d => d.electricity_kwh), mode: 'markers', name: 'Dark', marker: { color: 'rgba(99,102,241,.5)', size: 4 } }], { ...L, xaxis: { ...L.xaxis, title: 'Vehicles' }, yaxis: { ...L.yaxis, title: 'kWh' }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C) }

function cHeat(data) { const dn = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], g = Array.from({ length: 7 }, () => Array(24).fill(0)), c = Array.from({ length: 7 }, () => Array(24).fill(0)); data.forEach(d => { const dt = new Date(d.timestamp), dw = dt.getDay() === 0 ? 6 : dt.getDay() - 1, h = dt.getHours(); g[dw][h] += d.electricity_kwh; c[dw][h]++ }); Plotly.newPlot('heatChart', [{ z: g.map((r, i) => r.map((v, j) => c[i][j] ? v / c[i][j] : 0)), x: Array.from({ length: 24 }, (_, i) => `${i}:00`), y: dn, type: 'heatmap', colorscale: [[0, 'rgb(20,15,5)'], [.5, 'rgb(250,204,21)'], [1, 'rgb(239,68,68)']], colorbar: { title: 'kWh', thickness: 10 } }], L, C) }

async function load() { const [h, p, s, la] = await Promise.all([f('historical'), f('predict'), f('stats'), f('lighting-analysis')]); if (p) uPred(p); if (la) uLight(la); if (s) { uMetrics(s); cFeat(s.feature_importance) } if (h) { cTS(h.data); cHourly(h.data); cCorr(h.data); cHeat(h.data) } document.getElementById('lastUpdate').textContent = `Updated: ${new Date().toLocaleTimeString()}` }
document.addEventListener('DOMContentLoaded', () => { load(); setInterval(load, 30000) });
