const API = 'http://localhost:5005/api';
const L = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: 'Inter,sans-serif', color: '#e5e5e5', size: 12 }, xaxis: { gridcolor: 'rgba(255,255,255,.06)', zerolinecolor: 'rgba(255,255,255,.08)' }, yaxis: { gridcolor: 'rgba(255,255,255,.06)', zerolinecolor: 'rgba(255,255,255,.08)' }, margin: { t: 30, r: 30, b: 50, l: 60 }, hovermode: 'closest' };
const C = { responsive: true, displayModeBar: false };
async function f(ep) { try { const r = await fetch(`${API}/${ep}`); const d = await r.json(); if (d.success) return d; throw new Error(d.error) } catch (e) { console.error(ep, e); return null } }

function updatePrediction(d) {
    const el = document.getElementById('predictionContent'); if (!d) return; const { prediction: p, current: c } = d;
    const tag = c.is_night ? '<span class="night-tag night"><i class="fa-solid fa-moon"></i> Night</span>' : '<span class="night-tag day"><i class="fa-solid fa-sun"></i> Day</span>';
    el.innerHTML = `<div class="prediction-label">Forecasted Energy</div><div class="prediction-value">${p.prediction} <span style="font-size:1.2rem">kWh</span></div>${tag}
    <div style="font-size:.8rem;color:var(--text-muted);margin-top:.4rem"><i class="fa-solid fa-users"></i> ${c.users} users · ${c.floodlights ? '<i class="fa-solid fa-lightbulb"></i> Floodlights ON' : 'Floodlights OFF'}</div>
    <div class="confidence-bar"><div class="ci-item"><div class="ci-label">Lower</div><div class="ci-value">${p.lower_bound}</div></div><div class="ci-item"><div class="ci-label">Conf</div><div class="ci-value">${p.confidence_level}%</div></div><div class="ci-item"><div class="ci-label">Upper</div><div class="ci-value">${p.upper_bound}</div></div></div>`
}

function updateNight(d) {
    const el = document.getElementById('nightAnalysis'); if (!d) return; const a = d.analysis;
    el.innerHTML = `<div class="insight-grid">
    <div class="insight-item"><div class="il"><i class="fa-solid fa-moon"></i> Night Avg</div><div class="iv purple">${a.night_avg} kWh</div></div>
    <div class="insight-item"><div class="il"><i class="fa-solid fa-sun"></i> Day Avg</div><div class="iv">${a.day_avg} kWh</div></div>
    <div class="insight-item"><div class="il"><i class="fa-solid fa-lightbulb"></i> Floodlight Hrs</div><div class="iv">${a.floodlight_hours}</div></div>
    <div class="insight-item"><div class="il">🏆 Events</div><div class="iv">${a.event_count}</div></div></div>`
}

function updateMetrics(d) {
    const el = document.getElementById('modelMetrics'); if (!d) return; const m = d.metrics;
    el.innerHTML = `<div class="metrics-grid"><div class="metric-item"><div class="metric-label">R²</div><div class="metric-value good">${m.r2}</div></div><div class="metric-item"><div class="metric-label">RMSE</div><div class="metric-value">${m.rmse}<span class="metric-unit">kWh</span></div></div><div class="metric-item"><div class="metric-label">MAE</div><div class="metric-value">${m.mae}<span class="metric-unit">kWh</span></div></div><div class="metric-item"><div class="metric-label">MAPE</div><div class="metric-value">${m.mape}<span class="metric-unit">%</span></div></div></div>`
}

function chartTS(data) {
    const night = data.filter(d => d.is_night), day = data.filter(d => !d.is_night);
    Plotly.newPlot('timeSeriesChart', [
        { x: day.map(d => d.timestamp), y: day.map(d => d.electricity_kwh), type: 'scatter', mode: 'lines', name: 'Day', line: { color: 'rgb(234,179,8)', width: 1.5 } },
        { x: night.map(d => d.timestamp), y: night.map(d => d.electricity_kwh), type: 'scatter', mode: 'lines', name: 'Night', line: { color: 'rgb(168,85,247)', width: 2 } }
    ], { ...L, xaxis: { ...L.xaxis, title: 'Date' }, yaxis: { ...L.yaxis, title: 'Electricity (kWh)' }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C);
}

function chartHourly(data) {
    const wd = {}, we = {}; data.forEach(d => { const h = new Date(d.timestamp).getHours(), b = d.is_weekend ? we : wd; if (!b[h]) b[h] = { t: 0, c: 0 }; b[h].t += d.electricity_kwh; b[h].c++ });
    const hrs = Array.from({ length: 24 }, (_, i) => i);
    Plotly.newPlot('hourlyChart', [
        { x: hrs, y: hrs.map(h => wd[h] ? wd[h].t / wd[h].c : 0), type: 'bar', name: 'Weekday', marker: { color: 'rgba(168,85,247,.7)' } },
        { x: hrs, y: hrs.map(h => we[h] ? we[h].t / we[h].c : 0), type: 'bar', name: 'Weekend', marker: { color: 'rgba(234,179,8,.7)' } }
    ], { ...L, barmode: 'group', xaxis: { ...L.xaxis, title: 'Hour', tickmode: 'linear', dtick: 2 }, yaxis: { ...L.yaxis, title: 'Avg kWh' }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C);
}

function chartFeatures(fd) { const s = Object.entries(fd).sort((a, b) => b[1] - a[1]); Plotly.newPlot('featureChart', [{ x: s.map(([, v]) => v), y: s.map(([n]) => n.replace(/_/g, ' ')), type: 'bar', orientation: 'h', marker: { color: s.map(([, v]) => v), colorscale: [[0, 'rgb(168,85,247)'], [1, 'rgb(99,102,241)']], line: { width: 1, color: 'rgba(255,255,255,.1)' } } }], { ...L, margin: { ...L.margin, l: 140 }, xaxis: { ...L.xaxis, title: 'Importance' } }, C) }

function chartCorr(data) { const n = data.filter(d => d.is_night), dy = data.filter(d => !d.is_night); Plotly.newPlot('corrChart', [{ x: dy.map(d => d.users), y: dy.map(d => d.electricity_kwh), mode: 'markers', name: 'Day', marker: { color: 'rgba(234,179,8,.5)', size: 4 } }, { x: n.map(d => d.users), y: n.map(d => d.electricity_kwh), mode: 'markers', name: 'Night', marker: { color: 'rgba(168,85,247,.5)', size: 4 } }], { ...L, xaxis: { ...L.xaxis, title: 'Users' }, yaxis: { ...L.yaxis, title: 'kWh' }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C) }

function chartHeatmap(data) { const dn = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], g = Array.from({ length: 7 }, () => Array(24).fill(0)), c = Array.from({ length: 7 }, () => Array(24).fill(0)); data.forEach(d => { const dt = new Date(d.timestamp), dw = dt.getDay() === 0 ? 6 : dt.getDay() - 1, h = dt.getHours(); g[dw][h] += d.electricity_kwh; c[dw][h]++ }); Plotly.newPlot('heatmapChart', [{ z: g.map((r, i) => r.map((v, j) => c[i][j] ? v / c[i][j] : 0)), x: Array.from({ length: 24 }, (_, i) => `${i}:00`), y: dn, type: 'heatmap', colorscale: [[0, 'rgb(20,15,35)'], [.5, 'rgb(168,85,247)'], [1, 'rgb(234,179,8)']], colorbar: { title: 'kWh', thickness: 10 } }], L, C) }

async function load() { const [h, p, s, n] = await Promise.all([f('historical'), f('predict'), f('stats'), f('night-analysis')]); if (p) updatePrediction(p); if (n) updateNight(n); if (s) { updateMetrics(s); chartFeatures(s.feature_importance) } if (h) { chartTS(h.data); chartHourly(h.data); chartCorr(h.data); chartHeatmap(h.data) } document.getElementById('lastUpdate').textContent = `Updated: ${new Date().toLocaleTimeString()}` }
document.addEventListener('DOMContentLoaded', () => { load(); setInterval(load, 30000) });
