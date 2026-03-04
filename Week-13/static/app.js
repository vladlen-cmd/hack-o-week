const API = 'http://localhost:5012/api';
const L = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: 'Inter,sans-serif', color: '#e5e5e5', size: 12 }, xaxis: { gridcolor: 'rgba(255,255,255,.06)' }, yaxis: { gridcolor: 'rgba(255,255,255,.06)' }, margin: { t: 30, r: 30, b: 50, l: 60 }, hovermode: 'closest' };
const C = { responsive: true, displayModeBar: false };
const COL = { energy: 'rgb(192,132,252)', network: 'rgb(56,189,248)', footfall: 'rgb(52,211,153)', aqi: 'rgb(251,191,36)', event: 'rgb(244,63,94)' };
async function f(ep) { try { const r = await fetch(`${API}/${ep}`); const d = await r.json(); if (d.success) return d; throw new Error(d.error) } catch (e) { console.error(ep, e); return null } }

function uKPI(d) {
    const el = document.getElementById('kpiCards'); if (!d) return; const s = d.summary;
    const et = s.energy_trend, at = s.aqi_trend;
    el.innerHTML = `
    <div class="card kpi-card"><div class="ki"><i class="fa-solid fa-bolt"></i></div><div class="kv" style="color:${COL.energy}">${s.avg_energy}</div><div class="kl">Avg Energy (kWh)</div><div class="kt ${et < 0 ? 'up' : 'down'}">${et > 0 ? '↑' : '↓'}${Math.abs(et)} 7d trend</div></div>
    <div class="card kpi-card"><div class="ki"><i class="fa-solid fa-globe"></i></div><div class="kv" style="color:${COL.network}">${s.avg_bandwidth}</div><div class="kl">Avg Bandwidth (Mbps)</div></div>
    <div class="card kpi-card"><div class="ki"><i class="fa-solid fa-person-walking"></i></div><div class="kv" style="color:${COL.footfall}">${s.avg_footfall.toLocaleString()}</div><div class="kl">Avg Footfall</div></div>
    <div class="card kpi-card"><div class="ki"><i class="fa-solid fa-wind"></i></div><div class="kv" style="color:${COL.aqi}">${s.avg_aqi}</div><div class="kl">Avg AQI</div><div class="kt ${at < 0 ? 'up' : 'down'}">${at > 0 ? '↑' : '↓'}${Math.abs(at)} 7d</div></div>
    <div class="card kpi-card"><div class="ki"><i class="fa-solid fa-calendar-check"></i></div><div class="kv" style="color:${COL.event}">${s.total_events}</div><div class="kl">Total Events</div></div>`
}

function cTrend(d) {
    const daily = d.daily;
    Plotly.newPlot('trendChart', [
        { x: daily.map(r => r.date), y: daily.map(r => r.energy_kwh), type: 'scatter', mode: 'lines', name: 'Energy (kWh)', line: { color: COL.energy, width: 2 } },
        { x: daily.map(r => r.date), y: daily.map(r => r.footfall), type: 'scatter', mode: 'lines', name: 'Footfall', yaxis: 'y2', line: { color: COL.footfall, width: 1.5, dash: 'dot' } }
    ], { ...L, xaxis: { ...L.xaxis, title: 'Date' }, yaxis: { ...L.yaxis, title: 'Energy (kWh)' }, yaxis2: { title: 'Footfall', overlaying: 'y', side: 'right', gridcolor: 'rgba(0,0,0,0)', titlefont: { color: COL.footfall }, tickfont: { color: COL.footfall } }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C)
}

function cBW(d) {
    const h = d.hourly;
    Plotly.newPlot('bwChart', [{ x: h.map(r => r.hour), y: h.map(r => r.bandwidth_mbps), type: 'scatter', mode: 'lines+markers', line: { color: COL.network, width: 2.5, shape: 'spline' }, marker: { size: 5 }, fill: 'tozeroy', fillcolor: 'rgba(56,189,248,.1)' }], { ...L, xaxis: { ...L.xaxis, title: 'Hour', tickmode: 'linear', dtick: 2 }, yaxis: { ...L.yaxis, title: 'Bandwidth (Mbps)' } }, C)
}

function cAQI(d) {
    const h = d.hourly;
    const colors = h.map(r => r.aqi <= 50 ? 'rgb(52,211,153)' : r.aqi <= 100 ? 'rgb(251,191,36)' : 'rgb(239,68,68)');
    Plotly.newPlot('aqiChart', [{ x: h.map(r => r.hour), y: h.map(r => r.aqi), type: 'bar', marker: { color: colors, line: { color: 'rgba(255,255,255,.1)', width: 1 } } }], { ...L, xaxis: { ...L.xaxis, title: 'Hour', tickmode: 'linear', dtick: 2 }, yaxis: { ...L.yaxis, title: 'AQI' } }, C)
}

function cHeatmap(data) {
    const dn = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], g = Array.from({ length: 7 }, () => Array(24).fill(0)), c = Array.from({ length: 7 }, () => Array(24).fill(0));
    data.forEach(d => { const dt = new Date(d.timestamp), dw = dt.getDay() === 0 ? 6 : dt.getDay() - 1, h = dt.getHours(); g[dw][h] += d.footfall; c[dw][h]++ });
    Plotly.newPlot('heatChart', [{ z: g.map((r, i) => r.map((v, j) => c[i][j] ? Math.round(v / c[i][j]) : 0)), x: Array.from({ length: 24 }, (_, i) => `${i}:00`), y: dn, type: 'heatmap', colorscale: [[0, 'rgb(15,20,35)'], [.5, 'rgb(52,211,153)'], [1, 'rgb(239,68,68)']], colorbar: { title: 'Footfall', thickness: 10 } }], L, C)
}

function cCorr(data) {
    const vars = ['energy_kwh', 'bandwidth_mbps', 'footfall', 'aqi'];
    const labels = ['Energy', 'Bandwidth', 'Footfall', 'AQI'];
    const vals = vars.map(v => data.map(d => d[v]));
    const n = vars.length; const mat = [];
    for (let i = 0; i < n; i++) {
        mat[i] = []; for (let j = 0; j < n; j++) {
            const x = vals[i], y = vals[j]; const mx = x.reduce((a, b) => a + b, 0) / x.length, my = y.reduce((a, b) => a + b, 0) / y.length;
            let num = 0, dx = 0, dy = 0; for (let k = 0; k < x.length; k++) { num += (x[k] - mx) * (y[k] - my); dx += (x[k] - mx) ** 2; dy += (y[k] - my) ** 2 }
            mat[i][j] = dx && dy ? +(num / Math.sqrt(dx * dy)).toFixed(2) : 0
        }
    }
    Plotly.newPlot('corrChart', [{ z: mat, x: labels, y: labels, type: 'heatmap', colorscale: [[0, 'rgb(59,130,246)'], [.5, 'rgb(15,20,35)'], [1, 'rgb(239,68,68)']], zmin: -1, zmax: 1, text: mat.map(r => r.map(v => v.toFixed(2))), texttemplate: '%{text}', colorbar: { title: 'r', thickness: 10 } }], { ...L, margin: { ...L.margin, l: 100 } }, C)
}

function cEvents(d) {
    const evts = d.daily.filter(r => r.event_attendance > 0);
    Plotly.newPlot('eventChart', [{ x: evts.map(r => r.date), y: evts.map(r => r.event_attendance), type: 'bar', marker: { color: COL.event, line: { color: 'rgba(255,255,255,.1)', width: 1 } } }], { ...L, xaxis: { ...L.xaxis, title: 'Date' }, yaxis: { ...L.yaxis, title: 'Attendance' } }, C)
}

async function load() {
    const [su, hr, dl, dt] = await Promise.all([f('summary'), f('hourly'), f('daily'), f('data')]);
    if (su) uKPI(su); if (dl) { cTrend(dl); cEvents(dl) }
    if (hr) { cBW(hr); cAQI(hr) }
    if (dt) { cHeatmap(dt.data); cCorr(dt.data) }
    document.getElementById('lastUpdate').textContent = `Updated: ${new Date().toLocaleTimeString()}`
}
document.addEventListener('DOMContentLoaded', () => { load(); setInterval(load, 30000) });
