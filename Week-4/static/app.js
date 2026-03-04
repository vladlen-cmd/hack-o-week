const API = 'http://localhost:5003/api';
const L = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: 'Inter,sans-serif', color: '#e5e5e5', size: 12 }, xaxis: { gridcolor: 'rgba(255,255,255,.06)' }, yaxis: { gridcolor: 'rgba(255,255,255,.06)' }, margin: { t: 30, r: 30, b: 50, l: 60 }, hovermode: 'closest' };
const C = { responsive: true, displayModeBar: false };
const AC = 'rgb(251,191,36)';
async function f(ep) { try { const r = await fetch(`${API}/${ep}`); return await r.json() } catch (e) { return null } }

const socket = io('http://localhost:5003');
let lA = [], lP = [], lT = [], MX = 120;
socket.on('connect', () => { document.getElementById('wsBadge').innerHTML = '<span class="status-dot"></span> Live'; });
socket.on('disconnect', () => { document.getElementById('wsBadge').innerHTML = '<span class="status-dot off"></span> Offline'; });
socket.on('live_data', (d) => {
    lT.push(d.timestamp); lA.push(d.actual); lP.push(d.predicted);
    if (lT.length > MX) { lT.shift(); lA.shift(); lP.shift(); }
    Plotly.react('liveChart', [
        { x: lT, y: lA, type: 'scatter', mode: 'lines', name: 'Actual', line: { color: AC, width: 2 } },
        { x: lT, y: lP, type: 'scatter', mode: 'lines', name: 'Predicted', line: { color: 'rgb(253,224,71)', width: 2, dash: 'dot' } }
    ], { ...L, xaxis: { ...L.xaxis, title: 'Time' }, yaxis: { ...L.yaxis, title: 'Energy (kWh)' }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C);
    document.getElementById('lastUpdate').textContent = 'Updated: ' + new Date().toLocaleTimeString();
});

function uKPI(d) {
    const el = document.getElementById('kpiCards'); if (!d || !d.success) return; const m = d.metrics, s = d.summary;
    el.innerHTML = `
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-chart-line"></i></div><div class="kv">${m.r2}</div><div class="kl">R² Score</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-bolt"></i></div><div class="kv">${s.avg_energy}</div><div class="kl">Avg Energy (kWh)</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-sun"></i></div><div class="kv">${s.lunch_surge_avg}</div><div class="kl">Lunch Surge Avg</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-person-walking"></i></div><div class="kv">${s.avg_footfall}</div><div class="kl">Avg Footfall</div></div>`;
}

function cSurge(data) {
    const lunch = data.filter(d => d.hour >= 11 && d.hour <= 14 && !d.is_weekend);
    const other = data.filter(d => !(d.hour >= 11 && d.hour <= 14) || d.is_weekend);
    Plotly.newPlot('surgeChart', [
        { y: lunch.map(d => d.energy_kwh), type: 'box', name: 'Lunch (11-14h)', marker: { color: AC } },
        { y: other.map(d => d.energy_kwh), type: 'box', name: 'Other Hours', marker: { color: 'rgba(251,191,36,.3)' } }
    ], { ...L, showlegend: false }, C);
}

function cCoef(d) {
    if (!d || !d.metrics) return; const c = d.metrics.coefficients; const keys = Object.keys(c), vals = keys.map(k => c[k]);
    Plotly.newPlot('coefChart', [{ x: vals, y: keys, type: 'bar', orientation: 'h', marker: { color: vals.map(v => v >= 0 ? AC : 'rgb(239,68,68)') } }], { ...L, margin: { ...L.margin, l: 120 } }, C);
}

function cFoot(data) { Plotly.newPlot('footChart', [{ x: data.map(d => d.footfall), y: data.map(d => d.energy_kwh), type: 'scatter', mode: 'markers', marker: { size: 3, color: 'rgba(251,191,36,.3)' } }], { ...L, xaxis: { ...L.xaxis, title: 'Footfall' }, yaxis: { ...L.yaxis, title: 'Energy (kWh)' } }, C); }

function cHourly(data) {
    const h = {}; data.forEach(d => { if (!h[d.hour]) h[d.hour] = { t: 0, c: 0 }; h[d.hour].t += d.energy_kwh; h[d.hour].c++ });
    const hrs = Array.from({ length: 24 }, (_, i) => i), vals = hrs.map(i => h[i] ? +(h[i].t / h[i].c).toFixed(2) : 0);
    Plotly.newPlot('hourlyChart', [{ x: hrs, y: vals, type: 'bar', marker: { color: hrs.map(i => (i >= 11 && i <= 14) ? AC : 'rgba(251,191,36,.35)') } }], { ...L, xaxis: { ...L.xaxis, title: 'Hour', tickmode: 'linear', dtick: 2 }, yaxis: { ...L.yaxis, title: 'Avg kWh' } }, C);
}

async function load() {
    const [st, hist] = await Promise.all([f('stats'), f('historical')]);
    if (st) { uKPI(st); cCoef(st); } if (hist && hist.data) { cSurge(hist.data); cFoot(hist.data); cHourly(hist.data); }
}
document.addEventListener('DOMContentLoaded', load);
