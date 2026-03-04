const API = 'http://localhost:5004/api';
const L = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: 'Inter,sans-serif', color: '#e5e5e5', size: 12 }, xaxis: { gridcolor: 'rgba(255,255,255,.06)' }, yaxis: { gridcolor: 'rgba(255,255,255,.06)' }, margin: { t: 30, r: 30, b: 50, l: 60 }, hovermode: 'closest' };
const C = { responsive: true, displayModeBar: false };
const ZC = ['rgb(56,189,248)', 'rgb(52,211,153)', 'rgb(244,63,94)', 'rgb(250,204,21)'];
async function f(ep) { try { const r = await fetch(`${API}/${ep}`); return await r.json() } catch (e) { return null } }

function uKPI(d) {
    const el = document.getElementById('kpiCards'); if (!d || !d.success) return; const m = d.metrics, s = d.summary;
    el.innerHTML = `
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-chart-line"></i></div><div class="kv">${m.r2}</div><div class="kl">R² Score</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-snowflake"></i></div><div class="kv">${s.avg_cooling}</div><div class="kl">Avg Cooling (kWh)</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-layer-group"></i></div><div class="kv">${s.zones}</div><div class="kl">Active Zones</div></div>`;
}

function cHeat(d) {
    if (!d || !d.success) return;
    Plotly.newPlot('heatChart', [{
        z: d.matrix, x: d.hours.map(h => `${h}:00`), y: d.zones, type: 'heatmap',
        colorscale: [[0, 'rgb(10,20,40)'], [0.5, 'rgb(56,189,248)'], [1, 'rgb(239,68,68)']],
        colorbar: { title: 'kWh', thickness: 10 }
    }], { ...L, margin: { ...L.margin, l: 120 } }, C);
}

function cFI(d) {
    if (!d || !d.metrics) return; const fi = d.metrics.feature_importance; const keys = Object.keys(fi), vals = keys.map(k => fi[k]);
    const idx = [...vals.keys()].sort((a, b) => vals[a] - vals[b]);
    Plotly.newPlot('fiChart', [{
        x: idx.map(i => vals[i]), y: idx.map(i => keys[i]), type: 'bar', orientation: 'h',
        marker: { color: 'rgb(56,189,248)' }
    }], { ...L, margin: { ...L.margin, l: 140 } }, C);
}

function cTemp(data) {
    Plotly.newPlot('tempChart', [{
        x: data.map(d => d.outdoor_temp_c), y: data.map(d => d.cooling_kwh),
        type: 'scatter', mode: 'markers', marker: { size: 2, color: 'rgba(56,189,248,.25)' }
    }],
        { ...L, xaxis: { ...L.xaxis, title: 'Outdoor Temp (°C)' }, yaxis: { ...L.yaxis, title: 'Cooling (kWh)' } }, C);
}

function cOcc(data) {
    Plotly.newPlot('occChart', [{
        x: data.map(d => d.occupancy), y: data.map(d => d.cooling_kwh),
        type: 'scatter', mode: 'markers', marker: { size: 2, color: 'rgba(52,211,153,.25)' }
    }],
        { ...L, xaxis: { ...L.xaxis, title: 'Occupancy' }, yaxis: { ...L.yaxis, title: 'Cooling (kWh)' } }, C);
}

function cHourly(data) {
    const zones = [...new Set(data.map(d => d.zone_name))];
    const traces = zones.map((z, i) => {
        const zd = data.filter(d => d.zone_name === z); const h = {};
        zd.forEach(d => { if (!h[d.hour]) h[d.hour] = { t: 0, c: 0 }; h[d.hour].t += d.cooling_kwh; h[d.hour].c++ });
        const hrs = Array.from({ length: 24 }, (_, i) => i);
        return { x: hrs, y: hrs.map(hr => h[hr] ? +(h[hr].t / h[hr].c).toFixed(2) : 0), type: 'scatter', mode: 'lines', name: z, line: { color: ZC[i], width: 2 } };
    });
    Plotly.newPlot('hourlyChart', traces, { ...L, xaxis: { ...L.xaxis, title: 'Hour', tickmode: 'linear', dtick: 2 }, yaxis: { ...L.yaxis, title: 'Avg kWh' }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C);
}

async function load() {
    const [st, hm, hist] = await Promise.all([f('stats'), f('heatmap'), f('historical')]);
    if (st) { uKPI(st); cFI(st); } if (hm) cHeat(hm);
    if (hist && hist.data) { cTemp(hist.data); cOcc(hist.data); cHourly(hist.data); }
}
document.addEventListener('DOMContentLoaded', load);
