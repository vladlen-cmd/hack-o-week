const API = 'http://localhost:5005/api';
const L = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: 'Inter,sans-serif', color: '#e5e5e5', size: 12 }, xaxis: { gridcolor: 'rgba(255,255,255,.06)' }, yaxis: { gridcolor: 'rgba(255,255,255,.06)' }, margin: { t: 30, r: 30, b: 50, l: 60 }, hovermode: 'closest' };
const C = { responsive: true, displayModeBar: false };
const DTC = { weekday: 'rgb(168,85,247)', weekend: 'rgb(52,211,153)', event: 'rgb(244,63,94)' };
async function f(ep) { try { const r = await fetch(`${API}/${ep}`); return await r.json() } catch (e) { return null } }

function uKPI(d) {
    const el = document.getElementById('kpiCards'); if (!d) return; const m = d.metrics, da = d.day_analysis;
    el.innerHTML = `
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-chart-line"></i></div><div class="kv">${m.r2}</div><div class="kl">R² (${m.engine})</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-calendar"></i></div><div class="kv">${da.weekday?.avg || 0}</div><div class="kl">Weekday Avg kWh</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-futbol"></i></div><div class="kv">${da.event?.avg || 0}</div><div class="kl">Event Avg kWh</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-moon"></i></div><div class="kv">${da.weekend?.avg || 0}</div><div class="kl">Weekend Avg kWh</div></div>`;
}

function cTime(data) {
    const colors = data.map(d => DTC[d.day_type] || '#888');
    Plotly.newPlot('timeChart', [{ x: data.map(d => d.timestamp), y: data.map(d => d.energy_kwh), type: 'scatter', mode: 'lines', line: { color: 'rgb(168,85,247)', width: 1 } }], { ...L, xaxis: { ...L.xaxis, title: 'Time' }, yaxis: { ...L.yaxis, title: 'Energy (kWh)' } }, C);
}

function cDay(da) {
    const types = Object.keys(da), vals = types.map(t => da[t].avg);
    Plotly.newPlot('dayChart', [{ labels: types, values: vals, type: 'pie', marker: { colors: types.map(t => DTC[t]) }, textinfo: 'label+percent', hole: .4 }], { ...L, showlegend: false }, C);
}

function cHourly(data) {
    const types = ['weekday', 'weekend', 'event'];
    const traces = types.map(t => {
        const td = data.filter(d => d.day_type === t); const h = {};
        td.forEach(d => { if (!h[d.hour]) h[d.hour] = { t: 0, c: 0 }; h[d.hour].t += d.energy_kwh; h[d.hour].c++ });
        const hrs = Array.from({ length: 24 }, (_, i) => i);
        return { x: hrs, y: hrs.map(hr => h[hr] ? +(h[hr].t / h[hr].c).toFixed(2) : 0), type: 'scatter', mode: 'lines+markers', name: t, line: { color: DTC[t], width: 2 }, marker: { size: 3 } };
    });
    Plotly.newPlot('hourlyChart', traces, { ...L, xaxis: { ...L.xaxis, title: 'Hour', tickmode: 'linear', dtick: 2 }, yaxis: { ...L.yaxis, title: 'Avg kWh' }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C);
}

function cNight(data) {
    const night = data.filter(d => d.hour >= 18 || d.hour < 6), day = data.filter(d => d.hour >= 6 && d.hour < 18);
    Plotly.newPlot('nightChart', [
        { y: night.map(d => d.energy_kwh), type: 'box', name: 'Night (18-6h)', marker: { color: 'rgb(168,85,247)' } },
        { y: day.map(d => d.energy_kwh), type: 'box', name: 'Day (6-18h)', marker: { color: 'rgba(168,85,247,.3)' } }
    ], L, C);
}

function cEvent(data) {
    const evts = data.filter(d => d.is_event); const h = {};
    evts.forEach(d => { if (!h[d.hour]) h[d.hour] = { t: 0, c: 0 }; h[d.hour].t += d.energy_kwh; h[d.hour].c++ });
    const hrs = Array.from({ length: 24 }, (_, i) => i);
    Plotly.newPlot('eventChart', [{ x: hrs, y: hrs.map(hr => h[hr] ? +(h[hr].t / h[hr].c).toFixed(2) : 0), type: 'bar', marker: { color: hrs.map(h => (h >= 18 && h <= 22) ? 'rgb(244,63,94)' : 'rgba(244,63,94,.3)') } }], { ...L, xaxis: { ...L.xaxis, title: 'Hour', tickmode: 'linear', dtick: 2 }, yaxis: { ...L.yaxis, title: 'Event Avg kWh' } }, C);
}

async function filterData() {
    const dt = document.getElementById('dayFilter').value;
    const ep = dt ? `historical?day_type=${dt}` : 'historical';
    const d = await f(ep); if (d && d.data) cTime(d.data);
}

async function load() {
    const [st, hist] = await Promise.all([f('stats'), f('historical')]);
    if (st) { uKPI(st); cDay(st.day_analysis); }
    if (hist && hist.data) { cTime(hist.data); cHourly(hist.data); cNight(hist.data); cEvent(hist.data); }
}
document.addEventListener('DOMContentLoaded', load);
