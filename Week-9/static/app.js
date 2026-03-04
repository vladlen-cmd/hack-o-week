const API = 'http://localhost:5008/api';
const L = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: 'Inter,sans-serif', color: '#e5e5e5', size: 12 }, xaxis: { gridcolor: 'rgba(255,255,255,.06)' }, yaxis: { gridcolor: 'rgba(255,255,255,.06)' }, margin: { t: 30, r: 30, b: 50, l: 60 }, hovermode: 'closest' };
const C = { responsive: true, displayModeBar: false };
const AC = 'rgb(45,212,191)';
let allData = [];
async function f(ep) { try { const r = await fetch(`${API}/${ep}`); return await r.json() } catch (e) { return null } }

function uKPI(d) {
    const el = document.getElementById('kpiCards'); if (!d) return;
    el.innerHTML = `
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-brain"></i></div><div class="kv">${d.nb_accuracy}</div><div class="kl">NaiveBayes Accuracy</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-bolt"></i></div><div class="kv">${d.summary.avg_load}</div><div class="kl">Avg Load (kWh)</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-calendar-day"></i></div><div class="kv">${d.summary.sunday_avg}</div><div class="kl">Sunday Avg (kWh)</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-gear"></i></div><div class="kv">${d.engine}</div><div class="kl">Forecast Engine</div></div>`;
}

function cForecast(d) {
    if (!d || !d.forecast) return; const fc = d.forecast;
    Plotly.newPlot('forecastChart', [
        { x: fc.map(f => f.ds), y: fc.map(f => f.yhat), type: 'scatter', mode: 'lines+markers', name: 'Forecast', line: { color: AC, width: 2 }, marker: { size: 4 } },
        { x: fc.map(f => f.ds), y: fc.map(f => f.yhat_upper), type: 'scatter', mode: 'lines', name: 'Upper', line: { color: 'rgba(45,212,191,.3)', width: 1 }, showlegend: false },
        { x: fc.map(f => f.ds), y: fc.map(f => f.yhat_lower), type: 'scatter', mode: 'lines', name: 'Lower', line: { color: 'rgba(45,212,191,.3)', width: 1 }, fill: 'tonexty', fillcolor: 'rgba(45,212,191,.08)', showlegend: false }
    ], { ...L, xaxis: { ...L.xaxis, title: 'Date' }, yaxis: { ...L.yaxis, title: 'Daily Load (kWh)' }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C);
}

function cSlider(data) {
    Plotly.newPlot('sliderChart', [{ x: data.map(d => d.timestamp), y: data.map(d => d.load_kwh), type: 'scatter', mode: 'lines', line: { color: AC, width: 1 } }], { ...L, xaxis: { ...L.xaxis, title: 'Time' }, yaxis: { ...L.yaxis, title: 'Load (kWh)' } }, C);
}

function onSlider(v) {
    const days = parseInt(v); const total = allData.length; const cutoff = Math.floor(total * (days / 90));
    const sub = allData.slice(0, Math.max(cutoff, 24));
    document.getElementById('sliderVal').textContent = days < 90 ? `First ${days} days` : 'All data';
    cSlider(sub);
}

function cHourly(data) {
    const h = {}; data.forEach(d => { if (!h[d.hour]) h[d.hour] = { t: 0, c: 0 }; h[d.hour].t += d.load_kwh; h[d.hour].c++ });
    const hrs = Array.from({ length: 24 }, (_, i) => i), vals = hrs.map(i => h[i] ? +(h[i].t / h[i].c).toFixed(2) : 0);
    Plotly.newPlot('hourlyChart', [{ x: hrs, y: vals, type: 'bar', marker: { color: vals.map(v => v > 8 ? AC : 'rgba(45,212,191,.35)') } }], { ...L, xaxis: { ...L.xaxis, title: 'Hour', tickmode: 'linear', dtick: 2 }, yaxis: { ...L.yaxis, title: 'Avg kWh' } }, C);
}

function cDaily(data) {
    const dn = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']; const d = {};
    data.forEach(r => { const dw = r.day_of_week; if (!d[dw]) d[dw] = { t: 0, c: 0 }; d[dw].t += r.load_kwh; d[dw].c++ });
    Plotly.newPlot('dailyChart', [{ x: dn, y: Array.from({ length: 7 }, (_, i) => d[i] ? +(d[i].t / d[i].c).toFixed(2) : 0), type: 'bar', marker: { color: Array.from({ length: 7 }, (_, i) => i === 6 ? AC : 'rgba(45,212,191,.4)') } }], { ...L, yaxis: { ...L.yaxis, title: 'Avg kWh' } }, C);
}

async function load() {
    const [st, fc, hist] = await Promise.all([f('stats'), f('forecast'), f('historical')]);
    if (st) uKPI(st); if (fc) cForecast(fc);
    if (hist && hist.data) { allData = hist.data; cSlider(hist.data); cHourly(hist.data); cDaily(hist.data); }
}
document.addEventListener('DOMContentLoaded', load);
