const API = 'http://localhost:5006/api';
const L = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: 'Inter,sans-serif', color: '#e5e5e5', size: 12 }, xaxis: { gridcolor: 'rgba(255,255,255,.06)' }, yaxis: { gridcolor: 'rgba(255,255,255,.06)' }, margin: { t: 30, r: 30, b: 50, l: 60 }, hovermode: 'closest' };
const C = { responsive: true, displayModeBar: false };
const CC = { high_usage: 'rgb(239,68,68)', medium_usage: 'rgb(148,163,184)', low_usage: 'rgb(52,211,153)' };
async function f(ep) { try { const r = await fetch(`${API}/${ep}`); return await r.json() } catch (e) { return null } }

function uKPI(d) {
    const el = document.getElementById('kpiCards'); if (!d) return; const m = d.metrics;
    el.innerHTML = `
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-chart-line"></i></div><div class="kv">${m.r2}</div><div class="kl">R² Score</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-layer-group"></i></div><div class="kv">${m.n_clusters}</div><div class="kl">Usage Clusters</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-piggy-bank"></i></div><div class="kv">${d.potential_kwh}</div><div class="kl">Savings Potential (kWh)</div></div>`;
}

function cPie(d) {
    if (!d || !d.savings) return; const s = d.savings, keys = Object.keys(s);
    Plotly.newPlot('pieChart', [{
        labels: keys.map(k => k.replace('_', ' ')), values: keys.map(k => s[k].count_days),
        type: 'pie', marker: { colors: keys.map(k => CC[k] || '#888') }, textinfo: 'label+percent', hole: .4,
        textfont: { color: '#e5e5e5' }
    }], { ...L, showlegend: false }, C);
}

function cCluster(d) {
    if (!d || !d.centers) return; const centers = d.centers;
    const traces = centers.map((c, i) => {
        const lbl = Object.entries(d.savings).find(([k, v]) => true);
        return { y: c, x: Array.from({ length: c.length }, (_, j) => j), type: 'scatter', mode: 'lines', name: `Cluster ${i}`, line: { width: 2 } };
    });
    Plotly.newPlot('clusterChart', traces, { ...L, xaxis: { ...L.xaxis, title: 'Hour' }, yaxis: { ...L.yaxis, title: 'Avg kWh' }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C);
}

function cTime(data) {
    if (!data[0].cluster_label) return;
    const labels = [...new Set(data.map(d => d.cluster_label))];
    const traces = labels.map(lbl => {
        const pts = data.filter(d => d.cluster_label === lbl);
        return { x: pts.map(d => d.timestamp), y: pts.map(d => d.energy_kwh), type: 'scatter', mode: 'markers', name: lbl.replace('_', ' '), marker: { size: 2, color: CC[lbl] || '#888' } };
    });
    Plotly.newPlot('timeChart', traces, { ...L, xaxis: { ...L.xaxis, title: 'Time' }, yaxis: { ...L.yaxis, title: 'kWh' }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C);
}

function cHourly(data) {
    if (!data[0].cluster_label) return;
    const labels = [...new Set(data.map(d => d.cluster_label))];
    const traces = labels.map(lbl => {
        const pts = data.filter(d => d.cluster_label === lbl); const h = {};
        pts.forEach(d => { if (!h[d.hour]) h[d.hour] = { t: 0, c: 0 }; h[d.hour].t += d.energy_kwh; h[d.hour].c++ });
        const hrs = Array.from({ length: 24 }, (_, i) => i);
        return { x: hrs, y: hrs.map(hr => h[hr] ? +(h[hr].t / h[hr].c).toFixed(2) : 0), type: 'scatter', mode: 'lines', name: lbl.replace('_', ' '), line: { color: CC[lbl], width: 2 } };
    });
    Plotly.newPlot('hourlyChart', traces, { ...L, xaxis: { ...L.xaxis, title: 'Hour', tickmode: 'linear', dtick: 2 }, yaxis: { ...L.yaxis, title: 'Avg kWh' }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C);
}

function cWknd(data) {
    const wk = data.filter(d => !d.is_weekend), we = data.filter(d => d.is_weekend);
    Plotly.newPlot('wkndChart', [
        { y: wk.map(d => d.energy_kwh), type: 'box', name: 'Weekday', marker: { color: 'rgb(148,163,184)' } },
        { y: we.map(d => d.energy_kwh), type: 'box', name: 'Weekend', marker: { color: 'rgb(52,211,153)' } }
    ], L, C);
}

async function load() {
    const [st, hist] = await Promise.all([f('stats'), f('historical')]);
    if (st) { uKPI(st); cPie(st); cCluster(st); }
    if (hist && hist.data) { cTime(hist.data); cHourly(hist.data); cWknd(hist.data); }
}
document.addEventListener('DOMContentLoaded', load);
