const API = 'http://localhost:5014/api';
const L = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: 'Inter,sans-serif', color: '#e5e5e5', size: 12 }, xaxis: { gridcolor: 'rgba(255,255,255,.06)' }, yaxis: { gridcolor: 'rgba(255,255,255,.06)' }, margin: { t: 30, r: 30, b: 50, l: 60 }, hovermode: 'closest' };
const C = { responsive: true, displayModeBar: false };
const AC = 'rgb(251,146,60)';
async function f(ep) { try { const r = await fetch(`${API}/${ep}`); return await r.json() } catch (e) { return null } }

function uKPI(d) {
    const el = document.getElementById('kpiCards'); if (!d) return; const m = d.metrics;
    el.innerHTML = `
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-bullseye"></i></div><div class="kv">${m.f1}</div><div class="kl">F1 Score</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-crosshairs"></i></div><div class="kv">${m.precision}</div><div class="kl">Precision</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-satellite-dish"></i></div><div class="kv">${m.recall}</div><div class="kl">Recall</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-triangle-exclamation"></i></div><div class="kv">${m.total_anomalies}</div><div class="kl">Flagged</div></div>`;
}

function cHR(data) {
    const normal = data.filter(d => !d.predicted_anomaly), anom = data.filter(d => d.predicted_anomaly);
    Plotly.newPlot('hrChart', [
        { x: normal.map(d => d.timestamp), y: normal.map(d => d.heart_rate), type: 'scatter', mode: 'markers', name: 'Normal', marker: { size: 2, color: 'rgba(251,146,60,.25)' } },
        { x: anom.map(d => d.timestamp), y: anom.map(d => d.heart_rate), type: 'scatter', mode: 'markers', name: 'Anomaly', marker: { size: 5, color: 'rgb(239,68,68)', symbol: 'x' } }
    ], { ...L, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' }, yaxis: { ...L.yaxis, title: 'BPM' } }, C);
}

function cScore(data) {
    Plotly.newPlot('scoreChart', [{
        x: data.map(d => d.anomaly_score), type: 'histogram',
        marker: { color: AC }, nbinsx: 50
    }], { ...L, xaxis: { ...L.xaxis, title: 'Anomaly Score' }, yaxis: { ...L.yaxis, title: 'Count' } }, C);
}

function cCM(d) {
    if (!d) return; const m = d.metrics;
    Plotly.newPlot('cmChart', [{
        z: [[m.tn, m.fp], [m.fn, m.tp]], x: ['Predicted Normal', 'Predicted Anomaly'],
        y: ['Actual Normal', 'Actual Anomaly'], type: 'heatmap', colorscale: [[0, 'rgb(20,20,30)'], [1, AC]],
        text: [[m.tn, m.fp], [m.fn, m.tp]], texttemplate: '%{text}', textfont: { color: 'white', size: 16 },
        showscale: false
    }], { ...L, yaxis: { ...L.yaxis, autorange: 'reversed' } }, C);
}

async function checkHR() {
    const body = {
        heart_rate: parseFloat(document.getElementById('chkHR').value),
        resting_hr: parseFloat(document.getElementById('chkRHR').value),
        hr_variability: parseFloat(document.getElementById('chkHRV').value)
    };
    try {
        const r = await fetch(`${API}/check`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        const d = await r.json();
        const el = document.getElementById('chkResult');
        if (d.is_anomaly) {
            el.innerHTML = `<span style="color:rgb(239,68,68)"><i class="fa-solid fa-triangle-exclamation"></i> IRREGULAR — Score: ${d.anomaly_score}</span>`;
        } else { el.innerHTML = `<span style="color:${AC}"><i class="fa-solid fa-check-circle"></i> NORMAL — Score: ${d.anomaly_score}</span>`; }
    } catch (e) { }
}

async function load() {
    const [st, dt] = await Promise.all([f('stats'), f('data')]);
    if (st) { uKPI(st); cCM(st); } if (dt && dt.data) { cHR(dt.data); cScore(dt.data); }
}
document.addEventListener('DOMContentLoaded', load);
