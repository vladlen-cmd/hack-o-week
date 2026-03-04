const API = 'http://localhost:5007/api';
const L = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: 'Inter,sans-serif', color: '#e5e5e5', size: 12 }, xaxis: { gridcolor: 'rgba(255,255,255,.06)' }, yaxis: { gridcolor: 'rgba(255,255,255,.06)' }, margin: { t: 30, r: 30, b: 50, l: 60 }, hovermode: 'closest' };
const C = { responsive: true, displayModeBar: false };
const AC = 'rgb(250,204,21)';
async function f(ep) { try { const r = await fetch(`${API}/${ep}`); return await r.json() } catch (e) { return null } }

function uKPI(d) {
    const el = document.getElementById('kpiCards'); if (!d) return; const m = d.metrics, s = d.summary;
    el.innerHTML = `
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-chart-line"></i></div><div class="kv">${m.r2}</div><div class="kl">R² Score</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-lightbulb"></i></div><div class="kv">${s.avg_lighting}</div><div class="kl">Avg Lighting (kWh)</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-car"></i></div><div class="kv">${s.avg_vehicles}</div><div class="kl">Avg Vehicles</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-superscript"></i></div><div class="kv">deg ${m.poly_degree}</div><div class="kl">Polynomial Degree</div></div>`;
}

function cBar(data) {
    const recent = data.slice(-200);
    const colors = recent.map(d => d.is_anomaly ? 'rgb(239,68,68)' : AC);
    Plotly.newPlot('barChart', [{
        x: recent.map(d => d.timestamp), y: recent.map(d => d.lighting_kwh), type: 'bar',
        marker: { color: colors, line: { color: 'rgba(255,255,255,.1)', width: 0.5 } }
    }],
        { ...L, xaxis: { ...L.xaxis, title: 'Time' }, yaxis: { ...L.yaxis, title: 'Lighting (kWh)' } }, C);
}

function cVeh(data) {
    Plotly.newPlot('vehChart', [{
        x: data.map(d => d.vehicle_count), y: data.map(d => d.lighting_kwh),
        type: 'scatter', mode: 'markers', marker: { size: 2, color: data.map(d => d.is_anomaly ? 'rgb(239,68,68)' : 'rgba(250,204,21,.25)') }
    }],
        { ...L, xaxis: { ...L.xaxis, title: 'Vehicle Count' }, yaxis: { ...L.yaxis, title: 'Lighting (kWh)' } }, C);
}

function cHourly(data) {
    const h = {}; data.forEach(d => { if (!h[d.hour]) h[d.hour] = { t: 0, c: 0 }; h[d.hour].t += d.lighting_kwh; h[d.hour].c++ });
    const hrs = Array.from({ length: 24 }, (_, i) => i), vals = hrs.map(i => h[i] ? +(h[i].t / h[i].c).toFixed(2) : 0);
    Plotly.newPlot('hourlyChart', [{ x: hrs, y: vals, type: 'bar', marker: { color: vals.map(v => v > 15 ? AC : 'rgba(250,204,21,.35)') } }], { ...L, xaxis: { ...L.xaxis, title: 'Hour', tickmode: 'linear', dtick: 2 }, yaxis: { ...L.yaxis, title: 'Avg kWh' } }, C);
}

function uTable(data) {
    const anoms = data.filter(d => d.is_anomaly).slice(-15).reverse();
    if (!anoms.length) { document.getElementById('anomTable').innerHTML = '<p style="color:var(--text-muted)">No anomalies</p>'; return; }
    let html = '<table class="anom-table"><thead><tr><th>Time</th><th>Actual</th><th>Predicted</th><th>Residual</th><th>Vehicles</th></tr></thead><tbody>';
    anoms.forEach(a => { html += `<tr><td>${new Date(a.timestamp).toLocaleString()}</td><td>${a.lighting_kwh} kWh</td><td>${a.predicted} kWh</td><td style="color:rgb(239,68,68)">${a.residual.toFixed(2)}</td><td>${a.vehicle_count}</td></tr>` });
    html += '</tbody></table>'; document.getElementById('anomTable').innerHTML = html;
}

async function load() {
    const [st, hist] = await Promise.all([f('stats'), f('historical')]);
    if (st) uKPI(st);
    if (hist && hist.data) {
        cBar(hist.data); cVeh(hist.data); cHourly(hist.data); uTable(hist.data);
        document.getElementById('alertCount').textContent = hist.anomaly_count;
    }
}
document.addEventListener('DOMContentLoaded', () => { load(); setInterval(load, 30000) });
