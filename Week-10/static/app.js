const API = 'http://localhost:5009/api';
const L = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: 'Inter,sans-serif', color: '#e5e5e5', size: 12 }, xaxis: { gridcolor: 'rgba(255,255,255,.06)' }, yaxis: { gridcolor: 'rgba(255,255,255,.06)' }, margin: { t: 30, r: 30, b: 50, l: 60 }, hovermode: 'closest' };
const C = { responsive: true, displayModeBar: false };
const GC = 'rgb(52,211,153)';
async function f(ep) { try { const r = await fetch(`${API}/${ep}`); return await r.json() } catch (e) { return null } }

function uKPI(d) {
    const el = document.getElementById('kpiCards'); if (!d) return; const m = d.metrics, c = d.carbon;
    el.innerHTML = `
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-chart-line"></i></div><div class="kv">${m.r2_ensemble}</div><div class="kl">R² (Ensemble)</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-leaf"></i></div><div class="kv">${c.saved_kg} kg</div><div class="kl">CO₂ Saved</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-solar-panel"></i></div><div class="kv">${c.solar_total_kwh}</div><div class="kl">Solar kWh</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-percent"></i></div><div class="kv">${c.pct_saved}%</div><div class="kl">Carbon Reduction</div></div>`;
}

function cCarbon(d) {
    if (!d) return; const c = d.carbon;
    Plotly.newPlot('carbonChart', [{ labels: ['Carbon Saved', 'Carbon Emitted'], values: [c.saved_kg, c.total_kg - c.saved_kg], type: 'pie', hole: .5, marker: { colors: [GC, 'rgba(239,68,68,.6)'] }, textfont: { color: '#e5e5e5' } }], { ...L, showlegend: true, legend: { x: 0, y: 0, bgcolor: 'rgba(0,0,0,.3)' } }, C);
}

function cEnergy(data) {
    const daily = {}; data.forEach(d => {
        const dt = d.timestamp.split('T')[0] || d.timestamp.split(' ')[0];
        if (!daily[dt]) daily[dt] = { e: 0, s: 0 }; daily[dt].e += d.energy_kwh; daily[dt].s += d.solar_kwh
    });
    const dates = Object.keys(daily).sort();
    Plotly.newPlot('energyChart', [
        { x: dates, y: dates.map(d => daily[d].e.toFixed(1)), type: 'scatter', mode: 'lines', name: 'Energy', line: { color: 'rgb(239,68,68)', width: 2 } },
        { x: dates, y: dates.map(d => daily[d].s.toFixed(1)), type: 'scatter', mode: 'lines', name: 'Solar', line: { color: GC, width: 2 } }
    ], { ...L, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C);
}

function cDD(data) {
    const bldgs = {}; data.forEach(d => { if (!bldgs[d.building]) bldgs[d.building] = { e: 0, c: 0 }; bldgs[d.building].e += d.energy_kwh; bldgs[d.building].c++ });
    const names = Object.keys(bldgs), vals = names.map(b => +(bldgs[b].e / bldgs[b].c).toFixed(2));
    Plotly.newPlot('ddChart', [{ x: names, y: vals, type: 'bar', marker: { color: GC } }], { ...L, xaxis: { ...L.xaxis, title: 'Building' }, yaxis: { ...L.yaxis, title: 'Avg kWh' } }, C);
}

function cHourly(data) {
    const h = {}; data.forEach(d => { const hr = d.hour; if (!h[hr]) h[hr] = { t: 0, c: 0 }; h[hr].t += d.energy_kwh; h[hr].c++ });
    const hrs = Array.from({ length: 24 }, (_, i) => i);
    Plotly.newPlot('hourlyChart', [{ x: hrs, y: hrs.map(i => h[i] ? +(h[i].t / h[i].c).toFixed(2) : 0), type: 'bar', marker: { color: GC } }], { ...L, xaxis: { ...L.xaxis, title: 'Hour', tickmode: 'linear', dtick: 2 }, yaxis: { ...L.yaxis, title: 'Avg kWh' } }, C);
}

function cWater(data) { Plotly.newPlot('waterChart', [{ x: data.map(d => d.occupancy), y: data.map(d => d.water_liters), type: 'scatter', mode: 'markers', marker: { size: 2, color: 'rgba(56,189,248,.3)' } }], { ...L, xaxis: { ...L.xaxis, title: 'Occupancy' }, yaxis: { ...L.yaxis, title: 'Water (L)' } }, C); }

async function drilldown() {
    const b = document.getElementById('buildingFilter').value;
    const ep = b ? `drilldown?building=${b}` : 'drilldown';
    const d = await f(ep); if (d && d.data) { cDD(d.data); cHourly(d.data); cEnergy(d.data); }
}

async function load() {
    const [st, dd] = await Promise.all([f('sustainability'), f('drilldown')]);
    if (st) { uKPI(st); cCarbon(st); }
    if (dd && dd.data) {
        cEnergy(dd.data); cDD(dd.data); cHourly(dd.data); cWater(dd.data);
        const sel = document.getElementById('buildingFilter'); dd.buildings.forEach(b => { const o = document.createElement('option'); o.value = b; o.textContent = b; sel.appendChild(o) });
    }
}
document.addEventListener('DOMContentLoaded', load);
