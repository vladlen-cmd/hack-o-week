const API = 'http://localhost:5009/api';
const L = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: 'Inter,sans-serif', color: '#e5e5e5', size: 12 }, xaxis: { gridcolor: 'rgba(255,255,255,.06)' }, yaxis: { gridcolor: 'rgba(255,255,255,.06)' }, margin: { t: 30, r: 30, b: 50, l: 60 }, hovermode: 'closest' };
const C = { responsive: true, displayModeBar: false };
async function f(ep) { try { const r = await fetch(`${API}/${ep}`); const d = await r.json(); if (d.success) return d; throw new Error(d.error) } catch (e) { console.error(ep, e); return null } }

function uSummary(d) {
    const el = document.getElementById('summaryCards'); if (!d) return; const s = d.summary;
    el.innerHTML = `
    <div class="card summary-card"><div class="sc-icon"><i class="fa-solid fa-bolt"></i></div><div class="sc-value" style="color:hsl(145,65%,45%)">${s.avg_daily_elec}</div><div class="sc-label">Avg Daily kWh</div><div class="sc-sub">${s.solar_pct}% solar</div></div>
    <div class="card summary-card"><div class="sc-icon"><i class="fa-solid fa-sun"></i></div><div class="sc-value" style="color:hsl(45,90%,55%)">${(s.total_solar_kwh / 1000).toFixed(1)}k</div><div class="sc-label">Solar kWh Generated</div></div>
    <div class="card summary-card"><div class="sc-icon"><i class="fa-solid fa-earth-americas"></i></div><div class="sc-value" style="color:hsl(0,65%,55%)">${(s.total_co2_kg / 1000).toFixed(1)}t</div><div class="sc-label">CO₂ Emitted</div><div class="sc-sub">${(s.total_co2_saved_kg / 1000).toFixed(1)}t saved</div></div>
    <div class="card summary-card"><div class="sc-icon"><i class="fa-solid fa-chart-bar"></i></div><div class="sc-value" style="color:hsl(145,65%,45%)">${s.avg_score}</div><div class="sc-label">Sustainability Score</div><div class="sc-sub">${s.score_trend > 0 ? '↑' : '↓'}${Math.abs(s.score_trend)} trend</div></div>`
}

function cTS(data) {
    const recent = data.slice(-720); // last 30 days
    Plotly.newPlot('tsChart', [
        { x: recent.map(d => d.timestamp), y: recent.map(d => d.total_electricity_kwh), type: 'scatter', mode: 'lines', name: 'Consumption', line: { color: 'rgb(239,68,68)', width: 1.5 } },
        { x: recent.map(d => d.timestamp), y: recent.map(d => d.solar_generation_kwh), type: 'scatter', mode: 'lines', name: 'Solar', line: { color: 'rgb(250,204,21)', width: 1.5 }, fill: 'tozeroy', fillcolor: 'rgba(250,204,21,.1)' }
    ], { ...L, xaxis: { ...L.xaxis, title: 'Date' }, yaxis: { ...L.yaxis, title: 'kWh' }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C)
}

function cCO2(data) {
    // Aggregate by week
    const weeks = {}; data.forEach(d => { const w = d.timestamp.slice(0, 10); if (!weeks[w]) weeks[w] = { em: 0, sv: 0 }; weeks[w].em += d.co2_emissions_kg; weeks[w].sv += d.co2_saved_kg });
    const dates = Object.keys(weeks).sort().slice(-30);
    Plotly.newPlot('co2Chart', [
        { x: dates, y: dates.map(w => Math.round(weeks[w].em)), type: 'bar', name: 'Emitted', marker: { color: 'rgba(239,68,68,.7)' } },
        { x: dates, y: dates.map(w => Math.round(weeks[w].sv)), type: 'bar', name: 'Saved', marker: { color: 'rgba(52,211,153,.7)' } }
    ], { ...L, barmode: 'group', xaxis: { ...L.xaxis, title: 'Date' }, yaxis: { ...L.yaxis, title: 'CO₂ (kg)' }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C)
}

function cScore(data) {
    const daily = {}; data.forEach(d => { const day = d.timestamp.slice(0, 10); if (!daily[day]) daily[day] = { t: 0, c: 0 }; daily[day].t += d.sustainability_score; daily[day].c++ });
    const dates = Object.keys(daily).sort(), vals = dates.map(d => daily[d].t / daily[d].c);
    Plotly.newPlot('scoreChart', [{ x: dates, y: vals, type: 'scatter', mode: 'lines+markers', line: { color: 'rgb(52,211,153)', width: 2 }, marker: { size: 3 } }], { ...L, xaxis: { ...L.xaxis, title: 'Date' }, yaxis: { ...L.yaxis, title: 'Score', range: [0, 100] } }, C)
}

function cWater(data) {
    const hr = {}; data.forEach(d => { const h = new Date(d.timestamp).getHours(); if (!hr[h]) hr[h] = { t: 0, c: 0 }; hr[h].t += d.water_consumption_l; hr[h].c++ });
    const hrs = Array.from({ length: 24 }, (_, i) => i), vals = hrs.map(h => hr[h] ? Math.round(hr[h].t / hr[h].c) : 0);
    Plotly.newPlot('waterChart', [{ x: hrs, y: vals, type: 'bar', marker: { color: 'rgba(59,130,246,.7)', line: { color: 'rgb(59,130,246)', width: 1 } } }], { ...L, xaxis: { ...L.xaxis, title: 'Hour', tickmode: 'linear', dtick: 2 }, yaxis: { ...L.yaxis, title: 'Avg Water (L)' } }, C)
}

function cFeat(fd) { const s = Object.entries(fd).sort((a, b) => b[1] - a[1]); Plotly.newPlot('featChart', [{ x: s.map(([, v]) => v), y: s.map(([n]) => n.replace(/_/g, ' ')), type: 'bar', orientation: 'h', marker: { color: s.map(([, v]) => v), colorscale: [[0, 'rgb(52,211,153)'], [1, 'rgb(34,197,94)']], line: { width: 1, color: 'rgba(255,255,255,.1)' } } }], { ...L, margin: { ...L.margin, l: 180 }, xaxis: { ...L.xaxis, title: 'Importance' } }, C) }

async function load() { const [h, su, st] = await Promise.all([f('historical'), f('sustainability'), f('stats')]); if (su) uSummary(su); if (st) cFeat(st.feature_importance); if (h) { cTS(h.data); cCO2(h.data); cScore(h.data); cWater(h.data) } document.getElementById('lastUpdate').textContent = `Updated: ${new Date().toLocaleTimeString()}` }
document.addEventListener('DOMContentLoaded', () => { load(); setInterval(load, 30000) });
