const API = 'http://localhost:5011/api';
const L = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: 'Inter,sans-serif', color: '#e5e5e5', size: 12 }, xaxis: { gridcolor: 'rgba(255,255,255,.06)' }, yaxis: { gridcolor: 'rgba(255,255,255,.06)' }, margin: { t: 30, r: 30, b: 50, l: 60 }, hovermode: 'closest' };
const C = { responsive: true, displayModeBar: false };
const USER_COLORS = ['rgb(244,63,94)', 'rgb(59,130,246)', 'rgb(52,211,153)', 'rgb(250,204,21)', 'rgb(168,85,247)'];
async function f(ep) { try { const r = await fetch(`${API}/${ep}`); const d = await r.json(); if (d.success) return d; throw new Error(d.error) } catch (e) { console.error(ep, e); return null } }

function uStats(d) {
    const el = document.getElementById('statCards'); if (!d) return; const s = d.stats;
    el.innerHTML = `
    <div class="card stat-card"><div class="si"><i class="fa-solid fa-heartbeat"></i></div><div class="sv" style="color:var(--accent-hr)">${s.avg_heart_rate}</div><div class="sl">Avg Heart Rate (bpm)</div></div>
    <div class="card stat-card"><div class="si"><i class="fa-solid fa-shoe-prints"></i></div><div class="sv" style="color:var(--accent-steps)">${s.avg_daily_steps_per_user.toLocaleString()}</div><div class="sl">Avg Daily Steps</div></div>
    <div class="card stat-card"><div class="si"><i class="fa-solid fa-lungs"></i></div><div class="sv" style="color:var(--accent-spo2)">${s.avg_spo2}%</div><div class="sl">Avg SpO2</div></div>
    <div class="card stat-card"><div class="si"><i class="fa-solid fa-chart-bar"></i></div><div class="sv" style="color:var(--accent-primary)">${(s.total_records / 1000).toFixed(1)}k</div><div class="sl">Records Ingested</div></div>`
}

function cHR(data) {
    const users = {}; data.forEach(d => { if (!users[d.user_id]) users[d.user_id] = []; users[d.user_id].push(d) });
    const traces = Object.entries(users).map(([uid, pts], i) => ({ x: pts.map(p => p.timestamp), y: pts.map(p => p.heart_rate_bpm), type: 'scatter', mode: 'lines', name: uid, line: { color: USER_COLORS[i % USER_COLORS.length], width: 1 } }));
    Plotly.newPlot('hrChart', traces, { ...L, xaxis: { ...L.xaxis, title: 'Time' }, yaxis: { ...L.yaxis, title: 'Heart Rate (bpm)' }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C)
}

function cHourly(d) {
    const h = d.hourly;
    Plotly.newPlot('hourlyChart', [
        { x: h.map(r => r.hour), y: h.map(r => r.heart_rate_bpm), type: 'scatter', mode: 'lines+markers', name: 'HR', line: { color: 'rgb(244,63,94)', width: 2 }, marker: { size: 4 } },
        { x: h.map(r => r.hour), y: h.map(r => r.steps), type: 'bar', name: 'Steps', yaxis: 'y2', marker: { color: 'rgba(52,211,153,.4)' } }
    ], { ...L, barmode: 'overlay', xaxis: { ...L.xaxis, title: 'Hour', tickmode: 'linear', dtick: 2 }, yaxis: { ...L.yaxis, title: 'HR (bpm)' }, yaxis2: { title: 'Steps', overlaying: 'y', side: 'right', gridcolor: 'rgba(0,0,0,0)', titlefont: { color: 'rgb(52,211,153)' }, tickfont: { color: 'rgb(52,211,153)' } }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C)
}

function cUsers(d) {
    const u = d.users;
    Plotly.newPlot('usersChart', [{ x: u.map(us => us.user_id), y: u.map(us => us.avg_hr), type: 'bar', marker: { color: u.map((_, i) => USER_COLORS[i % USER_COLORS.length]), line: { color: 'rgba(255,255,255,.15)', width: 1 } } }], { ...L, xaxis: { ...L.xaxis }, yaxis: { ...L.yaxis, title: 'Avg HR (bpm)' } }, C)
}

function cSteps(data) {
    const hr = {}; data.forEach(d => { const h = new Date(d.timestamp).getHours(); if (!hr[h]) hr[h] = { t: 0, c: 0 }; hr[h].t += d.steps; hr[h].c++ });
    const hrs = Array.from({ length: 24 }, (_, i) => i), vals = hrs.map(h => hr[h] ? Math.round(hr[h].t / hr[h].c) : 0);
    Plotly.newPlot('stepsChart', [{ x: hrs, y: vals, type: 'bar', marker: { color: vals.map(v => v > 400 ? 'rgb(52,211,153)' : 'rgba(52,211,153,.4)'), line: { color: 'rgba(255,255,255,.1)', width: 1 } } }], { ...L, xaxis: { ...L.xaxis, title: 'Hour', tickmode: 'linear', dtick: 2 }, yaxis: { ...L.yaxis, title: 'Avg Steps' } }, C)
}

function cVitals(data) {
    const hrs = {}; data.forEach(d => { const h = new Date(d.timestamp).getHours(); if (!hrs[h]) hrs[h] = { spo2_t: 0, temp_t: 0, c: 0 }; hrs[h].spo2_t += d.spo2_pct; hrs[h].temp_t += d.skin_temp_c; hrs[h].c++ });
    const h = Array.from({ length: 24 }, (_, i) => i);
    Plotly.newPlot('vitalsChart', [
        { x: h, y: h.map(hr => hrs[hr] ? +(hrs[hr].spo2_t / hrs[hr].c).toFixed(1) : 0), type: 'scatter', mode: 'lines+markers', name: 'SpO2 %', line: { color: 'rgb(59,130,246)', width: 2 }, marker: { size: 3 } },
        { x: h, y: h.map(hr => hrs[hr] ? +(hrs[hr].temp_t / hrs[hr].c).toFixed(1) : 0), type: 'scatter', mode: 'lines+markers', name: 'Skin Temp °C', yaxis: 'y2', line: { color: 'rgb(250,204,21)', width: 2 }, marker: { size: 3 } }
    ], { ...L, xaxis: { ...L.xaxis, title: 'Hour', tickmode: 'linear', dtick: 2 }, yaxis: { ...L.yaxis, title: 'SpO2 (%)' }, yaxis2: { title: 'Temp (°C)', overlaying: 'y', side: 'right', gridcolor: 'rgba(0,0,0,0)' }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C)
}

function uDeviceTable(d) {
    const el = document.getElementById('deviceTable'); if (!d) return;
    let html = '<table class="device-table"><thead><tr><th>User</th><th>Device</th><th>Records</th><th>Avg HR</th><th>Total Steps</th><th>Avg SpO2</th><th>Last Seen</th></tr></thead><tbody>';
    d.users.forEach(u => { html += `<tr><td>${u.user_id}</td><td><span class="device-badge">${u.device_type}</span></td><td>${u.records.toLocaleString()}</td><td>${u.avg_hr} bpm</td><td>${u.total_steps.toLocaleString()}</td><td>${u.avg_spo2}%</td><td>${new Date(u.last_seen).toLocaleString()}</td></tr>` });
    html += '</tbody></table>'; el.innerHTML = html
}

async function load() {
    const [st, dt, hr, us] = await Promise.all([f('stats'), f('data?limit=2000'), f('hourly'), f('users')]);
    if (st) uStats(st); if (hr) cHourly(hr); if (us) { cUsers(us); uDeviceTable(us) }
    if (dt) { cHR(dt.data); cSteps(dt.data); cVitals(dt.data) }
    document.getElementById('lastUpdate').textContent = `Updated: ${new Date().toLocaleTimeString()}`
}
document.addEventListener('DOMContentLoaded', () => { load(); setInterval(load, 30000) });
