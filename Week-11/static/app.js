const API = 'http://localhost:5010/api';
let TOKEN = null;

async function post(ep, body, auth = false) {
    const headers = { 'Content-Type': 'application/json' };
    if (auth && TOKEN) headers['Authorization'] = `Bearer ${TOKEN}`;
    try { const r = await fetch(`${API}/${ep}`, { method: 'POST', headers, body: JSON.stringify(body) }); return await r.json() } catch (e) { return { success: false, error: e.message } }
}

async function get(ep) {
    const headers = {}; if (TOKEN) headers['Authorization'] = `Bearer ${TOKEN}`;
    try { const r = await fetch(`${API}/${ep}`, { headers }); return await r.json() } catch (e) { return { success: false } }
}

function switchTab(t) {
    document.getElementById('loginForm').style.display = t === 'login' ? 'block' : 'none';
    document.getElementById('registerForm').style.display = t === 'register' ? 'block' : 'none';
    document.querySelectorAll('.tab').forEach((e, i) => e.classList.toggle('active', i === (t === 'login' ? 0 : 1)));
}

async function register() {
    const d = await post('register', { username: document.getElementById('rUser').value, email: document.getElementById('rEmail').value, password: document.getElementById('rPass').value });
    document.getElementById('rMsg').textContent = d.success ? 'Registered! Please login.' : d.error;
    document.getElementById('rMsg').style.color = d.success ? 'var(--accent-primary)' : 'rgb(239,68,68)';
    if (d.success) switchTab('login');
}

async function login() {
    const d = await post('login', { username: document.getElementById('lUser').value, password: document.getElementById('lPass').value });
    document.getElementById('lMsg').textContent = d.success ? 'Login successful!' : d.error;
    document.getElementById('lMsg').style.color = d.success ? 'var(--accent-primary)' : 'rgb(239,68,68)';
    if (d.success) { TOKEN = d.token; showDash(d.username); }
}

function showDash(username) {
    document.getElementById('authSection').style.display = 'none';
    document.getElementById('dashSection').style.display = 'block';
    document.getElementById('authStatus').innerHTML = `<span class="status-dot"></span> ${username}`;
    loadDash();
}

async function loadDash() {
    const [profile, users] = await Promise.all([get('profile'), get('users')]);
    if (profile && profile.success) {
        const u = profile.user;
        document.getElementById('kpiCards').innerHTML = `
        <div class="card kpi"><div class="ki"><i class="fa-solid fa-user"></i></div><div class="kv">${u.username}</div><div class="kl">Username</div></div>
        <div class="card kpi"><div class="ki"><i class="fa-solid fa-clock"></i></div><div class="kv">${u.wearable_records}</div><div class="kl">Wearable Records</div></div>
        <div class="card kpi"><div class="ki"><i class="fa-solid fa-key"></i></div><div class="kv">JWT</div><div class="kl">Auth Type</div></div>`;
        document.getElementById('profileCard').innerHTML = `<div style="padding:.5rem"><p><strong>Email:</strong> ${u.email}</p><p><strong>Member since:</strong> ${u.created_at}</p><p><strong>Profile hash:</strong> <code>${u.profile._hash || 'none'}</code></p></div>`;
    }
    if (users && users.success) {
        let html = '<table class="anom-table"><thead><tr><th>Username</th><th>Email</th><th>Joined</th></tr></thead><tbody>';
        users.users.forEach(u => { html += `<tr><td>${u.username}</td><td>${u.email}</td><td>${u.created_at}</td></tr>` });
        html += '</tbody></table>'; document.getElementById('userTable').innerHTML = html;
    }
}

async function syncWearable() {
    const d = await post('wearable-sync', {
        heart_rate: parseInt(document.getElementById('wHR').value) || 75,
        steps: parseInt(document.getElementById('wSteps').value) || 1000,
        calories: parseInt(document.getElementById('wCal').value) || 200, device: 'WebPortal'
    }, true);
    document.getElementById('wMsg').textContent = d.success ? `Synced! Total: ${d.total_records} records` : d.error;
    document.getElementById('wMsg').style.color = d.success ? 'var(--accent-primary)' : 'rgb(239,68,68)';
    if (d.success) loadDash();
}
