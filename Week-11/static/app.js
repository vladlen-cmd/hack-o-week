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
    document.getElementById('loginForm').style.display = t === 'login' ? 'flex' : 'none';
    document.getElementById('registerForm').style.display = t === 'register' ? 'flex' : 'none';
    document.querySelectorAll('.auth-tab').forEach((e, i) => e.classList.toggle('active', i === (t === 'login' ? 0 : 1)));
    document.getElementById('lMsg').textContent = '';
    document.getElementById('rMsg').textContent = '';
}

async function register(e) {
    e.preventDefault();
    const d = await post('register', { username: document.getElementById('rUser').value, email: document.getElementById('rEmail').value, password: document.getElementById('rPass').value });
    const msg = document.getElementById('rMsg');
    if (d.success) { msg.className = 'form-success'; msg.textContent = 'Account created! Switch to Login.'; setTimeout(() => switchTab('login'), 1200); }
    else { msg.className = 'form-error'; msg.textContent = d.error || 'Registration failed'; }
}

async function login(e) {
    e.preventDefault();
    const d = await post('login', { username: document.getElementById('lUser').value, password: document.getElementById('lPass').value });
    const msg = document.getElementById('lMsg');
    if (d.success) { TOKEN = d.token; showDash(d.username); }
    else { msg.className = 'form-error'; msg.textContent = d.error || 'Invalid credentials'; }
}

function showDash(username) {
    document.getElementById('authView').style.display = 'none';
    document.getElementById('dashView').style.display = 'block';
    document.getElementById('dashUsername').textContent = username;
    document.getElementById('dashAvatar').textContent = username.charAt(0).toUpperCase();
    loadDash();
}

function logout() {
    TOKEN = null;
    document.getElementById('authView').style.display = 'flex';
    document.getElementById('dashView').style.display = 'none';
}

async function loadDash() {
    const [profile, users] = await Promise.all([get('profile'), get('users')]);

    if (profile && profile.success) {
        const u = profile.user;
        document.getElementById('profileContent').innerHTML = `
            <div class="profile-avatar">
                <div class="avatar" style="background:var(--gradient-primary)">${u.username.charAt(0).toUpperCase()}</div>
                <div class="profile-avatar-info">
                    <div class="pn">${u.username}</div>
                    <div class="pe">${u.email}</div>
                </div>
            </div>
            <div class="profile-info">
                <div class="profile-row"><span class="profile-label">Member Since</span><span class="profile-value">${u.created_at}</span></div>
                <div class="profile-row"><span class="profile-label">Wearable Records</span><span class="profile-value">${u.wearable_records}</span></div>
                <div class="profile-row"><span class="profile-label">Auth Type</span><span class="profile-value"><span class="role-badge student">JWT</span></span></div>
                <div class="profile-row"><span class="profile-label">Profile Hash</span><span class="profile-value" style="font-family:monospace;font-size:.8rem">${u.profile._hash || '—'}</span></div>
            </div>`;
    }

    if (users && users.success) {
        const colors = ['hsl(270,70%,55%)', 'hsl(340,65%,55%)', 'hsl(200,70%,50%)', 'hsl(140,60%,45%)', 'hsl(30,80%,55%)'];
        let html = '<table class="users-table"><thead><tr><th>User</th><th>Email</th><th>Role</th><th>Joined</th></tr></thead><tbody>';
        users.users.forEach((u, i) => {
            html += `<tr>
                <td><div class="user-cell"><div class="avatar" style="background:${colors[i % colors.length]}">${u.username.charAt(0).toUpperCase()}</div>${u.username}</div></td>
                <td>${u.email}</td>
                <td><span class="role-badge student">Student</span></td>
                <td>${u.created_at}</td></tr>`;
        });
        html += '</tbody></table>';
        document.getElementById('userDirectory').innerHTML = users.users.length ? html : '<p style="color:var(--text-muted);padding:1rem;text-align:center">No users registered yet</p>';
    }
}

async function syncWearable(e) {
    e.preventDefault();
    const d = await post('wearable-sync', {
        heart_rate: parseInt(document.getElementById('wHR').value) || 75,
        steps: parseInt(document.getElementById('wSteps').value) || 1000,
        calories: parseInt(document.getElementById('wCal').value) || 200, device: 'WebPortal'
    }, true);
    const msg = document.getElementById('wMsg');
    if (d.success) { msg.className = 'form-success'; msg.textContent = `Synced! Total records: ${d.total_records}`; loadDash(); }
    else { msg.className = 'form-error'; msg.textContent = d.error || 'Sync failed'; }
}
