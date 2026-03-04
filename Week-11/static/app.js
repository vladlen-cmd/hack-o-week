const API = 'http://localhost:5010/api';
let currentUser = null;

function switchTab(tab) {
    document.getElementById('tabRegister').classList.toggle('active', tab === 'register');
    document.getElementById('tabLogin').classList.toggle('active', tab === 'login');
    document.getElementById('registerForm').style.display = tab === 'register' ? 'flex' : 'none';
    document.getElementById('loginForm').style.display = tab === 'login' ? 'flex' : 'none';
    document.getElementById('authSubtitle').textContent = tab === 'register' ? 'Create your account' : 'Welcome back';
    document.getElementById('regError').textContent = '';
    document.getElementById('loginError').textContent = '';
}

async function handleRegister(e) {
    e.preventDefault();
    const err = document.getElementById('regError');
    const pw = document.getElementById('regPassword').value;
    if (pw !== document.getElementById('regConfirm').value) { err.textContent = 'Passwords do not match'; return; }
    const btn = document.getElementById('regBtn'); btn.disabled = true; btn.textContent = 'Creating...';
    try {
        const res = await fetch(`${API}/register`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                username: document.getElementById('regUsername').value, email: document.getElementById('regEmail').value,
                password: pw, full_name: document.getElementById('regFullName').value, department: document.getElementById('regDept').value
            })
        });
        const data = await res.json();
        if (data.success) {
            switchTab('login'); document.getElementById('loginError').textContent = ''; err.textContent = '';
            document.getElementById('loginId').value = document.getElementById('regUsername').value;
        } else { err.textContent = data.error; }
    } catch (e) { err.textContent = 'Network error'; }
    btn.disabled = false; btn.textContent = 'Create Account';
}

async function handleLogin(e) {
    e.preventDefault();
    const err = document.getElementById('loginError');
    const btn = document.getElementById('loginBtn'); btn.disabled = true; btn.textContent = 'Signing in...';
    try {
        const res = await fetch(`${API}/login`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'include',
            body: JSON.stringify({ identifier: document.getElementById('loginId').value, password: document.getElementById('loginPassword').value })
        });
        const data = await res.json();
        if (data.success) { currentUser = data.user; showDashboard(); } else { err.textContent = data.error; }
    } catch (e) { err.textContent = 'Network error'; }
    btn.disabled = false; btn.textContent = 'Sign In';
}

function showDashboard() {
    document.getElementById('authView').style.display = 'none';
    document.getElementById('dashView').style.display = 'block';
    const u = currentUser;
    const initials = (u.full_name || u.username).split(' ').map(w => w[0]).join('').slice(0, 2);
    document.getElementById('userAvatar').style.background = u.avatar_color;
    document.getElementById('userAvatar').textContent = initials;
    document.getElementById('userName').textContent = u.full_name || u.username;
    updateProfile();
    loadUsers();
    document.getElementById('editName').value = u.full_name || '';
    document.getElementById('editDept').value = u.department || '';
}

function updateProfile() {
    const u = currentUser;
    const initials = (u.full_name || u.username).split(' ').map(w => w[0]).join('').slice(0, 2);
    document.getElementById('profileContent').innerHTML = `
        <div class="profile-avatar"><div class="avatar" style="background:${u.avatar_color}">${initials}</div>
        <div class="profile-avatar-info"><div class="pn">${u.full_name || u.username}</div><div class="pe">${u.email}</div></div></div>
        <div class="profile-info">
            <div class="profile-row"><span class="profile-label">Username</span><span class="profile-value">${u.username}</span></div>
            <div class="profile-row"><span class="profile-label">Department</span><span class="profile-value">${u.department || '—'}</span></div>
            <div class="profile-row"><span class="profile-label">Role</span><span class="profile-value"><span class="role-badge ${u.role}">${u.role}</span></span></div>
            <div class="profile-row"><span class="profile-label">Member since</span><span class="profile-value">${new Date(u.created_at).toLocaleDateString()}</span></div>
        </div>`;
}

async function handleEditProfile(e) {
    e.preventDefault();
    const msg = document.getElementById('editMsg');
    try {
        const res = await fetch(`${API}/profile`, {
            method: 'PUT', headers: { 'Content-Type': 'application/json' }, credentials: 'include',
            body: JSON.stringify({ full_name: document.getElementById('editName').value, department: document.getElementById('editDept').value })
        });
        const data = await res.json();
        if (data.success) {
            currentUser.full_name = document.getElementById('editName').value;
            currentUser.department = document.getElementById('editDept').value;
            updateProfile();
            const initials = (currentUser.full_name || currentUser.username).split(' ').map(w => w[0]).join('').slice(0, 2);
            document.getElementById('userAvatar').textContent = initials;
            document.getElementById('userName').textContent = currentUser.full_name || currentUser.username;
            msg.textContent = '✓ Profile updated'; setTimeout(() => msg.textContent = '', 3000);
        }
    } catch (e) { msg.textContent = 'Error updating'; }
}

async function loadUsers() {
    try {
        const res = await fetch(`${API}/users`);
        const data = await res.json();
        if (!data.success) return;
        document.getElementById('userCount').textContent = `${data.total} registered users`;
        let html = '<table class="users-table"><thead><tr><th>User</th><th>Department</th><th>Role</th><th>Joined</th></tr></thead><tbody>';
        data.users.forEach(u => {
            const initials = (u.full_name || u.username).split(' ').map(w => w[0]).join('').slice(0, 2);
            html += `<tr><td><div class="user-cell"><div class="avatar" style="background:${u.avatar_color}">${initials}</div><span>${u.full_name || u.username}</span></div></td>
                <td>${u.department || '—'}</td><td><span class="role-badge ${u.role}">${u.role}</span></td>
                <td>${new Date(u.created_at).toLocaleDateString()}</td></tr>`;
        });
        html += '</tbody></table>';
        document.getElementById('usersTable').innerHTML = html;
    } catch (e) { console.error(e); }
}

async function handleLogout() {
    await fetch(`${API}/logout`, { method: 'POST', credentials: 'include' });
    currentUser = null;
    document.getElementById('authView').style.display = 'flex';
    document.getElementById('dashView').style.display = 'none';
}

// Check if already logged in
async function checkSession() {
    try {
        const res = await fetch(`${API}/profile`, { credentials: 'include' });
        const data = await res.json();
        if (data.success) { currentUser = data.user; showDashboard(); }
    } catch (e) { }
}

document.addEventListener('DOMContentLoaded', checkSession);
