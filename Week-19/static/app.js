const loginForm = document.getElementById('loginForm');
const panel = document.getElementById('panel');
const loginCard = document.getElementById('loginCard');
const userBadge = document.getElementById('userBadge');
const statusEl = document.getElementById('status');
const auditBody = document.getElementById('auditBody');
const reportBody = document.getElementById('reportBody');
const severityFilter = document.getElementById('severityFilter');
const refreshBtn = document.getElementById('refreshBtn');
const logoutBtn = document.getElementById('logoutBtn');

const kpiAudit = document.getElementById('kpiAudit');
const kpiOpen = document.getElementById('kpiOpen');
const kpiCritical = document.getElementById('kpiCritical');

let token = localStorage.getItem('week19_token') || '';
let currentUser = null;

function setStatus(message, tone = 'info') {
    statusEl.textContent = message;
    statusEl.classList.remove('info', 'ok', 'error');
    statusEl.classList.add(tone);
}

function severityBadge(severity) {
    return `<span class="sev ${severity}">${severity}</span>`;
}

async function api(path, opts = {}) {
    const headers = { ...(opts.headers || {}) };
    if (token) {
        headers.Authorization = `Bearer ${token}`;
    }
    const res = await fetch(path, { ...opts, headers });
    const data = await res.json().catch(() => ({}));
    return { res, data };
}

function updateKpis(logs, reports) {
    kpiAudit.textContent = String(logs.length);
    const open = reports.filter((r) => r.status !== 'resolved').length;
    const critical = reports.filter((r) => r.severity === 'critical').length;
    kpiOpen.textContent = String(open);
    kpiCritical.textContent = String(critical);
}

function renderAudit(logs) {
    auditBody.innerHTML = logs
        .map((l) => `
            <tr>
                <td>${l.accessed_at}</td>
                <td>${l.actor}</td>
                <td>${l.role}</td>
                <td>${l.action}</td>
                <td>${l.resource}</td>
                <td><code>${JSON.stringify(l.payload)}</code></td>
            </tr>
        `)
        .join('');

    if (!logs.length) {
        auditBody.innerHTML = '<tr><td colspan="6">No audit records found.</td></tr>';
    }
}

function renderReports(reports) {
    reportBody.innerHTML = reports
        .map((r) => `
            <tr>
                <td>${r.created_at}</td>
                <td>${r.source}</td>
                <td>${severityBadge(r.severity)}</td>
                <td>${r.status}</td>
                <td>${r.details.user_id}</td>
                <td>${r.details.reason}</td>
            </tr>
        `)
        .join('');

    if (!reports.length) {
        reportBody.innerHTML = '<tr><td colspan="6">No anomaly reports found.</td></tr>';
    }
}

async function loadPanelData() {
    refreshBtn.disabled = true;
    setStatus('Loading encrypted audit logs and anomaly reports...', 'info');

    try {
        let logs = [];
        if (currentUser.role === 'admin') {
            const { res: logRes, data: logData } = await api('/api/admin/audit-logs?limit=100');
            if (!logRes.ok) throw new Error(logData.error || 'Failed to load audit logs');
            logs = logData.logs || [];
            renderAudit(logs);
        } else {
            document.getElementById('auditCard').classList.add('hidden');
        }

        const sev = severityFilter.value;
        const { res: reportRes, data: reportData } = await api(`/api/admin/anomaly-reports${sev ? `?severity=${sev}` : ''}`);
        if (!reportRes.ok) throw new Error(reportData.error || 'Failed to load anomaly reports');

        const reports = reportData.reports || [];
        renderReports(reports);
        updateKpis(logs, reports);
        setStatus('Compliance data updated.', 'ok');
    } catch (err) {
        setStatus(`Error: ${err.message}`, 'error');
    } finally {
        refreshBtn.disabled = false;
    }
}

function showPanel(user) {
    currentUser = user;
    loginCard.classList.add('hidden');
    panel.classList.remove('hidden');
    userBadge.classList.remove('hidden');
    userBadge.textContent = `${user.display_name} (${user.role})`;
}

function showLogin() {
    currentUser = null;
    panel.classList.add('hidden');
    loginCard.classList.remove('hidden');
    userBadge.classList.add('hidden');
    token = '';
    localStorage.removeItem('week19_token');
}

loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(loginForm);

    const username = String(formData.get('username') || '').trim();
    const password = String(formData.get('password') || '').trim();

    try {
        const { res, data } = await api('/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });

        if (!res.ok) throw new Error(data.error || 'Login failed');

        token = data.token;
        localStorage.setItem('week19_token', token);
        showPanel(data.user);
        await loadPanelData();
    } catch (err) {
        alert(err.message);
    }
});

refreshBtn.addEventListener('click', loadPanelData);
severityFilter.addEventListener('change', loadPanelData);

logoutBtn.addEventListener('click', async () => {
    try {
        await api('/api/logout', { method: 'POST' });
    } finally {
        showLogin();
    }
});

async function restoreSession() {
    if (!token) return;

    const { res, data } = await api('/api/me');
    if (!res.ok || !data.success) {
        showLogin();
        return;
    }

    showPanel(data.user);
    await loadPanelData();
}

restoreSession();
