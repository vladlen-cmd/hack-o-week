const healthEl = document.getElementById('healthStatus');
const statsEl = document.getElementById('stats');
const alertsEl = document.getElementById('alerts');
const feedEl = document.getElementById('feed');
const form = document.getElementById('streamForm');
const submitResultEl = document.getElementById('submitResult');

const socket = io();

function addItem(container, html, klass = 'info') {
  const div = document.createElement('div');
  div.className = `item ${klass}`;
  div.innerHTML = html;
  container.prepend(div);
}

function renderStats(stats) {
  const cards = [
    ['Readings', stats.total_readings],
    ['Alerts', stats.total_alerts],
    ['Undelivered', stats.undelivered_alerts],
    ['Critical', stats.critical_alerts]
  ];
  statsEl.innerHTML = cards.map(([k, v]) => (
    `<div class="stat"><div class="label">${k}</div><div class="value">${v}</div></div>`
  )).join('');
}

async function fetchHealth() {
  const res = await fetch('/api/health');
  const data = await res.json();
  healthEl.textContent = data.success ? `Operational - ${data.encryption}` : 'Degraded';
}

async function fetchStats() {
  const res = await fetch('/api/stats');
  const data = await res.json();
  if (data.success) renderStats(data);
}

async function fetchAlerts() {
  const res = await fetch('/api/alerts?limit=10');
  const data = await res.json();
  if (!data.success) return;
  alertsEl.innerHTML = '';
  data.alerts.forEach(a => {
    addItem(
      alertsEl,
      `<strong>${a.severity.toUpperCase()}</strong><br/>${a.reason_summary}<br/><span class="mono small">cipher: ${a.encrypted_payload.slice(0, 52)}...</span>`,
      a.severity
    );
  });
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const payload = {
    user_id: document.getElementById('user_id').value,
    device: document.getElementById('device').value,
    bpm: Number(document.getElementById('bpm').value),
    spo2: Number(document.getElementById('spo2').value),
    temp_c: Number(document.getElementById('temp_c').value),
  };

  const res = await fetch('/api/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  const data = await res.json();
  submitResultEl.textContent = JSON.stringify(data, null, 2);
  await fetchStats();
  await fetchAlerts();
});

socket.on('status', (msg) => {
  addItem(feedEl, `<strong>Socket:</strong> ${msg.message}<br/><span class="small">${msg.timestamp}</span>`, 'info');
});

socket.on('stream_processed', (msg) => {
  addItem(
    feedEl,
    `<strong>${msg.user_id}</strong> · BPM ${msg.bpm} · ${msg.is_anomaly ? 'ANOMALY' : 'normal'}<br/><span class="small">${msg.received_at}</span>`,
    msg.severity || 'info'
  );
});

socket.on('encrypted_alert', (msg) => {
  addItem(
    alertsEl,
    `<strong>${msg.severity.toUpperCase()}</strong><br/><span class="mono small">cipher: ${msg.ciphertext.slice(0, 52)}...</span>`,
    msg.severity
  );
  fetchStats();
});

Promise.all([fetchHealth(), fetchStats(), fetchAlerts()]);
setInterval(fetchStats, 5000);
