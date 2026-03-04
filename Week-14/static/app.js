const API = 'http://localhost:5013/api';
const CRYPTO_KEY = 'hack-o-week-secret-key-2026';

async function post(ep, body) {
    try { const r = await fetch(`${API}/${ep}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }); return await r.json() } catch (e) { return { success: false, error: e.message } }
}

async function loadStats() {
    try {
        const r = await fetch(`${API}/stats`); const d = await r.json(); if (!d.success) return;
        document.getElementById('kpiCards').innerHTML = `
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-database"></i></div><div class="kv">${d.total_records}</div><div class="kl">Stored Ciphertexts</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-file-zipper"></i></div><div class="kv">${d.total_encrypted_bytes}</div><div class="kl">Encrypted Bytes</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-shield-halved"></i></div><div class="kv">CryptoJS</div><div class="kl">Client Encryption</div></div>
    <div class="card kpi"><div class="ki"><i class="fa-solid fa-server"></i></div><div class="kv">Fernet</div><div class="kl">Server Encryption</div></div>`;
    } catch (e) { }
}

async function encryptPipeline() {
    const text = document.getElementById('eText').value;
    const label = document.getElementById('eLabel').value;
    const steps = document.getElementById('encSteps');

    // Step 1: Client-side CryptoJS AES encryption
    const clientCipher = CryptoJS.AES.encrypt(text, CRYPTO_KEY).toString();
    steps.innerHTML = `<div class="step"><i class="fa-solid fa-check" style="color:var(--accent-primary)"></i> <strong>Step 1:</strong> CryptoJS AES → <code>${clientCipher.substring(0, 40)}...</code></div>`;

    // Step 2: Send to server for Fernet encryption + DB storage
    const result = await post('encrypt', { text: clientCipher, label });
    if (result.success) {
        steps.innerHTML += `<div class="step"><i class="fa-solid fa-check" style="color:var(--accent-primary)"></i> <strong>Step 2:</strong> Fernet → <code>${result.ciphertext.substring(0, 40)}...</code></div>`;
        steps.innerHTML += `<div class="step"><i class="fa-solid fa-check" style="color:var(--accent-primary)"></i> <strong>Step 3:</strong> Stored in DB (${result.original_size} → ${result.encrypted_size} bytes)</div>`;
        document.getElementById('eMsg').textContent = 'End-to-end encrypted & stored!';
        document.getElementById('eMsg').style.color = 'var(--accent-primary)';
        loadStats(); loadStore();
    } else { document.getElementById('eMsg').textContent = result.error; document.getElementById('eMsg').style.color = 'rgb(239,68,68)'; }
}

async function decryptData() {
    const ct = document.getElementById('dText').value;
    const result = await post('decrypt', { ciphertext: ct });
    if (result.success) {
        // Server decrypted Fernet layer, now decrypt CryptoJS layer
        try {
            const bytes = CryptoJS.AES.decrypt(result.plaintext, CRYPTO_KEY);
            const original = bytes.toString(CryptoJS.enc.Utf8);
            document.getElementById('dResult').innerHTML = `<div class="step"><i class="fa-solid fa-lock-open" style="color:var(--accent-primary)"></i> <strong>Decrypted:</strong> <code>${original}</code></div>`;
            document.getElementById('dMsg').textContent = 'Decrypted successfully!';
            document.getElementById('dMsg').style.color = 'var(--accent-primary)';
        } catch (e) {
            document.getElementById('dResult').innerHTML = `<div class="step"><strong>Fernet decrypted:</strong> <code>${result.plaintext}</code></div>`;
            document.getElementById('dMsg').textContent = 'Fernet layer decrypted (CryptoJS layer may need different key)';
            document.getElementById('dMsg').style.color = 'rgb(251,191,36)';
        }
    } else { document.getElementById('dMsg').textContent = result.error; document.getElementById('dMsg').style.color = 'rgb(239,68,68)'; }
}

async function loadStore() {
    try {
        const r = await fetch(`${API}/store`); const d = await r.json(); if (!d.success) return;
        let html = '<table class="anom-table"><thead><tr><th>ID</th><th>Label</th><th>Algorithm</th><th>Size</th><th>Time</th></tr></thead><tbody>';
        d.records.forEach(r => { html += `<tr><td>${r.id}</td><td>${r.label}</td><td>${r.algorithm}</td><td>${r.original_size}→${r.encrypted_size}</td><td>${r.timestamp}</td></tr>` });
        html += '</tbody></table>'; document.getElementById('storeTable').innerHTML = html;
    } catch (e) { }
}

document.addEventListener('DOMContentLoaded', () => { loadStats(); loadStore() });
