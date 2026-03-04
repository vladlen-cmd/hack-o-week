const API = 'http://localhost:5013/api';
const CRYPTO_KEY = 'hack-o-week-secret-key-2026';

async function post(ep, body) {
    try { const r = await fetch(`${API}/${ep}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }); return await r.json() } catch (e) { return { success: false, error: e.message } }
}

async function loadStats() {
    try {
        const r = await fetch(`${API}/stats`); const d = await r.json(); if (!d.success) return;
        document.getElementById('kTotal').textContent = d.total_records;
        document.getElementById('kBytes').textContent = d.total_encrypted_bytes.toLocaleString();
    } catch (e) { }
}

async function encryptPipeline() {
    const text = document.getElementById('eText').value;
    const label = document.getElementById('eLabel').value;
    const steps = document.getElementById('encSteps');
    const t0 = performance.now();

    // Step 1: Client-side CryptoJS AES
    const clientCipher = CryptoJS.AES.encrypt(text, CRYPTO_KEY).toString();
    const t1 = performance.now();
    steps.innerHTML = `<div class="pipeline-step">
        <div class="ps-head"><span class="ps-name"><i class="fa-solid fa-check-circle" style="color:var(--accent-success)"></i> Step 1: CryptoJS AES (Client)</span><span class="ps-time">${(t1 - t0).toFixed(1)}ms</span></div>
        <div class="ps-output">${clientCipher.substring(0, 60)}${clientCipher.length > 60 ? '...' : ''}</div>
    </div>`;

    // Step 2: Send to server for Fernet encryption + DB storage
    const result = await post('encrypt', { text: clientCipher, label });
    const t2 = performance.now();
    if (result.success) {
        steps.innerHTML += `<div class="pipeline-step">
            <div class="ps-head"><span class="ps-name"><i class="fa-solid fa-check-circle" style="color:var(--accent-success)"></i> Step 2: Fernet Encryption (Server)</span><span class="ps-time">${(t2 - t1).toFixed(1)}ms</span></div>
            <div class="ps-output">${result.ciphertext.substring(0, 60)}...</div>
        </div>`;
        steps.innerHTML += `<div class="pipeline-step">
            <div class="ps-head"><span class="ps-name"><i class="fa-solid fa-check-circle" style="color:var(--accent-success)"></i> Step 3: Stored in SQLite DB</span><span class="ps-time">${(t2 - t0).toFixed(1)}ms total</span></div>
            <div class="ps-output">${result.original_size} bytes → ${result.encrypted_size} bytes (${((result.encrypted_size / result.original_size) * 100).toFixed(0)}% expansion)</div>
        </div>`;
        document.getElementById('ePerf').innerHTML = `<i class="fa-solid fa-bolt"></i> Pipeline complete in ${(t2 - t0).toFixed(1)}ms — dual-layer encrypted & persisted`;
        loadStats(); loadStore();
    } else {
        document.getElementById('ePerf').innerHTML = `<span style="color:var(--accent-primary)"><i class="fa-solid fa-triangle-exclamation"></i> ${result.error}</span>`;
    }
}

async function decryptData() {
    const ct = document.getElementById('dText').value.trim();
    if (!ct) return;
    const result = await post('decrypt', { ciphertext: ct });
    const el = document.getElementById('dResult');
    if (result.success) {
        try {
            const bytes = CryptoJS.AES.decrypt(result.plaintext, CRYPTO_KEY);
            const original = bytes.toString(CryptoJS.enc.Utf8);
            el.innerHTML = `<div class="pipeline-step" style="border-color:var(--accent-success)">
                <div class="ps-head"><span class="ps-name"><i class="fa-solid fa-lock-open" style="color:var(--accent-success)"></i> Fully Decrypted</span><span class="ps-time"><span class="method-badge">Fernet → AES</span></span></div>
                <div class="ps-output" style="color:var(--accent-success);max-height:none">${original}</div>
            </div>`;
        } catch (e) {
            el.innerHTML = `<div class="pipeline-step">
                <div class="ps-head"><span class="ps-name"><i class="fa-solid fa-lock-open"></i> Fernet Layer Decrypted</span><span class="ps-time"><span class="method-badge">Partial</span></span></div>
                <div class="ps-output">${result.plaintext}</div>
            </div>`;
        }
    } else {
        el.innerHTML = `<div class="pipeline-step" style="border-color:var(--accent-primary)">
            <div class="ps-head"><span class="ps-name" style="color:var(--accent-primary)"><i class="fa-solid fa-triangle-exclamation"></i> Decryption Failed</span></div>
            <div class="ps-output" style="color:var(--accent-primary)">${result.error}</div>
        </div>`;
    }
}

async function loadStore() {
    try {
        const r = await fetch(`${API}/store`); const d = await r.json(); if (!d.success) return;
        if (!d.records.length) { document.getElementById('storeTable').innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:2rem">No encrypted records yet — use the form above to encrypt data</p>'; return; }
        let html = '<table class="hist-table"><thead><tr><th>ID</th><th>Label</th><th>Algorithm</th><th>Size</th><th>Timestamp</th><th>Action</th></tr></thead><tbody>';
        d.records.forEach(r => {
            html += `<tr>
            <td>${r.id}</td>
            <td><strong>${r.label}</strong></td>
            <td><span class="method-badge">${r.algorithm}</span></td>
            <td class="mono">${r.original_size} → ${r.encrypted_size}</td>
            <td>${r.timestamp}</td>
            <td><button class="btn btn-outline btn-sm" onclick="document.getElementById('dText').value='${r.ciphertext}';window.scrollTo({top:0,behavior:'smooth'})"><i class="fa-solid fa-copy"></i></button></td>
        </tr>`;
        });
        html += '</tbody></table>'; document.getElementById('storeTable').innerHTML = html;
    } catch (e) { }
}

document.addEventListener('DOMContentLoaded', () => { loadStats(); loadStore() });
