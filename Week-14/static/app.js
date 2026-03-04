const API = 'http://localhost:5013/api';
async function post(ep, body) { try { const r = await fetch(`${API}/${ep}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }); return await r.json() } catch (e) { console.error(e); return { success: false, error: e.message } } }
async function get(ep) { try { const r = await fetch(`${API}/${ep}`); return await r.json() } catch (e) { return null } }

async function doEncrypt() {
    const text = document.getElementById('inputText').value;
    const method = document.getElementById('method').value;
    if (!text) return;
    const r = await post('encrypt', { text, method });
    if (r.success) { document.getElementById('resultText').value = r.ciphertext; document.getElementById('perf').textContent = `✓ ${r.method} · ${r.input_size}→${r.output_size} bytes · ${r.elapsed_ms}ms` }
    else { document.getElementById('resultText').value = 'Error: ' + r.error }
    refreshStats()
}

async function doDecrypt() {
    const ct = document.getElementById('resultText').value;
    const method = document.getElementById('method').value;
    if (!ct) return;
    const r = await post('decrypt', { ciphertext: ct, method });
    if (r.success) { document.getElementById('inputText').value = r.plaintext; document.getElementById('perf').textContent = `✓ Decrypted · ${r.elapsed_ms}ms` }
    else { document.getElementById('perf').textContent = '✗ ' + r.error }
    refreshStats()
}

async function doHash() {
    const text = document.getElementById('hashInput').value;
    const algo = document.getElementById('hashAlgo').value;
    if (!text) return;
    const r = await post('hash', { text, algorithm: algo });
    if (r.success) { document.getElementById('hashResult').value = r.hash; document.getElementById('hashPerf').textContent = `✓ ${r.algorithm} · ${r.elapsed_ms}ms` }
    refreshStats()
}

async function doPipeline() {
    const text = document.getElementById('pipeInput').value;
    if (!text) return;
    const r = await post('pipeline', { text });
    const el = document.getElementById('pipelineResults');
    if (!r.success) { el.innerHTML = '<p style="color:var(--accent-primary)">Error: ' + r.error + '</p>'; return }
    let html = '';
    r.steps.forEach(s => {
        const out = s.ciphertext || s.hash || s.encoded || s.plaintext || '';
        const verified = s.verified !== undefined ? ` · ${s.verified ? '✅ Verified' : '❌ Failed'}` : '';
        html += `<div class="pipeline-step"><div class="ps-head"><span class="ps-name">Step ${s.step}: ${s.name}</span><span class="ps-time">${s.elapsed_ms || '—'}ms${verified}</span></div><div class="ps-output">${out.slice(0, 120)}${out.length > 120 ? '…' : ''}</div></div>`
    });
    el.innerHTML = html; refreshStats()
}

async function refreshStats() {
    const r = await get('stats');
    if (!r || !r.success) return; const s = r.stats;
    document.getElementById('encCount').textContent = s.total_encrypted;
    document.getElementById('decCount').textContent = s.total_decrypted;
    document.getElementById('hashCount').textContent = s.total_hashed;
    document.getElementById('statsInfo').textContent = `${(s.bytes_processed / 1024).toFixed(1)} KB processed`;
    // Update history table
    if (s.history.length > 0) {
        let html = '<table class="hist-table"><thead><tr><th>Method</th><th>Input</th><th>Output</th><th>Time</th></tr></thead><tbody>';
        s.history.slice(-15).reverse().forEach(h => { html += `<tr><td><span class="method-badge">${h.method}</span></td><td>${h.input_size} B</td><td>${h.output_size} B</td><td>${h.elapsed_ms}ms</td></tr>` });
        html += '</tbody></table>';
        document.getElementById('historyTable').innerHTML = html
    }
}

document.addEventListener('DOMContentLoaded', refreshStats);
