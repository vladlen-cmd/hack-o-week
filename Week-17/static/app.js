function showStatus(message, type = 'error') {
    const banner = document.getElementById('status-banner');
    banner.textContent = message;
    banner.classList.remove('hidden', 'error', 'info');
    banner.classList.add(type);
}

async function fetchData() {
    if (window.location.protocol === 'file:') {
        showStatus('This dashboard must be opened via Flask (http://127.0.0.1:5000), not as a local file.', 'error');
        return;
    }

    let res;
    try {
        res = await fetch('/data');
    } catch (err) {
        showStatus('Cannot reach backend. Start app.py and open http://127.0.0.1:5000.', 'error');
        return;
    }

    if (!res.ok) {
        showStatus(`Backend error loading data (${res.status}).`, 'error');
        return;
    }

    const data = await res.json();
    if (data.length === 0) {
        showStatus('No analyzed records found.', 'info');
        return;
    }

    const headerRow = document.getElementById('table-header');
    headerRow.innerHTML = '';
    Object.keys(data[0]).forEach(key => {
        const th = document.createElement('th');
        th.textContent = key;
        headerRow.appendChild(th);
    });
    const body = document.getElementById('table-body');
    body.innerHTML = '';
    data.forEach(row => {
        const tr = document.createElement('tr');
        if (row.anomaly_highlight === 'YES') tr.classList.add('anomaly');
        Object.values(row).forEach(val => {
            const td = document.createElement('td');
            td.textContent = val;
            tr.appendChild(td);
        });
        body.appendChild(tr);
    });
}

function exportData(format) {
    if (window.location.protocol === 'file:') {
        showStatus('Export requires backend routes. Run Flask and use http://127.0.0.1:5000.', 'error');
        return;
    }
    window.location = `/export?format=${format}`;
}

fetchData();
