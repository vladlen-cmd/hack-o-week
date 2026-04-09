let probChart;
let signalChart;

const statusEl = document.getElementById('status');
const metricsGrid = document.getElementById('metricsGrid');
const rowsBody = document.getElementById('rowsBody');
const kpiPredicted = document.getElementById('kpiPredicted');
const kpiAvgProb = document.getElementById('kpiAvgProb');
const kpiSamples = document.getElementById('kpiSamples');
const runBtn = document.getElementById('runBtn');
const reloadBtn = document.getElementById('reloadBtn');

let isBusy = false;

function setStatus(msg, tone = 'info') {
    statusEl.textContent = msg;
    statusEl.classList.remove('info', 'ok', 'error');
    statusEl.classList.add(tone);
}

function badge(flag) {
    if (Number(flag) === 1) {
        return '<span class="badge alert">Anomaly</span>';
    }
    return '<span class="badge ok">Normal</span>';
}

function renderMetrics(metrics, threshold, totalRows, viewRows) {
    const cards = [
        ['Accuracy', metrics.accuracy],
        ['Precision', metrics.precision],
        ['Recall', metrics.recall],
        ['F1', metrics.f1],
        ['Threshold', threshold],
        ['Shown', `${viewRows}/${totalRows}`],
    ];

    metricsGrid.innerHTML = cards
        .map(([k, v]) => `<div class="metric"><div class="k">${k}</div><div class="v">${v}</div></div>`)
        .join('');
}

function renderTable(rows) {
    const predicted = rows.filter((r) => r.predicted_anomaly === 1);
    const slice = predicted.slice(-20).reverse();

    rowsBody.innerHTML = slice
        .map((r) => {
            const trClass = r.predicted_anomaly === 1 ? 'predicted-anomaly' : '';
            return `
                <tr class="${trClass}">
                    <td>${r.timestamp}</td>
                    <td>${r.sleep_hours}</td>
                    <td>${r.activity_minutes}</td>
                    <td>${badge(r.actual_anomaly)}</td>
                    <td>${badge(r.predicted_anomaly)}</td>
                    <td>${r.anomaly_probability}</td>
                </tr>`;
        })
        .join('');

    if (!slice.length) {
        rowsBody.innerHTML = '<tr><td colspan="6">No predicted anomalies in this window.</td></tr>';
    }
}

function renderKpis(rows, metrics) {
    const predictedCount = rows.filter((r) => r.predicted_anomaly === 1).length;
    const avgProb = rows.length
        ? rows.reduce((sum, r) => sum + Number(r.anomaly_probability), 0) / rows.length
        : 0;

    kpiPredicted.textContent = String(predictedCount);
    kpiAvgProb.textContent = avgProb.toFixed(3);
    kpiSamples.textContent = String(metrics.samples_test ?? rows.length);
}

function renderCharts(rows, threshold) {
    const labels = rows.map((r) => r.timestamp);
    const probs = rows.map((r) => r.anomaly_probability);
    const thresholdLine = rows.map(() => threshold);
    const sleep = rows.map((r) => r.sleep_hours);
    const activity = rows.map((r) => r.activity_minutes);

    const anomalyPoints = rows.map((r, i) => (r.predicted_anomaly ? { x: labels[i], y: probs[i] } : null)).filter(Boolean);

    if (probChart) probChart.destroy();
    if (signalChart) signalChart.destroy();

    probChart = new Chart(document.getElementById('probChart'), {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Anomaly Probability',
                    data: probs,
                    borderColor: '#0f766e',
                    backgroundColor: 'rgba(15,118,110,0.14)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.24,
                    pointRadius: 0,
                },
                {
                    label: 'Threshold',
                    data: thresholdLine,
                    borderColor: '#ea580c',
                    borderWidth: 1.5,
                    borderDash: [6, 4],
                    pointRadius: 0,
                },
                {
                    type: 'scatter',
                    label: 'Predicted Anomaly',
                    data: anomalyPoints,
                    backgroundColor: '#be123c',
                    pointRadius: 4,
                },
            ],
        },
        options: {
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 0,
                    max: 1,
                },
                x: {
                    ticks: {
                        maxTicksLimit: 10,
                    },
                },
            },
            plugins: {
                legend: { position: 'bottom' },
            },
        },
    });

    signalChart = new Chart(document.getElementById('signalChart'), {
        data: {
            labels,
            datasets: [
                {
                    type: 'line',
                    label: 'Sleep Hours',
                    data: sleep,
                    borderColor: '#1d4ed8',
                    backgroundColor: 'rgba(29,78,216,0.12)',
                    borderWidth: 2,
                    tension: 0.25,
                    yAxisID: 'ySleep',
                    pointRadius: 0,
                },
                {
                    type: 'bar',
                    label: 'Activity Minutes',
                    data: activity,
                    backgroundColor: 'rgba(251,146,60,0.45)',
                    borderColor: '#f97316',
                    borderWidth: 1,
                    yAxisID: 'yActivity',
                },
            ],
        },
        options: {
            maintainAspectRatio: false,
            scales: {
                ySleep: {
                    type: 'linear',
                    position: 'left',
                    title: { display: true, text: 'Sleep Hours' },
                },
                yActivity: {
                    type: 'linear',
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    title: { display: true, text: 'Activity Minutes' },
                },
                x: {
                    ticks: {
                        maxTicksLimit: 10,
                    },
                },
            },
            plugins: {
                legend: { position: 'bottom' },
            },
        },
    });
}

async function runPrediction() {
    if (isBusy) {
        return;
    }
    isBusy = true;
    runBtn.disabled = true;
    reloadBtn.disabled = true;
    setStatus('Training LSTM and generating predictions...', 'info');
    try {
        const res = await fetch('/api/predictions');
        const payload = await res.json();
        if (!res.ok || !payload.success) {
            throw new Error('Prediction request failed.');
        }

        renderMetrics(payload.metrics, payload.threshold, payload.total_rows, payload.rows.length);
        renderKpis(payload.rows, payload.metrics);
        renderTable(payload.rows);
        renderCharts(payload.rows, payload.threshold);
        setStatus('Complete. Visualization updated.', 'ok');
    } catch (err) {
        setStatus(`Error: ${err.message}`, 'error');
    } finally {
        isBusy = false;
        runBtn.disabled = false;
        reloadBtn.disabled = false;
    }
}

async function regenerateData() {
    if (isBusy) {
        return;
    }
    isBusy = true;
    runBtn.disabled = true;
    reloadBtn.disabled = true;
    setStatus('Regenerating sleep/activity dataset...', 'info');
    try {
        const res = await fetch('/api/regenerate', { method: 'POST' });
        if (!res.ok) throw new Error('Regenerate failed.');
        setStatus('Dataset regenerated. Run prediction again.', 'ok');
    } catch (err) {
        setStatus(`Error: ${err.message}`, 'error');
    } finally {
        isBusy = false;
        runBtn.disabled = false;
        reloadBtn.disabled = false;
    }
}

runBtn.addEventListener('click', runPrediction);
reloadBtn.addEventListener('click', regenerateData);
