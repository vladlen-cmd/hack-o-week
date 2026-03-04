const API_BASE = 'http://localhost:5002/api';
const REFRESH_INTERVAL = 30000;

const plotlyLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {
        family: 'Inter, sans-serif',
        color: '#e5e5e5',
        size: 12
    },
    xaxis: {
        gridcolor: 'rgba(255,255,255,0.06)',
        zerolinecolor: 'rgba(255,255,255,0.08)'
    },
    yaxis: {
        gridcolor: 'rgba(255,255,255,0.06)',
        zerolinecolor: 'rgba(255,255,255,0.08)'
    },
    margin: { t: 30, r: 30, b: 50, l: 60 },
    hovermode: 'closest'
};

const plotlyConfig = {
    responsive: true,
    displayModeBar: false
};

// ===== DATA FETCHERS =====

async function fetchJSON(endpoint) {
    try {
        const res = await fetch(`${API_BASE}/${endpoint}`);
        const data = await res.json();
        if (data.success) return data;
        throw new Error(data.error);
    } catch (err) {
        console.error(`Error fetching ${endpoint}:`, err);
        return null;
    }
}

// ===== CARD UPDATERS =====

function updatePredictionCard(data) {
    const el = document.getElementById('predictionContent');
    if (!data) { el.innerHTML = '<p style="color:var(--text-muted)">No prediction available</p>'; return; }

    const { prediction, current } = data;
    const examTag = current.is_exam
        ? '<span class="status-tag exam">📕 Exam Period</span>'
        : '<span class="status-tag normal">📗 Normal Period</span>';

    el.innerHTML = `
        <div class="prediction-label">Forecasted Electricity</div>
        <div class="prediction-value">${prediction.prediction} <span style="font-size:1.2rem">kWh</span></div>
        ${examTag}
        <div class="confidence-bar">
            <div class="ci-item">
                <div class="ci-label">Lower</div>
                <div class="ci-value">${prediction.lower_bound}</div>
            </div>
            <div class="ci-item">
                <div class="ci-label">Confidence</div>
                <div class="ci-value">${prediction.confidence_level}%</div>
            </div>
            <div class="ci-item">
                <div class="ci-label">Upper</div>
                <div class="ci-value">${prediction.upper_bound}</div>
            </div>
        </div>
    `;
}

function updateExamComparison(data) {
    const el = document.getElementById('examComparison');
    if (!data) { el.innerHTML = '<p style="color:var(--text-muted)">No data</p>'; return; }

    const { exam, normal } = data.comparison;
    const pctIncrease = (((exam.avg_electricity - normal.avg_electricity) / normal.avg_electricity) * 100).toFixed(1);

    el.innerHTML = `
        <div class="comparison-grid">
            <div class="comparison-item exam">
                <div class="comp-label">📕 Exam Avg</div>
                <div class="comp-value">${exam.avg_electricity}</div>
                <div class="comp-sub">kWh / hour</div>
            </div>
            <div class="comparison-item normal">
                <div class="comp-label">📗 Normal Avg</div>
                <div class="comp-value">${normal.avg_electricity}</div>
                <div class="comp-sub">kWh / hour</div>
            </div>
            <div class="comparison-item exam">
                <div class="comp-label">📕 Peak Occupancy</div>
                <div class="comp-value">${exam.max_occupancy}</div>
                <div class="comp-sub">students</div>
            </div>
            <div class="comparison-item normal">
                <div class="comp-label">📗 Peak Occupancy</div>
                <div class="comp-value">${normal.max_occupancy}</div>
                <div class="comp-sub">students</div>
            </div>
        </div>
        <div style="text-align:center; margin-top:var(--spacing-sm); color:var(--accent-exam); font-weight:600; font-size:0.9rem;">
            <i class="fa-solid fa-bolt"></i> ${pctIncrease}% higher during exams
        </div>
    `;
}

function updateModelMetrics(data) {
    const el = document.getElementById('modelMetrics');
    if (!data) { el.innerHTML = '<p style="color:var(--text-muted)">No metrics</p>'; return; }

    const m = data.metrics;
    el.innerHTML = `
        <div class="metrics-grid">
            <div class="metric-item">
                <div class="metric-label">R² Score</div>
                <div class="metric-value good">${m.r2}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">RMSE</div>
                <div class="metric-value">${m.rmse}<span class="metric-unit">kWh</span></div>
            </div>
            <div class="metric-item">
                <div class="metric-label">MAE</div>
                <div class="metric-value">${m.mae}<span class="metric-unit">kWh</span></div>
            </div>
            <div class="metric-item">
                <div class="metric-label">MAPE</div>
                <div class="metric-value">${m.mape}<span class="metric-unit">%</span></div>
            </div>
        </div>
    `;
}

// ===== CHARTS =====

function createTimeSeriesChart(data) {
    const examData = data.filter(d => d.is_exam_period === 1);
    const normalData = data.filter(d => d.is_exam_period === 0);

    const traces = [
        {
            x: normalData.map(d => d.timestamp),
            y: normalData.map(d => d.electricity_kwh),
            type: 'scatter', mode: 'lines',
            name: 'Normal Period',
            line: { color: 'rgb(52, 211, 153)', width: 1.5 }
        },
        {
            x: examData.map(d => d.timestamp),
            y: examData.map(d => d.electricity_kwh),
            type: 'scatter', mode: 'lines',
            name: 'Exam Period',
            line: { color: 'rgb(244, 114, 182)', width: 2 }
        }
    ];

    // Add exam-period background shading
    const shapes = [];
    let examStart = null;
    for (let i = 0; i < data.length; i++) {
        if (data[i].is_exam_period === 1 && examStart === null) {
            examStart = data[i].timestamp;
        }
        if ((data[i].is_exam_period === 0 || i === data.length - 1) && examStart !== null) {
            shapes.push({
                type: 'rect', xref: 'x', yref: 'paper',
                x0: examStart, x1: data[i].timestamp,
                y0: 0, y1: 1,
                fillcolor: 'rgba(244, 114, 182, 0.08)',
                line: { width: 0 }
            });
            examStart = null;
        }
    }

    const layout = {
        ...plotlyLayout,
        shapes,
        xaxis: { ...plotlyLayout.xaxis, title: 'Date' },
        yaxis: { ...plotlyLayout.yaxis, title: 'Electricity (kWh)' },
        showlegend: true,
        legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,0.3)' }
    };

    Plotly.newPlot('timeSeriesChart', traces, layout, plotlyConfig);
}

function createHourlyComparisonChart(data) {
    const examHourly = {};
    const normalHourly = {};

    data.forEach(d => {
        const hour = new Date(d.timestamp).getHours();
        const bucket = d.is_exam_period === 1 ? examHourly : normalHourly;
        if (!bucket[hour]) bucket[hour] = { total: 0, count: 0 };
        bucket[hour].total += d.electricity_kwh;
        bucket[hour].count++;
    });

    const hours = Array.from({ length: 24 }, (_, i) => i);
    const examAvg = hours.map(h => examHourly[h] ? examHourly[h].total / examHourly[h].count : 0);
    const normalAvg = hours.map(h => normalHourly[h] ? normalHourly[h].total / normalHourly[h].count : 0);

    const traces = [
        {
            x: hours, y: normalAvg, type: 'bar', name: 'Normal',
            marker: { color: 'rgba(52, 211, 153, 0.7)', line: { color: 'rgb(52, 211, 153)', width: 1 } }
        },
        {
            x: hours, y: examAvg, type: 'bar', name: 'Exam',
            marker: { color: 'rgba(244, 114, 182, 0.7)', line: { color: 'rgb(244, 114, 182)', width: 1 } }
        }
    ];

    const layout = {
        ...plotlyLayout,
        barmode: 'group',
        xaxis: { ...plotlyLayout.xaxis, title: 'Hour of Day', tickmode: 'linear', dtick: 2 },
        yaxis: { ...plotlyLayout.yaxis, title: 'Avg Electricity (kWh)' },
        showlegend: true,
        legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,0.3)' }
    };

    Plotly.newPlot('hourlyComparisonChart', traces, layout, plotlyConfig);
}

function createFeatureImportanceChart(featureData) {
    const sorted = Object.entries(featureData).sort((a, b) => b[1] - a[1]);
    const names = sorted.map(([n]) => n.replace(/_/g, ' '));
    const values = sorted.map(([, v]) => v);

    const trace = {
        x: values, y: names, type: 'bar', orientation: 'h',
        marker: {
            color: values,
            colorscale: [[0, 'rgb(52, 211, 153)'], [1, 'rgb(59, 130, 246)']],
            line: { color: 'rgba(255,255,255,0.15)', width: 1 }
        }
    };

    const layout = {
        ...plotlyLayout,
        xaxis: { ...plotlyLayout.xaxis, title: 'Importance' },
        yaxis: { ...plotlyLayout.yaxis, automargin: true },
        margin: { ...plotlyLayout.margin, l: 180 }
    };

    Plotly.newPlot('featureImportanceChart', [trace], layout, plotlyConfig);
}

function createCorrelationChart(data) {
    const examPts = data.filter(d => d.is_exam_period === 1);
    const normalPts = data.filter(d => d.is_exam_period === 0);

    const traces = [
        {
            x: normalPts.map(d => d.occupancy), y: normalPts.map(d => d.electricity_kwh),
            type: 'scatter', mode: 'markers', name: 'Normal',
            marker: { color: 'rgba(52, 211, 153, 0.5)', size: 5 }
        },
        {
            x: examPts.map(d => d.occupancy), y: examPts.map(d => d.electricity_kwh),
            type: 'scatter', mode: 'markers', name: 'Exam',
            marker: { color: 'rgba(244, 114, 182, 0.5)', size: 5 }
        }
    ];

    const layout = {
        ...plotlyLayout,
        xaxis: { ...plotlyLayout.xaxis, title: 'Occupancy (students)' },
        yaxis: { ...plotlyLayout.yaxis, title: 'Electricity (kWh)' },
        showlegend: true,
        legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,0.3)' }
    };

    Plotly.newPlot('correlationChart', traces, layout, plotlyConfig);
}

function createHeatmapChart(data) {
    const dayNames = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    const grid = Array.from({ length: 7 }, () => Array(24).fill(null));
    const counts = Array.from({ length: 7 }, () => Array(24).fill(0));

    data.forEach(d => {
        const dt = new Date(d.timestamp);
        const dow = dt.getDay() === 0 ? 6 : dt.getDay() - 1; // Mon=0
        const hour = dt.getHours();
        grid[dow][hour] = (grid[dow][hour] || 0) + d.electricity_kwh;
        counts[dow][hour]++;
    });

    const z = grid.map((row, di) =>
        row.map((val, hi) => counts[di][hi] > 0 ? val / counts[di][hi] : 0)
    );

    const trace = {
        z, x: Array.from({ length: 24 }, (_, i) => `${i}:00`), y: dayNames,
        type: 'heatmap',
        colorscale: [[0, 'rgb(15, 23, 42)'], [0.5, 'rgb(52, 211, 153)'], [1, 'rgb(244, 114, 182)']],
        colorbar: { title: 'kWh', thickness: 10 }
    };

    const layout = {
        ...plotlyLayout,
        xaxis: { ...plotlyLayout.xaxis, title: 'Hour' },
        yaxis: { ...plotlyLayout.yaxis }
    };

    Plotly.newPlot('heatmapChart', [trace], layout, plotlyConfig);
}

// ===== MAIN =====

function updateTimestamp() {
    document.getElementById('lastUpdate').textContent = `Updated: ${new Date().toLocaleTimeString()}`;
}

async function loadDashboard() {
    console.log('Loading dashboard...');

    const [historicalRes, predictionRes, statsRes, compRes] = await Promise.all([
        fetchJSON('historical'),
        fetchJSON('predict'),
        fetchJSON('stats'),
        fetchJSON('exam-comparison')
    ]);

    if (predictionRes) updatePredictionCard(predictionRes);
    if (compRes) updateExamComparison(compRes);
    if (statsRes) {
        updateModelMetrics(statsRes);
        createFeatureImportanceChart(statsRes.feature_importance);
    }
    if (historicalRes) {
        createTimeSeriesChart(historicalRes.data);
        createHourlyComparisonChart(historicalRes.data);
        createCorrelationChart(historicalRes.data);
        createHeatmapChart(historicalRes.data);
    }

    updateTimestamp();
    console.log('Dashboard loaded.');
}

document.addEventListener('DOMContentLoaded', () => {
    loadDashboard();
    setInterval(loadDashboard, REFRESH_INTERVAL);
});
