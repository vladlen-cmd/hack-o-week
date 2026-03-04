const API_BASE = 'http://localhost:5003/api';
const REFRESH_INTERVAL = 30000;

const MEAL_COLORS = {
    'Off-hours': 'rgb(100, 116, 139)',
    'Breakfast': 'rgb(234, 179, 8)',
    'Lunch': 'rgb(239, 68, 68)',
    'Snacks': 'rgb(52, 211, 153)',
    'Dinner': 'rgb(168, 85, 247)'
};

const MEAL_PERIOD_MAP = { 0: 'Off-hours', 1: 'Breakfast', 2: 'Lunch', 3: 'Snacks', 4: 'Dinner' };
const MEAL_EMOJIS = { 'Off-hours': '⏸️', 'Breakfast': '🥐', 'Lunch': '🍛', 'Snacks': '☕', 'Dinner': '🍝' };

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

const plotlyConfig = { responsive: true, displayModeBar: false };

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
    if (!data) { el.innerHTML = '<p style="color:var(--text-muted)">No prediction</p>'; return; }

    const { prediction, current } = data;
    const mealKey = current.meal_period.toLowerCase().replace('-', '-');
    const mealClass = mealKey.replace(' ', '-');

    el.innerHTML = `
        <div class="prediction-label">Forecasted Electricity</div>
        <div class="prediction-value">${prediction.prediction} <span style="font-size:1.2rem">kWh</span></div>
        <span class="meal-tag ${mealClass}">${MEAL_EMOJIS[current.meal_period] || ''} ${current.meal_period}</span>
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

function updateMealBreakdown(data) {
    const el = document.getElementById('mealBreakdown');
    if (!data) { el.innerHTML = '<p style="color:var(--text-muted)">No data</p>'; return; }

    const meals = data.meal_analysis;
    const order = ['Breakfast', 'Lunch', 'Snacks', 'Dinner', 'Off-hours'];
    let html = '<div class="meal-list">';

    order.forEach(name => {
        const m = meals[name];
        if (!m) return;
        const emoji = MEAL_EMOJIS[name];
        html += `
            <div class="meal-item">
                <span class="meal-name">${emoji} ${name}</span>
                <span>
                    <span class="meal-val">${m.avg_electricity}</span>
                    <span class="meal-sub">kWh avg</span>
                </span>
            </div>
        `;
    });

    html += '</div>';
    el.innerHTML = html;
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
    // Group by meal period for coloring
    const groups = {};
    data.forEach(d => {
        const name = MEAL_PERIOD_MAP[d.meal_period];
        if (!groups[name]) groups[name] = { x: [], y: [] };
        groups[name].x.push(d.timestamp);
        groups[name].y.push(d.electricity_kwh);
    });

    // Use a single line for the overall trend, then overlay colored markers
    const traces = [
        {
            x: data.map(d => d.timestamp),
            y: data.map(d => d.electricity_kwh),
            type: 'scatter', mode: 'lines',
            name: 'Electricity',
            line: { color: 'rgba(245, 158, 11, 0.4)', width: 1 },
            showlegend: false
        }
    ];

    // Add colored scatter for each meal period
    Object.entries(groups).forEach(([name, pts]) => {
        traces.push({
            x: pts.x, y: pts.y,
            type: 'scatter', mode: 'markers',
            name,
            marker: { color: MEAL_COLORS[name], size: 3, opacity: 0.7 }
        });
    });

    const layout = {
        ...plotlyLayout,
        xaxis: { ...plotlyLayout.xaxis, title: 'Date' },
        yaxis: { ...plotlyLayout.yaxis, title: 'Electricity (kWh)' },
        showlegend: true,
        legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,0.3)' }
    };

    Plotly.newPlot('timeSeriesChart', traces, layout, plotlyConfig);
}

function createMealBarChart(mealData) {
    const order = ['Breakfast', 'Lunch', 'Snacks', 'Dinner', 'Off-hours'];
    const names = order.filter(n => mealData[n]);
    const avgVals = names.map(n => mealData[n].avg_electricity);
    const maxVals = names.map(n => mealData[n].max_electricity);
    const colors = names.map(n => MEAL_COLORS[n]);

    const traces = [
        {
            x: names, y: avgVals, type: 'bar', name: 'Average',
            marker: { color: colors.map(c => c.replace('rgb', 'rgba').replace(')', ', 0.7)')), line: { color: colors, width: 1 } }
        },
        {
            x: names, y: maxVals, type: 'bar', name: 'Peak',
            marker: { color: colors.map(c => c.replace('rgb', 'rgba').replace(')', ', 0.3)')), line: { color: colors, width: 1 } }
        }
    ];

    const layout = {
        ...plotlyLayout,
        barmode: 'group',
        xaxis: { ...plotlyLayout.xaxis },
        yaxis: { ...plotlyLayout.yaxis, title: 'Electricity (kWh)' },
        showlegend: true,
        legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,0.3)' }
    };

    Plotly.newPlot('mealBarChart', traces, layout, plotlyConfig);
}

function createFeatureImportanceChart(featureData) {
    const sorted = Object.entries(featureData).sort((a, b) => b[1] - a[1]);
    const names = sorted.map(([n]) => n.replace(/_/g, ' '));
    const values = sorted.map(([, v]) => v);

    const trace = {
        x: values, y: names, type: 'bar', orientation: 'h',
        marker: {
            color: values,
            colorscale: [[0, 'rgb(245, 158, 11)'], [1, 'rgb(239, 68, 68)']],
            line: { color: 'rgba(255,255,255,0.1)', width: 1 }
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
    const groups = {};
    data.forEach(d => {
        const name = MEAL_PERIOD_MAP[d.meal_period];
        if (!groups[name]) groups[name] = { x: [], y: [] };
        groups[name].x.push(d.footfall);
        groups[name].y.push(d.electricity_kwh);
    });

    const traces = Object.entries(groups).map(([name, pts]) => ({
        x: pts.x, y: pts.y,
        type: 'scatter', mode: 'markers',
        name,
        marker: { color: MEAL_COLORS[name], size: 5, opacity: 0.6 }
    }));

    const layout = {
        ...plotlyLayout,
        xaxis: { ...plotlyLayout.xaxis, title: 'Footfall' },
        yaxis: { ...plotlyLayout.yaxis, title: 'Electricity (kWh)' },
        showlegend: true,
        legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,0.3)' }
    };

    Plotly.newPlot('correlationChart', traces, layout, plotlyConfig);
}

function createHeatmapChart(data) {
    const dayNames = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    const grid = Array.from({ length: 7 }, () => Array(24).fill(0));
    const counts = Array.from({ length: 7 }, () => Array(24).fill(0));

    data.forEach(d => {
        const dt = new Date(d.timestamp);
        const dow = dt.getDay() === 0 ? 6 : dt.getDay() - 1;
        const hour = dt.getHours();
        grid[dow][hour] += d.electricity_kwh;
        counts[dow][hour]++;
    });

    const z = grid.map((row, di) =>
        row.map((val, hi) => counts[di][hi] > 0 ? val / counts[di][hi] : 0)
    );

    const trace = {
        z, x: Array.from({ length: 24 }, (_, i) => `${i}:00`), y: dayNames,
        type: 'heatmap',
        colorscale: [[0, 'rgb(30, 20, 15)'], [0.5, 'rgb(245, 158, 11)'], [1, 'rgb(239, 68, 68)']],
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

    const [historicalRes, predictionRes, statsRes, mealRes] = await Promise.all([
        fetchJSON('historical'),
        fetchJSON('predict'),
        fetchJSON('stats'),
        fetchJSON('meal-analysis')
    ]);

    if (predictionRes) updatePredictionCard(predictionRes);
    if (mealRes) {
        updateMealBreakdown(mealRes);
        createMealBarChart(mealRes.meal_analysis);
    }
    if (statsRes) {
        updateModelMetrics(statsRes);
        createFeatureImportanceChart(statsRes.feature_importance);
    }
    if (historicalRes) {
        createTimeSeriesChart(historicalRes.data);
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
