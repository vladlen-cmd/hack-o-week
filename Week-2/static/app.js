const API_BASE = 'http://localhost:5001/api';
const REFRESH_INTERVAL = 30000;
let historicalData = null;
let predictionData = null;

const plotlyLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {
        family: 'Inter, sans-serif',
        color: '#e5e5e5'
    },
    xaxis: {
        gridcolor: 'rgba(255,255,255,0.1)',
        zerolinecolor: 'rgba(255,255,255,0.1)'
    },
    yaxis: {
        gridcolor: 'rgba(255,255,255,0.1)',
        zerolinecolor: 'rgba(255,255,255,0.1)'
    },
    margin: { t: 30, r: 30, b: 50, l: 60 },
    hovermode: 'closest'
};

const plotlyConfig = {
    responsive: true,
    displayModeBar: false
};

async function fetchHistoricalData() {
    try {
        const response = await fetch(`${API_BASE}/historical`);
        const result = await response.json();

        if (result.success) {
            historicalData = result.data;
            return historicalData;
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        console.error('Error fetching historical data:', error);
        return null;
    }
}

// Fetch prediction
async function fetchPrediction() {
    try {
        const response = await fetch(`${API_BASE}/predict`);
        const result = await response.json();

        if (result.success) {
            predictionData = result;
            return predictionData;
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        console.error('Error fetching prediction:', error);
        return null;
    }
}

// Update prediction card
function updatePredictionCard(data) {
    const container = document.getElementById('predictionContent');

    if (!data) {
        container.innerHTML = '<p style="color: var(--text-muted);">No prediction available</p>';
        return;
    }

    const { prediction, current } = data;

    container.innerHTML = `
        <div class="prediction-label">Forecasted Electricity Draw</div>
        <div class="prediction-value">${prediction.value} <span style="font-size: 1.5rem;">kWh</span></div>
        
        <div class="confidence-interval">
            <div class="ci-item">
                <div class="ci-label">Lower Bound</div>
                <div class="ci-value">${prediction.lower_ci} kWh</div>
            </div>
            <div class="ci-item">
                <div class="ci-label">Confidence</div>
                <div class="ci-value">${prediction.confidence_level}%</div>
            </div>
            <div class="ci-item">
                <div class="ci-label">Upper Bound</div>
                <div class="ci-value">${prediction.upper_ci} kWh</div>
            </div>
        </div>
    `;
}

// Update current stats
function updateCurrentStats(data) {
    const container = document.getElementById('currentStats');

    if (!data) {
        container.innerHTML = '<p style="color: var(--text-muted);">No data available</p>';
        return;
    }

    const { current } = data;

    container.innerHTML = `
        <div class="stat-item">
            <div class="stat-label">Current Occupancy</div>
            <div class="stat-value">${current.occupancy}<span class="stat-unit">devices</span></div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Current Electricity</div>
            <div class="stat-value">${current.electricity}<span class="stat-unit">kWh</span></div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Last Reading</div>
            <div class="stat-value" style="font-size: 0.875rem;">${new Date(current.timestamp).toLocaleTimeString()}</div>
        </div>
    `;
}

// Create time series chart
function createTimeSeriesChart(data, prediction) {
    const timestamps = data.map(d => d.timestamp);
    const electricity = data.map(d => d.electricity_kwh);

    // Historical trace
    const historicalTrace = {
        x: timestamps,
        y: electricity,
        type: 'scatter',
        mode: 'lines',
        name: 'Historical',
        line: {
            color: 'rgb(59, 130, 246)',
            width: 2
        }
    };

    const traces = [historicalTrace];

    // Add prediction if available
    if (prediction) {
        const lastTimestamp = new Date(timestamps[timestamps.length - 1]);
        const nextTimestamp = new Date(lastTimestamp.getTime() + 3600000); // +1 hour

        // Prediction point
        traces.push({
            x: [nextTimestamp],
            y: [prediction.prediction.value],
            type: 'scatter',
            mode: 'markers',
            name: 'Forecast',
            marker: {
                color: 'rgb(168, 85, 247)',
                size: 10,
                symbol: 'star'
            }
        });

        // Confidence interval
        traces.push({
            x: [nextTimestamp, nextTimestamp],
            y: [prediction.prediction.lower_ci, prediction.prediction.upper_ci],
            type: 'scatter',
            mode: 'lines',
            name: '95% CI',
            line: {
                color: 'rgba(168, 85, 247, 0.3)',
                width: 0
            },
            fill: 'tonexty',
            fillcolor: 'rgba(168, 85, 247, 0.2)'
        });
    }

    const layout = {
        ...plotlyLayout,
        xaxis: {
            ...plotlyLayout.xaxis,
            title: 'Time'
        },
        yaxis: {
            ...plotlyLayout.yaxis,
            title: 'Electricity (kWh)'
        },
        showlegend: true,
        legend: {
            x: 0,
            y: 1,
            bgcolor: 'rgba(0,0,0,0.3)'
        }
    };

    Plotly.newPlot('timeSeriesChart', traces, layout, plotlyConfig);
}

// Create correlation chart
function createCorrelationChart(data) {
    const occupancy = data.map(d => d.occupancy);
    const electricity = data.map(d => d.electricity_kwh);

    const trace = {
        x: occupancy,
        y: electricity,
        type: 'scatter',
        mode: 'markers',
        marker: {
            color: electricity,
            colorscale: 'Viridis',
            size: 8,
            opacity: 0.6,
            colorbar: {
                title: 'kWh',
                thickness: 10
            }
        },
        name: 'Data Points'
    };

    const layout = {
        ...plotlyLayout,
        xaxis: {
            ...plotlyLayout.xaxis,
            title: 'Occupancy (devices)'
        },
        yaxis: {
            ...plotlyLayout.yaxis,
            title: 'Electricity (kWh)'
        },
        showlegend: false
    };

    Plotly.newPlot('correlationChart', [trace], layout, plotlyConfig);
}

// Create hourly pattern chart
function createHourlyPatternChart(data) {
    // Group by hour and calculate average
    const hourlyData = {};

    data.forEach(d => {
        const hour = new Date(d.timestamp).getHours();
        if (!hourlyData[hour]) {
            hourlyData[hour] = { total: 0, count: 0 };
        }
        hourlyData[hour].total += d.electricity_kwh;
        hourlyData[hour].count += 1;
    });

    const hours = Object.keys(hourlyData).map(Number).sort((a, b) => a - b);
    const avgElectricity = hours.map(h => hourlyData[h].total / hourlyData[h].count);

    const trace = {
        x: hours,
        y: avgElectricity,
        type: 'bar',
        marker: {
            color: avgElectricity,
            colorscale: 'Plasma',
            line: {
                color: 'rgba(255,255,255,0.2)',
                width: 1
            }
        },
        name: 'Average Usage'
    };

    const layout = {
        ...plotlyLayout,
        xaxis: {
            ...plotlyLayout.xaxis,
            title: 'Hour of Day',
            tickmode: 'linear',
            tick0: 0,
            dtick: 2
        },
        yaxis: {
            ...plotlyLayout.yaxis,
            title: 'Avg Electricity (kWh)'
        },
        showlegend: false
    };

    Plotly.newPlot('hourlyPatternChart', [trace], layout, plotlyConfig);
}

// Update last update timestamp
function updateTimestamp() {
    const now = new Date();
    document.getElementById('lastUpdate').textContent =
        `Updated: ${now.toLocaleTimeString()}`;
}

// Load all data and update dashboard
async function loadDashboard() {
    console.log('Loading dashboard data...');

    // Fetch data
    const [historical, prediction] = await Promise.all([
        fetchHistoricalData(),
        fetchPrediction()
    ]);

    if (!historical) {
        console.error('Failed to load historical data');
        return;
    }

    // Update UI
    updatePredictionCard(prediction);
    updateCurrentStats(prediction);

    // Create charts
    createTimeSeriesChart(historical, prediction);
    createCorrelationChart(historical);
    createHourlyPatternChart(historical);

    // Update timestamp
    updateTimestamp();

    console.log('Dashboard loaded successfully');
}

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing dashboard...');
    loadDashboard();

    // Auto-refresh
    setInterval(loadDashboard, REFRESH_INTERVAL);
});
