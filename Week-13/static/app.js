const API = 'http://localhost:5012/api';
const L = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: 'Inter,sans-serif', color: '#e5e5e5', size: 12 }, xaxis: { gridcolor: 'rgba(255,255,255,.06)' }, yaxis: { gridcolor: 'rgba(255,255,255,.06)' }, margin: { t: 30, r: 30, b: 50, l: 60 }, hovermode: 'closest' };
const C = { responsive: true, displayModeBar: false };

function KPICard({ icon, value, label }) {
    return React.createElement('div', { className: 'card kpi' },
        React.createElement('div', { className: 'ki' }, React.createElement('i', { className: icon })),
        React.createElement('div', { className: 'kv' }, value),
        React.createElement('div', { className: 'kl' }, label));
}

function ChartCard({ title, icon, id, sub }) {
    return React.createElement('div', { className: 'card' },
        React.createElement('h2', { className: 'card-title' }, React.createElement('i', { className: icon }), ` ${title}`),
        sub ? React.createElement('p', { className: 'sub' }, sub) : null,
        React.createElement('div', { id, className: 'chart-container', style: { minHeight: '300px' } }));
}

function App() {
    const [data, setData] = React.useState(null);
    const [summary, setSummary] = React.useState(null);

    React.useEffect(() => {
        fetch(`${API}/summary`).then(r => r.json()).then(d => { if (d.success) setSummary(d.summary) });
        fetch(`${API}/timeseries`).then(r => r.json()).then(d => { if (d.success) setData(d.data) });
    }, []);

    React.useEffect(() => {
        if (!data) return;
        // Daily Activity Trends
        Plotly.newPlot('activityChart', [
            { x: data.map(d => d.date), y: data.map(d => d.steps), type: 'scatter', mode: 'lines', name: 'Steps', line: { color: 'rgb(52,211,153)', width: 2 } },
            { x: data.map(d => d.date), y: data.map(d => d.calories), type: 'scatter', mode: 'lines', name: 'Calories', line: { color: 'rgb(251,191,36)', width: 2 }, yaxis: 'y2' }
        ], { ...L, yaxis: { ...L.yaxis, title: 'Steps' }, yaxis2: { title: 'Calories', overlaying: 'y', side: 'right', titlefont: { color: '#e5e5e5' }, tickfont: { color: '#e5e5e5' }, gridcolor: 'rgba(255,255,255,.03)' }, showlegend: true, legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,.3)' } }, C);

        // Heart Rate Trend
        Plotly.newPlot('hrChart', [{ x: data.map(d => d.date), y: data.map(d => d.heart_rate), type: 'scatter', mode: 'lines+markers', line: { color: 'rgb(244,114,182)', width: 2 }, marker: { size: 3 } }], { ...L, yaxis: { ...L.yaxis, title: 'BPM', range: [55, 95] } }, C);

        // Sleep vs Steps
        Plotly.newPlot('sleepChart', [{ x: data.map(d => d.sleep_hours), y: data.map(d => d.steps), type: 'scatter', mode: 'markers', marker: { color: 'rgba(168,85,247,.5)', size: 5 } }], { ...L, xaxis: { ...L.xaxis, title: 'Sleep (hours)' }, yaxis: { ...L.yaxis, title: 'Steps' } }, C);

        // Weekly Pattern
        const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
        const byDay = {}; data.forEach(d => { const dow = new Date(d.date).getDay(); const k = dow === 0 ? 6 : dow - 1; if (!byDay[k]) byDay[k] = { s: 0, c: 0 }; byDay[k].s += d.steps; byDay[k].c++ });
        Plotly.newPlot('weeklyChart', [{ x: days, y: days.map((_, i) => byDay[i] ? Math.round(byDay[i].s / byDay[i].c) : 0), type: 'bar', marker: { color: 'rgb(56,189,248)' } }], { ...L, yaxis: { ...L.yaxis, title: 'Avg Steps' } }, C);
    }, [data]);

    return React.createElement('div', null,
        React.createElement('header', { className: 'header' },
            React.createElement('div', { className: 'header-content' },
                React.createElement('div', null,
                    React.createElement('h1', null, React.createElement('i', { className: 'fa-solid fa-chart-bar' }), ' Dashboard Visualization'),
                    React.createElement('p', { className: 'sub' }, 'React · Daily Activity Trends · Decrypted Wearable Data')))),
        React.createElement('main', { className: 'container' },
            React.createElement('div', { className: 'dashboard-grid four-col' },
                summary ? [
                    React.createElement(KPICard, { key: 'd', icon: 'fa-solid fa-calendar', value: summary.days, label: 'Days Tracked' }),
                    React.createElement(KPICard, { key: 's', icon: 'fa-solid fa-shoe-prints', value: summary.avg_steps, label: 'Avg Steps' }),
                    React.createElement(KPICard, { key: 'h', icon: 'fa-solid fa-heart-pulse', value: summary.avg_hr, label: 'Avg HR (BPM)' }),
                    React.createElement(KPICard, { key: 'c', icon: 'fa-solid fa-fire', value: summary.avg_calories, label: 'Avg Calories' })
                ] : React.createElement('div', { className: 'loading' }, React.createElement('div', { className: 'spinner' }))),
            React.createElement('div', { className: 'dashboard-grid' },
                React.createElement(ChartCard, { title: 'Daily Activity Trends', icon: 'fa-solid fa-chart-line', id: 'activityChart', sub: 'Steps & Calories (decrypted)' })),
            React.createElement('div', { className: 'dashboard-grid' },
                React.createElement(ChartCard, { title: 'Heart Rate Trend', icon: 'fa-solid fa-heart-pulse', id: 'hrChart' }),
                React.createElement(ChartCard, { title: 'Sleep vs Steps', icon: 'fa-solid fa-bed', id: 'sleepChart' })),
            React.createElement('div', { className: 'dashboard-grid' },
                React.createElement(ChartCard, { title: 'Weekly Pattern', icon: 'fa-solid fa-calendar-week', id: 'weeklyChart' }))));
}

ReactDOM.createRoot(document.getElementById('root')).render(React.createElement(App));
