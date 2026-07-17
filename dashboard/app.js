/**
 * Smart Rainfall Prediction - Dashboard Application
 * ===================================================
 * Interactive analytics dashboard with Chart.js, Leaflet.js,
 * and real-time API integration.
 */

// ============================================
// Configuration
// ============================================
const API_BASE = 'http://127.0.0.1:8000';
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// Chart.js global defaults for dark theme
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";

// Color palette
const COLORS = {
    blue: '#3b82f6',
    cyan: '#06b6d4',
    purple: '#8b5cf6',
    green: '#10b981',
    amber: '#f59e0b',
    red: '#ef4444',
    pink: '#ec4899',
    indigo: '#6366f1',
};

const CHART_PALETTE = [
    '#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444',
    '#8b5cf6', '#ec4899', '#6366f1', '#14b8a6', '#f97316',
    '#84cc16', '#eab308'
];

const STATION_COLORS = {
    Mumbai: '#3b82f6',
    Delhi: '#ef4444',
    Chennai: '#10b981',
    Kolkata: '#f59e0b',
    Bangalore: '#8b5cf6',
    Ahmedabad: '#06b6d4',
    Jaipur: '#ec4899',
    Surat: '#6366f1',
};

// Chart instances
let charts = {};
let map = null;
let analyticsData = null;
let dashboardFilters = { years: [], regions: [], stations: [] };
let controlsInitialized = false;

function hexToRgba(hex, alpha) {
    const normalized = hex.replace('#', '');
    const value = normalized.length === 3
        ? normalized.split('').map(char => char + char).join('')
        : normalized;

    const red = parseInt(value.slice(0, 2), 16);
    const green = parseInt(value.slice(2, 4), 16);
    const blue = parseInt(value.slice(4, 6), 16);

    return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function formatStationLabel(label, maxLength = 24) {
    const cleaned = String(label || '').replace(/_/g, ' ');
    if (cleaned.length <= maxLength) return cleaned;
    return `${cleaned.slice(0, maxLength - 1)}…`;
}

function getSeriesColor(index) {
    return CHART_PALETTE[index % CHART_PALETTE.length];
}

function getRainfallColor(value, maxValue) {
    const ratio = maxValue > 0 ? value / maxValue : 0;
    if (ratio >= 0.8) return COLORS.red;
    if (ratio >= 0.6) return COLORS.amber;
    if (ratio >= 0.4) return COLORS.green;
    if (ratio >= 0.2) return COLORS.cyan;
    return COLORS.blue;
}

// ============================================
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    initTabs();
    initPredictionForm();
    checkAPIStatus();
    loadDashboardData();
});

// ============================================
// Particle Background
// ============================================
function initParticles() {
    const container = document.getElementById('particles');
    const count = 40;

    for (let i = 0; i < count; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.width = (Math.random() * 3 + 1) + 'px';
        particle.style.height = particle.style.width;
        particle.style.animationDuration = (Math.random() * 15 + 10) + 's';
        particle.style.animationDelay = (Math.random() * 10) + 's';
        particle.style.opacity = Math.random() * 0.3;

        const colors = ['#3b82f6', '#06b6d4', '#8b5cf6', '#60a5fa'];
        particle.style.background = colors[Math.floor(Math.random() * colors.length)];

        container.appendChild(particle);
    }
}

// ============================================
// Tab Navigation
// ============================================
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;

            // Update active tab
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Update content
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(`content-${tabId}`).classList.add('active');

            // Initialize tab-specific content
            if (tabId === 'geospatial' && !map) {
                setTimeout(initMap, 100);
            }

            // Resize charts
            setTimeout(() => {
                Object.values(charts).forEach(chart => {
                    if (chart && chart.resize) chart.resize();
                });
            }, 100);
        });
    });
}

// ============================================
// API Communication
// ============================================
async function fetchAPI(endpoint) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (error) {
        console.warn(`API call failed: ${endpoint}`, error.message);
        return null;
    }
}

async function checkAPIStatus() {
    const statusEl = document.getElementById('apiStatus');
    const dot = statusEl.querySelector('.status-dot');
    const text = statusEl.querySelector('.status-text');

    try {
        const data = await fetchAPI('/health');
        if (data && data.status) {
            dot.classList.add('online');
            dot.classList.remove('offline');
            text.textContent = 'API Online';
        } else {
            throw new Error('No response');
        }
    } catch {
        dot.classList.add('offline');
        dot.classList.remove('online');
        text.textContent = 'API Offline - Using Demo Data';
        loadDemoData();
    }
}

// ============================================
// Data Loading
// ============================================
async function loadDashboardData() {
    // Try API first
    const summary = await fetchAPI('/analytics/summary');

    if (summary) {
        analyticsData = summary;
        dashboardFilters = summary.filters || dashboardFilters;
        updateStatsCards(summary);
        renderOverviewCharts(summary);
        populateStationDropdowns(summary.stations);
        populateDashboardFilters(dashboardFilters);
        loadOverviewData();
        loadTimeSeriesData();
        loadGeospatialData();
        loadModelMetrics();
    } else {
        loadDemoData();
    }
}

function loadDemoData() {
    // Generate demo data when API is not available
    const demoStations = ['Mumbai', 'Delhi', 'Chennai', 'Kolkata', 'Bangalore', 'Ahmedabad', 'Jaipur', 'Surat'];
    const demoYears = [2021, 2022, 2023];
    const demoRegions = ['North', 'South', 'East', 'West'];
    const demoGeoData = [
        { station: 'Mumbai', latitude: 19.076, longitude: 72.877, region: 'West', avg_rainfall: 8.42, max_rainfall: 187.42, rainy_days_pct: 38.5, avg_temperature: 27.8, avg_humidity: 72.3 },
        { station: 'Delhi', latitude: 28.614, longitude: 77.209, region: 'North', avg_rainfall: 4.32, max_rainfall: 112.34, rainy_days_pct: 24.1, avg_temperature: 25.4, avg_humidity: 52.6 },
        { station: 'Chennai', latitude: 13.083, longitude: 80.271, region: 'South', avg_rainfall: 5.18, max_rainfall: 145.67, rainy_days_pct: 28.9, avg_temperature: 29.2, avg_humidity: 68.4 },
        { station: 'Kolkata', latitude: 22.573, longitude: 88.364, region: 'East', avg_rainfall: 6.14, max_rainfall: 156.89, rainy_days_pct: 32.7, avg_temperature: 26.9, avg_humidity: 71.2 },
        { station: 'Bangalore', latitude: 12.972, longitude: 77.595, region: 'South', avg_rainfall: 4.87, max_rainfall: 98.45, rainy_days_pct: 27.3, avg_temperature: 24.6, avg_humidity: 64.8 },
        { station: 'Ahmedabad', latitude: 23.023, longitude: 72.571, region: 'West', avg_rainfall: 5.63, max_rainfall: 134.56, rainy_days_pct: 22.8, avg_temperature: 28.5, avg_humidity: 54.7 },
        { station: 'Jaipur', latitude: 26.912, longitude: 75.787, region: 'North', avg_rainfall: 3.21, max_rainfall: 89.23, rainy_days_pct: 18.4, avg_temperature: 26.1, avg_humidity: 45.3 },
        { station: 'Surat', latitude: 21.170, longitude: 72.831, region: 'West', avg_rainfall: 7.89, max_rainfall: 167.34, rainy_days_pct: 35.6, avg_temperature: 28.1, avg_humidity: 68.9 },
    ];
    const demoMonthlySeries = demoStations.slice(0, 6).map((station, index) => ({
        station,
        values: MONTHS.map((_, monthIndex) => {
            const seasonalBoost = monthIndex >= 5 && monthIndex <= 8 ? 7 : monthIndex === 4 || monthIndex === 9 ? 3 : 0.8;
            return Number((seasonalBoost + Math.random() * 4 + index * 0.35).toFixed(2));
        })
    }));
    const demoHeatmap = {
        stations: demoStations,
        months: MONTHS.map((label, index) => ({ value: index + 1, label })),
        cells: demoStations.flatMap((station, stationIndex) =>
            MONTHS.map((label, monthIndex) => ({
                station,
                station_index: stationIndex,
                month: monthIndex + 1,
                month_label: label,
                value: Number(((monthIndex >= 5 && monthIndex <= 8 ? 6 : 1.2) + stationIndex * 0.45 + Math.random() * 3).toFixed(2)),
            }))
        )
    };

    const demoSummary = {
        total_records: 29200,
        stations: demoStations,
        date_range: { start: '2014-01-01', end: '2023-12-31' },
        rainfall_stats: { mean: 5.83, median: 0.0, max: 187.42, std: 14.21 },
        station_avg_rainfall: {
            Mumbai: 8.42, Delhi: 4.32, Chennai: 5.18, Kolkata: 6.14,
            Bangalore: 4.87, Ahmedabad: 5.63, Jaipur: 3.21, Surat: 7.89
        },
        monthly_avg_rainfall: {
            1: 0.82, 2: 0.56, 3: 1.12, 4: 2.34, 5: 4.56, 6: 12.34,
            7: 18.92, 8: 17.45, 9: 11.23, 10: 4.78, 11: 1.89, 12: 0.95
        },
        filters: {
            years: demoYears,
            regions: demoRegions,
            stations: demoStations,
        },
        demoMonthlySeries,
        demoGeoData,
        demoHeatmap,
    };

    analyticsData = demoSummary;
    dashboardFilters = demoSummary.filters;
    updateStatsCards(demoSummary);
    renderOverviewCharts(demoSummary);
    populateStationDropdowns(demoSummary.stations);
    populateDashboardFilters(demoSummary.filters);
    renderMonthlyRainfallChart({ months: Array.from({ length: 12 }, (_, index) => index + 1), series: demoMonthlySeries });
    renderHeatmapChart(demoHeatmap);
    renderDemoTimeSeries();
    renderDemoModelMetrics();
}

// ============================================
// Stats Cards
// ============================================
function updateStatsCards(data) {
    animateNumber('valTotalRecords', data.total_records, 0, true);
    animateNumber('valAvgRainfall', data.rainfall_stats.mean, 2);
    animateNumber('valMaxRainfall', data.rainfall_stats.max, 1);
    document.getElementById('valStations').textContent = data.stations.length;
}

function animateNumber(elementId, target, decimals = 0, isInteger = false) {
    const el = document.getElementById(elementId);
    const duration = 1200;
    const start = 0;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const easeOut = 1 - Math.pow(1 - progress, 3);

        const current = start + (target - start) * easeOut;

        if (isInteger) {
            el.textContent = Math.round(current).toLocaleString();
        } else {
            el.textContent = current.toFixed(decimals);
        }

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// ============================================
// Overview Charts
// ============================================
function renderOverviewCharts(data) {
    renderRainfallDistChart(data);
    renderStationCompChart(data);
}

function renderMonthlyRainfallChart(data) {
    const ctx = document.getElementById('monthlyRainfallChart');
    if (charts.monthly) charts.monthly.destroy();

    const labels = (data.months || Array.from({ length: 12 }, (_, index) => index + 1)).map(month => MONTHS[month - 1] || month);
    const series = data.series || [];

    charts.monthly = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: series.map((stationSeries, index) => {
                const color = STATION_COLORS[stationSeries.station] || getSeriesColor(index);
                return {
                    label: formatStationLabel(stationSeries.station, 26),
                    stationName: stationSeries.station,
                    data: stationSeries.values,
                    borderColor: color,
                    backgroundColor: hexToRgba(color, 0.14),
                    fill: false,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    borderWidth: 2.4,
                    tension: 0.3,
                };
            })
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        usePointStyle: true,
                        padding: 16,
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    borderColor: 'rgba(59, 130, 246, 0.3)',
                    borderWidth: 1,
                    titleFont: { weight: '600' },
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: ctx => `${ctx.dataset.stationName}: ${ctx.parsed.y.toFixed(2)} mm`
                    }
                }
            },
            scales: {
                x: { grid: { display: false } },
                y: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    title: { display: true, text: 'Rainfall (mm)', color: '#64748b' }
                }
            }
        }
    });
}

function renderRainfallDistChart(data) {
    const ctx = document.getElementById('rainfallDistChart');
    if (charts.dist) charts.dist.destroy();

    const stationData = data.station_avg_rainfall;
    const sorted = Object.entries(stationData)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8);
    const stations = sorted.map(([station]) => station);
    const values = sorted.map(([, value]) => value);

    charts.dist = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: stations.map(station => formatStationLabel(station, 20)),
            datasets: [{
                data: values,
                backgroundColor: stations.map((station, index) => STATION_COLORS[station] || getSeriesColor(index)),
                borderColor: 'rgba(10, 14, 26, 0.8)',
                borderWidth: 3,
                hoverOffset: 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '55%',
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        padding: 14,
                        boxWidth: 12,
                        boxHeight: 12,
                        borderRadius: 3,
                        usePointStyle: true,
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    borderColor: 'rgba(59, 130, 246, 0.3)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: ctx => ` ${stations[ctx.dataIndex]}: ${ctx.parsed.toFixed(2)} mm avg`
                    }
                }
            }
        }
    });
}

function renderStationCompChart(data) {
    const ctx = document.getElementById('stationCompChart');
    if (charts.stationComp) charts.stationComp.destroy();

    const stationData = data.station_avg_rainfall;
    const sorted = Object.entries(stationData)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 12);

    charts.stationComp = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sorted.map(s => formatStationLabel(s[0], 18)),
            datasets: [{
                label: 'Avg Rainfall (mm)',
                data: sorted.map(s => s[1]),
                backgroundColor: sorted.map((s, index) => {
                    const color = STATION_COLORS[s[0]] || getSeriesColor(index);
                    return hexToRgba(color, 0.82);
                }),
                borderRadius: 8,
                borderSkipped: false,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    padding: 12,
                    cornerRadius: 8,
                }
            },
            scales: {
                y: { grid: { display: false } },
                x: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    title: { display: true, text: 'Avg Rainfall (mm)', color: '#64748b' }
                }
            }
        }
    });
}

// ============================================
// Time Series Charts
// ============================================
async function loadTimeSeriesData() {
    const station = document.getElementById('tsStation').value;
    const years = document.getElementById('tsYears').value;

    const data = await fetchAPI(`/analytics/timeseries?station=${station}&years=${years}`);
    if (data) {
        renderTimeSeriesChart(data);
        renderCorrelationChart(data);
        renderSeasonalChart(data);
    }
}

function renderTimeSeriesChart(data) {
    const ctx = document.getElementById('timeseriesChart');
    if (charts.timeseries) charts.timeseries.destroy();

    const dates = data.data.map(d => d.date);
    const rainfall = data.data.map(d => d.rainfall_mm);

    // 7-day moving average
    const ma7 = rainfall.map((_, i) => {
        if (i < 6) return null;
        let sum = 0;
        for (let j = i - 6; j <= i; j++) sum += rainfall[j];
        return sum / 7;
    });

    charts.timeseries = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Daily Rainfall (mm)',
                    data: rainfall,
                    borderColor: 'rgba(59, 130, 246, 0.3)',
                    backgroundColor: 'rgba(59, 130, 246, 0.05)',
                    fill: true,
                    pointRadius: 0,
                    borderWidth: 1,
                    tension: 0,
                },
                {
                    label: '7-Day Moving Avg',
                    data: ma7,
                    borderColor: COLORS.cyan,
                    borderWidth: 2.5,
                    pointRadius: 0,
                    tension: 0.4,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: { labels: { padding: 16 } },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    borderColor: 'rgba(59, 130, 246, 0.3)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                }
            },
            scales: {
                x: {
                    type: 'category',
                    ticks: { maxTicksLimit: 12, maxRotation: 45 },
                    grid: { display: false }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    title: { display: true, text: 'Rainfall (mm)', color: '#64748b' }
                }
            }
        }
    });
}

function renderCorrelationChart(data) {
    const ctx = document.getElementById('corrChart');
    if (charts.corr) charts.corr.destroy();

    // Sample data for scatter
    const sampled = data.data.filter((_, i) => i % 3 === 0);

    charts.corr = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Humidity vs Rainfall',
                data: sampled.map(d => ({ x: d.humidity, y: d.rainfall_mm })),
                backgroundColor: 'rgba(59, 130, 246, 0.4)',
                borderColor: 'rgba(59, 130, 246, 0.6)',
                pointRadius: 3,
                pointHoverRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: ctx => `Humidity: ${ctx.parsed.x.toFixed(1)}%, Rainfall: ${ctx.parsed.y.toFixed(2)}mm`
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Humidity (%)', color: '#64748b' },
                    grid: { color: 'rgba(255,255,255,0.04)' }
                },
                y: {
                    title: { display: true, text: 'Rainfall (mm)', color: '#64748b' },
                    grid: { color: 'rgba(255,255,255,0.04)' }
                }
            }
        }
    });
}

function renderSeasonalChart(data) {
    const ctx = document.getElementById('seasonalChart');
    if (charts.seasonal) charts.seasonal.destroy();

    // Aggregate by month
    const monthlyAvg = {};
    const monthlyMax = {};
    data.data.forEach(d => {
        const month = new Date(d.date).getMonth();
        if (!monthlyAvg[month]) { monthlyAvg[month] = []; monthlyMax[month] = 0; }
        monthlyAvg[month].push(d.rainfall_mm);
        monthlyMax[month] = Math.max(monthlyMax[month], d.rainfall_mm);
    });

    const avgValues = MONTHS.map((_, i) => {
        const vals = monthlyAvg[i];
        return vals ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
    });

    const maxValues = MONTHS.map((_, i) => monthlyMax[i] || 0);

    charts.seasonal = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: MONTHS,
            datasets: [
                {
                    label: 'Avg Rainfall',
                    data: avgValues,
                    borderColor: COLORS.blue,
                    backgroundColor: 'rgba(59, 130, 246, 0.15)',
                    borderWidth: 2,
                    pointRadius: 4,
                    pointBackgroundColor: COLORS.blue,
                },
                {
                    label: 'Max Rainfall',
                    data: maxValues,
                    borderColor: COLORS.red,
                    backgroundColor: 'rgba(239, 68, 68, 0.08)',
                    borderWidth: 2,
                    pointRadius: 4,
                    pointBackgroundColor: COLORS.red,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { padding: 16 } },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    padding: 12,
                    cornerRadius: 8,
                }
            },
            scales: {
                r: {
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    angleLines: { color: 'rgba(255,255,255,0.06)' },
                    pointLabels: { font: { size: 11 } },
                    ticks: { display: false }
                }
            }
        }
    });
}

// Demo time series when API is offline
function renderDemoTimeSeries() {
    const dates = [];
    const rainfall = [];
    const humidity = [];

    const startDate = new Date('2021-01-01');
    for (let i = 0; i < 1095; i++) {
        const d = new Date(startDate);
        d.setDate(d.getDate() + i);
        dates.push(d.toISOString().split('T')[0]);

        const month = d.getMonth();
        let rain = 0;
        if (month >= 5 && month <= 8) { // Monsoon
            rain = Math.random() < 0.65 ? Math.random() * 40 + 5 : 0;
        } else if (month === 4 || month === 9) {
            rain = Math.random() < 0.3 ? Math.random() * 15 : 0;
        } else {
            rain = Math.random() < 0.1 ? Math.random() * 8 : 0;
        }
        rainfall.push(Math.round(rain * 100) / 100);
        humidity.push(40 + (month >= 5 && month <= 8 ? 35 : 15) + Math.random() * 15);
    }

    const data = {
        data: dates.map((d, i) => ({
            date: d,
            rainfall_mm: rainfall[i],
            humidity: humidity[i],
            temperature: 25 + 10 * Math.sin(2 * Math.PI * i / 365) + Math.random() * 5
        }))
    };

    renderTimeSeriesChart(data);
    renderCorrelationChart(data);
    renderSeasonalChart(data);
}

// ============================================
// Geospatial Map
// ============================================
async function loadGeospatialData() {
    const year = document.getElementById('geoYear').value;
    const region = document.getElementById('geoRegion').value;
    const topN = document.getElementById('heatmapTopStations').value;

    const geoParams = new URLSearchParams();
    if (year) geoParams.set('year', year);
    if (region && region !== 'all') geoParams.set('region', region);

    const heatmapParams = new URLSearchParams(geoParams);
    heatmapParams.set('top_n', topN);

    const geoEndpoint = geoParams.toString() ? `/analytics/geospatial?${geoParams.toString()}` : '/analytics/geospatial';
    const heatmapEndpoint = `/analytics/station-heatmap?${heatmapParams.toString()}`;
    const data = await fetchAPI(geoEndpoint);
    const heatmapData = await fetchAPI(heatmapEndpoint);

    if (data) {
        initMapWithData(data.data);
    } else if (analyticsData?.demoGeoData) {
        initMapWithData(analyticsData.demoGeoData);
    }

    if (heatmapData) {
        renderHeatmapChart(heatmapData);
    } else if (analyticsData?.demoHeatmap) {
        renderHeatmapChart(analyticsData.demoHeatmap);
    }
}

function initMap() {
    if (map) return;

    map = L.map('rainfallMap', {
        zoomControl: true,
        scrollWheelZoom: true,
    }).setView([20, 0], 2);

    // Dark themed tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(map);

    // Load data for map
    if (analyticsData) {
        loadGeospatialData();
    } else {
        renderDemoMap();
    }
}

function initMapWithData(stationData) {
    if (!map) {
        initMap();
        return;
    }

    // Clear existing markers
    map.eachLayer(layer => {
        if (layer instanceof L.CircleMarker) map.removeLayer(layer);
    });

    const maxRainfall = Math.max(...stationData.map(s => s.avg_rainfall), 1);
    const bounds = [];

    stationData.forEach(station => {
        const radius = 8 + (station.avg_rainfall / maxRainfall) * 25;
        const color = getRainfallColor(station.avg_rainfall, maxRainfall);
        bounds.push([station.latitude, station.longitude]);

        const circle = L.circleMarker([station.latitude, station.longitude], {
            radius: radius,
            fillColor: color,
            color: color,
            weight: 2,
            opacity: 0.9,
            fillOpacity: 0.5,
        }).addTo(map);

        circle.bindPopup(`
            <div style="font-family: Inter, sans-serif; min-width: 200px;">
                <h4 style="margin: 0 0 8px; color: #1e293b; font-size: 14px;">
                    ${formatStationLabel(station.station, 60)} (${station.region})
                </h4>
                <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 8px 0;">
                <table style="width: 100%; font-size: 12px; color: #475569;">
                    <tr><td>Avg Rainfall:</td><td style="text-align:right; font-weight:600;">${station.avg_rainfall} mm</td></tr>
                    <tr><td>Max Rainfall:</td><td style="text-align:right; font-weight:600;">${station.max_rainfall} mm</td></tr>
                    <tr><td>Rainy Days:</td><td style="text-align:right; font-weight:600;">${station.rainy_days_pct}%</td></tr>
                    <tr><td>Avg Temp:</td><td style="text-align:right; font-weight:600;">${station.avg_temperature}°C</td></tr>
                    <tr><td>Avg Humidity:</td><td style="text-align:right; font-weight:600;">${station.avg_humidity}%</td></tr>
                </table>
            </div>
        `, { className: 'custom-popup' });

        circle.bindTooltip(formatStationLabel(station.station, 26), {
            permanent: false,
            direction: 'top',
            offset: [0, -radius],
            className: 'station-tooltip'
        });
    });

    if (bounds.length > 0) {
        map.fitBounds(bounds, { padding: [28, 28] });
    }
}

function renderDemoMap() {
    const demoStations = [
        { station: 'Mumbai', latitude: 19.076, longitude: 72.877, region: 'West', avg_rainfall: 8.42, max_rainfall: 187.42, rainy_days_pct: 38.5, avg_temperature: 27.8, avg_humidity: 72.3 },
        { station: 'Delhi', latitude: 28.614, longitude: 77.209, region: 'North', avg_rainfall: 4.32, max_rainfall: 112.34, rainy_days_pct: 24.1, avg_temperature: 25.4, avg_humidity: 52.6 },
        { station: 'Chennai', latitude: 13.083, longitude: 80.271, region: 'South', avg_rainfall: 5.18, max_rainfall: 145.67, rainy_days_pct: 28.9, avg_temperature: 29.2, avg_humidity: 68.4 },
        { station: 'Kolkata', latitude: 22.573, longitude: 88.364, region: 'East', avg_rainfall: 6.14, max_rainfall: 156.89, rainy_days_pct: 32.7, avg_temperature: 26.9, avg_humidity: 71.2 },
        { station: 'Bangalore', latitude: 12.972, longitude: 77.595, region: 'South', avg_rainfall: 4.87, max_rainfall: 98.45, rainy_days_pct: 27.3, avg_temperature: 24.6, avg_humidity: 64.8 },
        { station: 'Ahmedabad', latitude: 23.023, longitude: 72.571, region: 'West', avg_rainfall: 5.63, max_rainfall: 134.56, rainy_days_pct: 22.8, avg_temperature: 28.5, avg_humidity: 54.7 },
        { station: 'Jaipur', latitude: 26.912, longitude: 75.787, region: 'North', avg_rainfall: 3.21, max_rainfall: 89.23, rainy_days_pct: 18.4, avg_temperature: 26.1, avg_humidity: 45.3 },
        { station: 'Surat', latitude: 21.170, longitude: 72.831, region: 'West', avg_rainfall: 7.89, max_rainfall: 167.34, rainy_days_pct: 35.6, avg_temperature: 28.1, avg_humidity: 68.9 },
    ];

    initMapWithData(demoStations);
    renderHeatmapChart(demoStations);
}

function renderHeatmapChart(stationData) {
    const ctx = document.getElementById('heatmapChart');
    if (charts.heatmap) charts.heatmap.destroy();

    const monthLabels = (stationData.months || []).map(month => month.label);
    const stationLabelMap = Object.fromEntries(
        (stationData.stations || []).map(station => [station, formatStationLabel(station, 24)])
    );
    const yLabels = (stationData.stations || []).map(station => stationLabelMap[station]);
    const maxValue = Math.max(...(stationData.cells || []).map(cell => cell.value), 1);

    charts.heatmap = new Chart(ctx, {
        type: 'matrix',
        data: {
            datasets: [{
                label: 'Avg Rainfall (mm)',
                data: (stationData.cells || []).map(cell => ({
                    x: cell.month_label,
                    y: stationLabelMap[cell.station],
                    v: cell.value,
                    station: cell.station,
                })),
                backgroundColor(context) {
                    const value = context.dataset.data[context.dataIndex]?.v || 0;
                    const ratio = maxValue > 0 ? value / maxValue : 0;

                    if (ratio >= 0.8) return 'rgba(239, 68, 68, 0.9)';
                    if (ratio >= 0.6) return 'rgba(245, 158, 11, 0.85)';
                    if (ratio >= 0.4) return 'rgba(16, 185, 129, 0.82)';
                    if (ratio >= 0.2) return 'rgba(6, 182, 212, 0.78)';
                    return 'rgba(59, 130, 246, 0.68)';
                },
                borderColor: 'rgba(15, 23, 42, 0.85)',
                borderWidth: 1,
                width({ chart }) {
                    const area = chart.chartArea;
                    return area ? Math.max((area.width / Math.max(monthLabels.length, 1)) - 6, 14) : 24;
                },
                height({ chart }) {
                    const area = chart.chartArea;
                    return area ? Math.max((area.height / Math.max(yLabels.length, 1)) - 6, 12) : 18;
                }
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            parsing: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        title(items) {
                            const point = items[0]?.raw;
                            return point ? `${point.station} • ${point.x}` : '';
                        },
                        label(context) {
                            return `Average rainfall: ${context.raw.v.toFixed(2)} mm`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'category',
                    labels: monthLabels,
                    offset: true,
                    grid: { display: false },
                    title: { display: true, text: 'Month', color: '#64748b' }
                },
                y: {
                    type: 'category',
                    labels: yLabels,
                    offset: true,
                    grid: { display: false },
                    ticks: { autoSkip: false },
                    title: { display: true, text: 'Station', color: '#64748b' }
                }
            }
        }
    });
}

// ============================================
// Prediction Form
// ============================================
function initPredictionForm() {
    const form = document.getElementById('predictForm');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const btn = document.getElementById('predictBtn');
        btn.classList.add('loading');
        btn.innerHTML = '<span class="spinner"></span> Predicting...';

        const payload = {
            temperature: parseFloat(document.getElementById('inputTemp').value),
            humidity: parseFloat(document.getElementById('inputHumidity').value),
            pressure: parseFloat(document.getElementById('inputPressure').value),
            wind_speed: parseFloat(document.getElementById('inputWind').value),
            cloud_cover: parseFloat(document.getElementById('inputCloud').value),
            dew_point: parseFloat(document.getElementById('inputDew').value),
        };

        try {
            const response = await fetch(`${API_BASE}/quick-predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (!response.ok) throw new Error('Prediction failed');

            const result = await response.json();
            displayPredictionResult(result);
        } catch (error) {
            // Demo prediction when API is offline
            const demoResult = generateDemoPrediction(payload);
            displayPredictionResult(demoResult);
        }

        btn.classList.remove('loading');
        btn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M22 11.08V12a10 10 0 11-5.93-9.14"/>
                <polyline points="22 4 12 14.01 9 11.01"/>
            </svg>
            Predict Rainfall
        `;
    });
}

function generateDemoPrediction(input) {
    // Simulate prediction based on inputs
    let rainfall = 0;
    if (input.humidity > 70) rainfall += (input.humidity - 70) * 0.3;
    if (input.cloud_cover > 60) rainfall += (input.cloud_cover - 60) * 0.25;
    if (input.pressure < 1010) rainfall += (1010 - input.pressure) * 0.5;
    if (input.wind_speed > 15) rainfall += (input.wind_speed - 15) * 0.15;

    rainfall = Math.max(0, rainfall + Math.random() * 3);
    rainfall = Math.round(rainfall * 100) / 100;

    let intensity, advisory;
    if (rainfall === 0) {
        intensity = 'No Rain';
        advisory = 'Clear conditions expected. No precautions needed.';
    } else if (rainfall < 2.5) {
        intensity = 'Light Rain';
        advisory = 'Light rain expected. Carry an umbrella.';
    } else if (rainfall < 15) {
        intensity = 'Moderate Rain';
        advisory = 'Moderate rainfall predicted. Avoid outdoor activities if possible.';
    } else if (rainfall < 65) {
        intensity = 'Heavy Rain';
        advisory = 'Heavy rainfall alert! Possibility of waterlogging. Stay indoors.';
    } else {
        intensity = 'Very Heavy Rain';
        advisory = 'Very heavy rainfall warning! Risk of flooding. Avoid travel.';
    }

    return {
        rainfall_mm: rainfall,
        intensity,
        advisory,
        confidence: Math.round((0.75 + Math.random() * 0.15) * 100) / 100,
        timestamp: new Date().toISOString(),
    };
}

function displayPredictionResult(result) {
    const container = document.getElementById('predictionResult');
    container.style.display = 'block';
    container.style.animation = 'fadeIn 0.5s ease';

    // Rainfall value with animation
    const rainEl = document.getElementById('resultRainfall');
    animateCountUp(rainEl, result.rainfall_mm, 2);

    // Intensity badge
    const badge = document.getElementById('resultIntensity');
    badge.textContent = result.intensity;
    badge.className = 'result-badge';
    if (result.intensity.includes('Light')) badge.classList.add('light');
    else if (result.intensity.includes('Moderate')) badge.classList.add('moderate');
    else if (result.intensity.includes('Heavy')) badge.classList.add('heavy');
    else if (result.intensity.includes('Extreme')) badge.classList.add('extreme');

    // Confidence
    const confPercent = Math.round(result.confidence * 100);
    document.getElementById('resultConfidence').style.width = confPercent + '%';
    document.getElementById('resultConfidenceText').textContent = confPercent + '%';

    // Advisory
    document.getElementById('advisoryText').textContent = result.advisory;

    // Timestamp
    document.getElementById('resultTimestamp').textContent = new Date(result.timestamp).toLocaleString();
}

function animateCountUp(element, target, decimals) {
    const duration = 800;
    const start = 0;
    const startTime = performance.now();

    function update(time) {
        const elapsed = time - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        element.textContent = (start + (target - start) * eased).toFixed(decimals);
        if (progress < 1) requestAnimationFrame(update);
    }

    requestAnimationFrame(update);
}

// ============================================
// Model Metrics
// ============================================
async function loadModelMetrics() {
    const metrics = await fetchAPI('/model/metrics');
    if (metrics) {
        displayModelMetrics(metrics);
    } else {
        renderDemoModelMetrics();
    }
}

function displayModelMetrics(metrics) {
    if (metrics.mse !== undefined) {
        document.getElementById('metricMSE').textContent = metrics.mse.toFixed(4);
        document.getElementById('metricRMSE').textContent = metrics.rmse.toFixed(4);
        document.getElementById('metricMAE').textContent = metrics.mae.toFixed(4);
        document.getElementById('metricR2').textContent = metrics.r2_score.toFixed(4);
        document.getElementById('metricEV').textContent = metrics.explained_variance.toFixed(4);
        document.getElementById('metricMAPE').textContent = metrics.mape.toFixed(2) + '%';
    } else {
        document.getElementById('metricMSE').textContent = metrics.final_train_loss?.toFixed(4) || '--';
        document.getElementById('metricMAE').textContent = metrics.final_train_mae?.toFixed(4) || '--';
    }
}

function renderDemoModelMetrics() {
    // Demo metrics
    document.getElementById('metricMSE').textContent = '0.0042';
    document.getElementById('metricRMSE').textContent = '0.0648';
    document.getElementById('metricMAE').textContent = '0.0312';
    document.getElementById('metricR2').textContent = '0.8456';
    document.getElementById('metricEV').textContent = '0.8521';
    document.getElementById('metricMAPE').textContent = '12.34%';

    // Demo training history chart
    renderDemoTrainingChart();
}

function renderDemoTrainingChart() {
    const ctx = document.getElementById('trainingHistoryChart');
    if (charts.training) charts.training.destroy();

    const epochs = 50;
    const trainLoss = [];
    const valLoss = [];
    const trainMAE = [];
    const valMAE = [];

    for (let i = 0; i < epochs; i++) {
        const decay = Math.exp(-i * 0.06);
        trainLoss.push(0.15 * decay + 0.003 + Math.random() * 0.002);
        valLoss.push(0.18 * decay + 0.004 + Math.random() * 0.003);
        trainMAE.push(0.12 * decay + 0.025 + Math.random() * 0.003);
        valMAE.push(0.15 * decay + 0.030 + Math.random() * 0.004);
    }

    charts.training = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({ length: epochs }, (_, i) => i + 1),
            datasets: [
                {
                    label: 'Train Loss',
                    data: trainLoss,
                    borderColor: COLORS.blue,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3,
                },
                {
                    label: 'Val Loss',
                    data: valLoss,
                    borderColor: COLORS.cyan,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3,
                    borderDash: [5, 3],
                },
                {
                    label: 'Train MAE',
                    data: trainMAE,
                    borderColor: COLORS.green,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3,
                },
                {
                    label: 'Val MAE',
                    data: valMAE,
                    borderColor: COLORS.amber,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3,
                    borderDash: [5, 3],
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: { labels: { padding: 16 } },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    padding: 12,
                    cornerRadius: 8,
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Epoch', color: '#64748b' },
                    grid: { display: false }
                },
                y: {
                    title: { display: true, text: 'Loss / MAE', color: '#64748b' },
                    grid: { color: 'rgba(255,255,255,0.04)' }
                }
            }
        }
    });
}

// ============================================
// Dropdown Population
// ============================================
function populateStationDropdowns(stations) {
    const selects = ['tsStation'];
    selects.forEach(id => {
        const select = document.getElementById(id);
        select.innerHTML = '';
        stations.forEach(station => {
            const opt = document.createElement('option');
            opt.value = station;
            opt.textContent = station;
            select.appendChild(opt);
        });
    });

    if (stations.length > 0 && !Array.from(document.getElementById('tsStation').options).some(option => option.selected)) {
        document.getElementById('tsStation').value = stations[0];
    }

    initDashboardControls();
}

function populateDashboardFilters(filters) {
    const years = filters?.years || [];
    const regions = filters?.regions || [];

    populateSelect('overviewYear', [{ value: 'all', label: 'All Years' }, ...years.map(year => ({ value: String(year), label: String(year) }))]);
    populateSelect('overviewRegion', [{ value: 'all', label: 'All Regions' }, ...regions.map(region => ({ value: region, label: region }))]);
    populateSelect('geoYear', [{ value: '', label: 'All Years' }, ...years.map(year => ({ value: String(year), label: String(year) }))]);
    populateSelect('geoRegion', [{ value: 'all', label: 'All Regions' }, ...regions.map(region => ({ value: region, label: region }))]);

    initDashboardControls();
}

function populateSelect(id, options) {
    const select = document.getElementById(id);
    if (!select) return;

    const currentValue = select.value;
    select.innerHTML = '';

    options.forEach(({ value, label }) => {
        const opt = document.createElement('option');
        opt.value = value;
        opt.textContent = label;
        select.appendChild(opt);
    });

    const hasCurrentValue = options.some(option => option.value === currentValue);
    select.value = hasCurrentValue ? currentValue : options[0]?.value || '';
}

function initDashboardControls() {
    if (controlsInitialized) return;
    controlsInitialized = true;

    document.getElementById('tsStation').addEventListener('change', loadTimeSeriesData);
    document.getElementById('tsYears').addEventListener('change', loadTimeSeriesData);
    document.getElementById('overviewYear').addEventListener('change', loadOverviewData);
    document.getElementById('overviewRegion').addEventListener('change', loadOverviewData);
    document.getElementById('overviewTopStations').addEventListener('change', loadOverviewData);
    document.getElementById('geoYear').addEventListener('change', loadGeospatialData);
    document.getElementById('geoRegion').addEventListener('change', loadGeospatialData);
    document.getElementById('heatmapTopStations').addEventListener('change', loadGeospatialData);
}

async function loadOverviewData() {
    const year = document.getElementById('overviewYear').value;
    const region = document.getElementById('overviewRegion').value;
    const topN = document.getElementById('overviewTopStations').value;

    const params = new URLSearchParams();
    if (year !== 'all') params.set('year', year);
    if (region !== 'all') params.set('region', region);
    params.set('top_n', topN);

    const endpoint = `/analytics/monthly-stations?${params.toString()}`;
    const data = await fetchAPI(endpoint);

    if (data) {
        renderMonthlyRainfallChart(data);
    } else if (analyticsData?.demoMonthlySeries) {
        renderMonthlyRainfallChart({ months: Array.from({ length: 12 }, (_, index) => index + 1), series: analyticsData.demoMonthlySeries });
    }

    // Time series station change
    document.getElementById('tsStation').addEventListener('change', loadTimeSeriesData);
    document.getElementById('tsYears').addEventListener('change', loadTimeSeriesData);
}
