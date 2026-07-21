# 🌧️ Smart Rainfall Prediction & Decision Support System

An AI-powered rainfall prediction system using deep learning models trained on historical weather data, with full MLOps architecture for reproducible training, experiment tracking, automated deployment, and continuous integration.

**MTech CSE | Semester II | Lab Practice II**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [MLOps Pipeline](#mlops-pipeline)
- [Dashboard](#dashboard)
- [API Documentation](#api-documentation)
- [Technologies Used](#technologies-used)

---

## 🎯 Overview

### Problem Statement
Develop a rainfall prediction system using machine learning/deep learning models trained on historical weather data. Perform temporal and spatial data analysis to visualize rainfall trends. Implement a full MLOps architecture to enable reproducible training, experiment tracking, automated deployment, and continuous integration.

### Core Components
| Component | Description |
|-----------|-------------|
| **Deep Learning** | Bidirectional LSTM model for rainfall prediction |
| **Data Visualization** | Time-series and geospatial rainfall dashboards |
| **MLOps** | Git + DVC, MLflow, FastAPI, Docker, CI/CD |

### Deliverables
- ✅ Trained rainfall prediction model (LSTM/GRU/CNN-LSTM)
- ✅ Cloud-based analytics dashboard with interactive visualizations
- ✅ Fully automated MLOps pipeline with CI/CD

---

## 📦 Dataset

| Attribute | Details |
|-----------|----------|
| **Source** | [Indian Weather Repository (Daily Snapshot)](https://www.kaggle.com/datasets/nelgiriyewithana/indian-weather-repository-daily-snapshot) — Kaggle |
| **Author** | Nidula Elgiriyewithana |
| **License** | CC0 1.0 Universal (Public Domain) |
| **Coverage** | 8 major Indian cities (filtered from full dataset) |
| **Period** | 2014-01-01 to 2023-12-31 (10 years, daily observations) |
| **Records** | ~29,200 daily rows (3,650 per city) |
| **Features** | `temp_c`, `humidity`, `pressure_mb`, `wind_kph`, `cloud`, `dewpoint_c`, `precip_mm` |

### Cities / Stations Used
| Station | Latitude | Longitude | Region |
|---------|----------|-----------|--------|
| Mumbai | 19.0760 | 72.8777 | West |
| Delhi | 28.6139 | 77.2090 | North |
| Chennai | 13.0827 | 80.2707 | South |
| Kolkata | 22.5726 | 88.3639 | East |
| Bangalore | 12.9716 | 77.5946 | South |
| Ahmedabad | 23.0225 | 72.5714 | West |
| Jaipur | 26.9124 | 75.7873 | North |
| Surat | 21.1702 | 72.8311 | West |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CI/CD Pipeline                        │
│                (GitHub Actions)                          │
├──────────┬──────────┬──────────────┬────────────────────┤
│   Lint   │   Test   │  DVC Train   │  Docker Build      │
│  flake8  │  pytest  │  dvc repro   │  Push to Registry  │
└──────────┴──────────┴──────────────┴────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│  Data Layer  │  │ Model Layer  │  │  Serving Layer   │
│              │  │              │  │                  │
│ • Synthetic  │  │ • Bi-LSTM    │  │ • FastAPI REST   │
│   Weather    │──│ • GRU        │──│ • Docker         │
│   Data       │  │ • CNN-LSTM   │  │ • Health Checks  │
│ • 8 Stations │  │ • MLflow     │  │ • Prediction API │
│ • 10 Years   │  │   Tracking   │  │                  │
└──────────────┘  └──────────────┘  └──────────────────┘
        │                                    │
        ▼                                    ▼
┌──────────────────────────────────────────────────────┐
│              Analytics Dashboard                      │
│                                                      │
│  📊 Time Series  │  🗺️ Geospatial  │  🤖 Predictions │
│  📈 Statistics   │  📉 Correlations │  📋 Metrics     │
└──────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
Smart-Rainfall-Prediction/
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # GitHub Actions CI/CD pipeline
├── api/
│   └── app.py                 # FastAPI REST API
├── configs/
│   └── config.yaml            # Central configuration
├── dashboard/
│   ├── index.html             # Analytics dashboard
│   ├── styles.css             # Premium dark theme CSS
│   └── app.js                 # Dashboard logic & charts
├── data/
│   ├── raw/                   # Raw rainfall data
│   └── processed/             # Processed features & sequences
├── models/                    # Trained models & artifacts
├── src/
│   ├── __init__.py
│   ├── data_preparation.py    # Data generation & preprocessing
│   ├── model.py               # DL model architectures
│   ├── train.py               # Training with MLflow tracking
│   ├── evaluate.py            # Model evaluation
│   └── predict.py             # Inference module
├── tests/
│   ├── test_model.py          # Model unit tests
│   └── test_api.py            # API integration tests
├── Dockerfile                 # Docker containerization
├── docker-compose.yml         # Multi-service orchestration
├── dvc.yaml                   # DVC pipeline definition
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- Git
- Docker (optional, for containerized deployment)

### 1. Clone & Install

```bash
# Clone the repository
git clone <repository-url>
cd Smart-Rainfall-Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize DVC & Git

```bash
git init
dvc init
```

---

## 🚀 Usage

### Step 1: Prepare Data
```bash
python src/data_preparation.py
```
Loads and preprocesses the [Indian Weather Repository (Daily Snapshot)](https://www.kaggle.com/datasets/nelgiriyewithana/indian-weather-repository-daily-snapshot) dataset from Kaggle (by Nidula Elgiriyewithana, CC0 license), filtering to 8 Indian cities over 10 years (2014–2023).

### Step 2: Train Model
```bash
python src/train.py
```
Trains the Bidirectional LSTM model with MLflow experiment tracking.

### Step 3: Evaluate Model
```bash
python src/evaluate.py
```
Generates evaluation metrics (MSE, RMSE, MAE, R², MAPE).

### Step 4: Start API Server
```bash
python api/app.py
# Or: uvicorn api.app:app --reload --port 8000
```
API docs available at: http://127.0.0.1:8000/docs

### Step 5: Open Dashboard
Open `dashboard/index.html` in your browser. The dashboard connects to the API automatically.

### Run with DVC Pipeline (Reproducible)
```bash
dvc repro
```

### Run with Docker
```bash
docker-compose up -d
```

---

## 🔄 MLOps Pipeline

### DVC Pipeline Stages
```
prepare_data ──→ train ──→ evaluate
     │              │          │
  raw data     model.h5    metrics.json
  sequences    MLflow run   plots
  scalers      history
```

### MLflow Experiment Tracking
- **Parameters**: model type, LSTM units, dropout, learning rate, epochs, batch size
- **Metrics**: MSE, MAE, R², training/validation loss per epoch
- **Artifacts**: model weights, training plots, scalers, configuration

### CI/CD Pipeline (GitHub Actions)
1. **Lint & Test**: flake8 linting + pytest unit tests
2. **Train**: DVC pipeline reproduction
3. **Docker Build**: Build and push Docker image
4. **Deploy**: Production deployment

### Production Deployment Secrets
Set these GitHub repository secrets before running the production deployment job:

- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`
- `PROD_HOST`
- `PROD_USER`
- `PROD_SSH_KEY`
- `PROD_SSH_PORT` (optional, defaults to `22`)

---

## 📊 Dashboard

The analytics dashboard provides five interactive panels:

| Panel | Features |
|-------|----------|
| **Overview** | Monthly rainfall charts, station comparison, distribution |
| **Time Series** | Rainfall trends, 7-day moving average, correlations |
| **Geospatial** | Interactive India map with station markers and heatmap |
| **Predict** | Real-time rainfall prediction with weather input |
| **Model Metrics** | Training history, evaluation metrics, architecture |

### Features
- 🌑 Premium dark theme with glassmorphism
- 📱 Fully responsive design
- 🎨 Interactive Chart.js visualizations
- 🗺️ Leaflet.js geospatial mapping
- ⚡ Real-time API integration
- 🎭 Smooth animations and transitions

---

## 🔌 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info & health |
| GET | `/health` | Detailed health check |
| POST | `/predict` | Predict from 30-day sequence |
| POST | `/quick-predict` | Quick predict from current conditions |
| GET | `/analytics/summary` | Dataset summary statistics |
| GET | `/analytics/station/{name}` | Station-specific analytics |
| GET | `/analytics/timeseries` | Time-series data for charts |
| GET | `/analytics/geospatial` | Geospatial station data |
| GET | `/model/metrics` | Model evaluation metrics |

### Example: Quick Prediction
```bash
curl -X POST http://127.0.0.1:8000/quick-predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 28.5,
    "humidity": 78.0,
    "pressure": 1006.5,
    "wind_speed": 18.2,
    "cloud_cover": 72.0,
    "dew_point": 24.1
  }'
```

Response:
```json
{
  "rainfall_mm": 12.45,
  "intensity": "Moderate Rain",
  "advisory": "Moderate rainfall predicted. Avoid outdoor activities if possible.",
  "confidence": 0.82,
  "timestamp": "2024-07-15T10:30:00"
}
```

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | TensorFlow/Keras, LSTM, GRU, CNN |
| **Data Science** | NumPy, Pandas, Scikit-learn |
| **Visualization** | Chart.js, Leaflet.js, Matplotlib, Seaborn |
| **MLOps** | MLflow, DVC, Docker, GitHub Actions |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Testing** | Pytest, FastAPI TestClient |
| **Frontend** | HTML5, CSS3 (Glassmorphism), JavaScript |

---

## 📝 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 👨‍🎓 Author

**MTech CSE - Semester II**  
Lab Practice II | 2025-2026

---
