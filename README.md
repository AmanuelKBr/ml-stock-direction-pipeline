# ML Stock Direction Pipeline

Machine learning pipeline for **predicting short-term stock market direction** using technical indicators and automated retraining.
The system continuously fetches fresh market data, generates features, trains a model, and serves predictions through a **FastAPI service deployed with Docker**.

This project demonstrates an **end-to-end applied machine learning system**, including data ingestion, feature engineering, model training, API serving, retraining triggers, and containerized deployment.

---

# Project Overview

This project predicts whether the **next market move for SPY (S&P 500 ETF)** will be **UP or DOWN** based on technical indicators derived from historical price data.

The system includes:

* Automated data ingestion from Yahoo Finance
* Feature engineering with financial indicators
* Machine learning model training
* REST API prediction service
* On-demand retraining endpoint
* Docker containerization
* Docker Compose orchestration

The goal of the project is to demonstrate a **production-style machine learning workflow**, not simply a notebook model.

---

# System Architecture

User / Client
↓
FastAPI Service
↓
Prediction Endpoint (`/predict`)
↓
Feature Generation
↓
Trained Model (`Random Forest`)
↓
Prediction Response

Optional admin trigger:

`POST /retrain`

↓

Retraining Pipeline

* Download latest market data
* Generate features
* Train model
* Save updated model

---

# Project Structure

```
ml-stock-direction-pipeline
│
├── src
│   ├── api.py
│   ├── retrain_pipeline.py
│   ├── train_model.py
│   ├── feature_engineering.py
│   ├── data_ingestion.py
│   └── predict.py
│
├── models
│   └── rf_model.pkl
│
├── data
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── Makefile
├── .gitignore
└── README.md
```

---

# Machine Learning Pipeline

## Data Ingestion

Historical SPY market data is downloaded using the Yahoo Finance API.

```
yfinance
```

The pipeline retrieves:

* Open
* High
* Low
* Close
* Volume

---

## Feature Engineering

Technical indicators are generated from price data.

Features include:

* Daily returns
* Moving averages (10, 50)
* RSI
* Volatility
* Lagged returns
* Momentum
* MACD

These features provide signals related to **trend, momentum, and volatility**.

---

## Model Training

A **Random Forest classifier** is used to predict the next market direction.

Target variable:

```
1 → Market Up
0 → Market Down
```

The dataset is split using **time-series ordering** to prevent look-ahead bias.

---

# API Endpoints

The model is served using **FastAPI**.

## Root Endpoint

```
GET /
```

Returns API status.

Example response:

```
{
 "message": "Stock Prediction API is running"
}
```

---

## Prediction Endpoint

```
GET /predict
```

Returns the predicted market direction and probability.

Example response:

```
{
 "prediction": "UP",
 "probability_up": 0.5723
}
```

---

## Retraining Endpoint

```
POST /retrain
```

Triggers the full retraining pipeline.

Steps executed:

1. Download latest market data
2. Rebuild feature dataset
3. Train model
4. Save new model artifact
5. Reload model into the API

Example response:

```
{
 "status": "Model retrained successfully"
}
```

---

# Running the Project Locally

## Clone Repository

```
git clone https://github.com/yourusername/ml-stock-direction-pipeline.git
cd ml-stock-direction-pipeline
```

---

## Run with Docker

Build and start the system:

```
docker compose up --build
```

API will be available at:

```
http://localhost:8000
```

Interactive API documentation:

```
http://localhost:8000/docs
```

---

# Containerized Deployment

The entire system runs inside a Docker container.

Key components:

* Python runtime
* FastAPI server
* Machine learning model
* Feature pipeline

Docker ensures the project runs consistently across environments.

---

# Example Workflow

Start the system

```
docker compose up
```

Request a prediction

```
GET /predict
```

Trigger retraining

```
POST /retrain
```

Receive updated predictions using the newly trained model.

---

# Technologies Used

* Python
* FastAPI
* Scikit-Learn
* Pandas
* NumPy
* TA (technical indicators)
* Yahoo Finance API
* Docker
* Docker Compose

---

# Future Improvements

Possible enhancements include:

* Model drift monitoring
* Scheduled retraining jobs
* Feature store integration
* Experiment tracking
* Cloud deployment
* Streamlit visualization dashboard

---

# Author

Amanuel Birri

Data & Analytics Engineer | Applied Machine Learning
Focused on building **production-ready ML systems and analytics pipelines**.

LinkedIn and portfolio links coming soon.
