# End-to-End Machine Learning Project: Sentiment Analysis with Transformers, MLflow & Docker

This repository showcases a complete, real-world machine learning project for **sentiment analysis** of movie reviews. It demonstrates a full ML pipeline from data preprocessing to model deployment, leveraging industry-standard tools like **PyTorch**, **Hugging Face Transformers**, **MLflow**, and **Docker**.

The goal of this project is to provide a robust, production-ready solution that can be run with a single command — ideal for demonstrating proficiency in **modern MLOps practices** for job interviews and building a strong data science portfolio.

---

## 🌟 Features

- **Real-world Dataset**: Utilizes the IMDB movie review dataset for binary sentiment classification (positive/negative).
- **Hugging Face Transformers**: Fine-tunes a pre-trained DistilBERT model using PyTorch.
- **Full ML Pipeline**: Covers data loading, preprocessing, model training, evaluation, inference, and API serving.

### 🧪 Experiment Tracking with MLflow:
- Logs all training parameters, metrics (loss, accuracy, precision, recall, F1-score), and artifacts.
- Enables easy comparison of different experiment runs via the MLflow UI.
- Automatically saves the best performing model.

### 📦 Model Registry with MLflow:
- Registers trained models for version control, staging, and production management.
- Facilitates easy loading of specific model versions for inference.

### 🐳 Containerization with Docker:
- Provides a Dockerfile to create a reproducible environment for the entire application.
- Uses `docker-compose.yml` to orchestrate multi-container services, including:
  - The ML application (for training/evaluation)
  - An MLflow Tracking Server (for persistence and UI)
  - A FastAPI web service for real-time model inference

### 🌐 REST API for Inference: (to be implemented)
- A lightweight FastAPI endpoint to serve predictions on new text data. 

### 💻 Web UI: (to be implemented)
- A simple HTML/JavaScript frontend to interact with the FastAPI for sentiment analysis.

### 🧼 Clean Code Organization:
- Structured into modular Python scripts for readability and maintainability.

---

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### 🔧 Prerequisites

- **Docker Desktop** (for Windows/macOS) or **Docker Engine & Compose** (for Linux). Ensure Docker is running.

## 📦 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ml_sentiment_project.git
cd ml_sentiment_project


ml_sentiment_project/
├── data/                    # Persistent data storage
├── mlruns/                  # MLflow tracking data (runs, metrics, models) will be generated 
├── src/
│   ├── data_processing.py   # Tokenization, Dataloader
│   ├── model_training.py    # Training loop, logging
│   ├── model_evaluation.py  # Accuracy, precision, recall, F1
│   ├── inference.py         # Prediction utilities
│   ├── utils.py             # Utilities
│   └── __init__.py
├── api.py                   # FastAPI app
├── main.py                  # Training script
├── requirements.txt         # Python dependencies
├── Dockerfile               # Image for ml_app and api
├── docker-compose.yml       # Multi-container orchestration
└── README.md


## 🛠️ Technologies Used

- **Python 3.9+**
- **PyTorch**
- **Hugging Face Transformers & Datasets**
- **scikit-learn**
- **MLflow**
- **FastAPI + Uvicorn**
- **Docker + Docker Compose**

