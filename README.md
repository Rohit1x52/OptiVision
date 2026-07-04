# OptiVision: End-to-End Computer Vision Optimization & Edge Deployment Platform

## Overview

OptiVision is an end-to-end Machine Learning and Deep Learning platform that automates the complete computer vision workflow—from dataset preprocessing and model training to hyperparameter optimization, model compression, benchmarking, and edge deployment.

The platform is designed to demonstrate production-ready ML engineering practices by integrating experiment tracking, model optimization, and deployment into a single pipeline.

---

## Key Features

* Upload custom image datasets
* Automatic data preprocessing and augmentation
* Train multiple CNN architectures
* Hyperparameter optimization using Optuna
* Experiment tracking with MLflow
* Model comparison and best model selection
* Model compression using Pruning and Quantization
* Performance benchmarking (Accuracy, Latency, FPS, Model Size)
* Export optimized models (ONNX, TorchScript)
* Deploy models using FastAPI
* Edge AI-ready optimized models

---

## Tech Stack

### Machine Learning & Deep Learning

* PyTorch
* TorchVision
* Scikit-learn
* OpenCV
* NumPy
* Pandas

### Hyperparameter Optimization

* Optuna

### Experiment Tracking

* MLflow

### Model Compression

* PyTorch Pruning API
* PyTorch Quantization

### Edge Deployment

* ONNX
* ONNX Runtime
* TorchScript

### Backend

* FastAPI

### Frontend

* Streamlit (Phase 1)
* React (Future Enhancement)

### Deployment

* Docker

---

## Project Workflow

```text
Dataset Upload
      │
      ▼
Data Validation
      │
      ▼
Preprocessing & Augmentation
      │
      ▼
Train Multiple CNN Models
      │
      ▼
Hyperparameter Optimization (Optuna)
      │
      ▼
MLflow Experiment Tracking
      │
      ▼
Best Model Selection
      │
      ▼
Model Compression
 ├── Pruning
 └── Quantization
      │
      ▼
Performance Benchmarking
      │
      ▼
Export (ONNX / TorchScript)
      │
      ▼
FastAPI Deployment
      │
      ▼
Edge Device Inference
```

---

## Project Structure

```text
OptiVision/
│
├── datasets/
│
├── models/
│     ├── mobilenet.py
│     ├── efficientnet.py
│     ├── resnet.py
│     └── densenet.py
│
├── preprocessing/
│     ├── augmentation.py
│     ├── preprocessing.py
│     └── dataset.py
│
├── training/
│     ├── trainer.py
│     ├── optimizer.py
│     ├── scheduler.py
│     └── train.py
│
├── tuning/
│     └── optuna_tuner.py
│
├── tracking/
│     └── mlflow_logger.py
│
├── compression/
│     ├── pruning.py
│     ├── quantization.py
│     ├── benchmark.py
│     └── compare.py
│
├── deployment/
│     ├── export_onnx.py
│     ├── export_torchscript.py
│     ├── fastapi_server.py
│     └── inference.py
│
├── dashboard/
│
├── utils/
│
├── app.py
│
├── requirements.txt
│
└── README.md
```

---

## Development Roadmap

### Phase 1 – Core Model Training

* Dataset upload
* Image preprocessing
* Data augmentation
* Train multiple CNN models
* Model evaluation

### Phase 2 – Hyperparameter Optimization

* Optuna integration
* MLflow experiment tracking
* Automatic best model selection

### Phase 3 – Model Optimization

* Structured & Unstructured Pruning
* Dynamic & Static Quantization
* Accuracy vs Speed comparison

### Phase 4 – Edge Deployment

* ONNX export
* TorchScript export
* FastAPI inference API
* Docker containerization

### Phase 5 – Dashboard & Benchmarking

* Interactive dashboard
* Model comparison charts
* Benchmark reports
* Download optimized models
* Edge deployment metrics

---

## Future Enhancements

* TensorFlow Lite (TFLite) export
* Raspberry Pi deployment
* NVIDIA Jetson deployment
* Knowledge Distillation module
* Automated Neural Architecture Search (NAS)
* Grad-CAM visualization
* Explainable AI dashboard
* CI/CD pipeline with GitHub Actions
* Cloud deployment on AWS/Azure/GCP

```
```
