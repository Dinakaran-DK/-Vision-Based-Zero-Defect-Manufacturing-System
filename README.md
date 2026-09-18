# -Vision-Based-Zero-Defect-Manufacturing-System
AI-powered industrial visual inspection system for defect detection, classification, localization, severity estimation, and probable-cause analysis using product images and production data
# 🔍 Intelligent Visual Inspection System

An AI-powered industrial inspection system that detects manufacturing defects and connects them with production conditions to identify probable contributing factors.

## 🚀 Features

* Defect Detection
* Defect Classification
* Defect Localization
* Severity Estimation
* Probable-Cause Analysis
* Production Feedback

## 🧠 AI Models

* **YOLO** – Defect detection & localization
* **Segmentation** – Precise defect region
* **XGBoost** – Production condition analysis
* **SHAP** – Explainable AI & contributing factors

## 🛠️ Tech Stack

**Python | PyTorch | OpenCV | FastAPI | PostgreSQL | Streamlit | YOLO | XGBoost**

## 🔄 Workflow

```text
Product Image
     ↓
Defect Detection
     ↓
Localization & Severity
     ↓
Production Data Analysis
     ↓
Probable Cause
     ↓
Production Feedback

AI Architecture
                  PRODUCT IMAGE
                       │
                       ▼
                ┌─────────────┐
                │    YOLO     │
                │   Detection │
                └──────┬──────┘
                       │
              ┌────────┴─────────┐
              ▼                  ▼
       Defect Classification   Location
                                  │
                                  ▼
                             Segmentation
                                  │
                                  ▼
                           Defect Area/Size
                                  │
                                  ▼
                          Severity Estimation
                                  │
                                  │
Machine Parameters ────────────────┤
Batch Information ────────────────┤
Operator Shift ──────────────────┤
Environmental Data ──────────────┤
                                  ▼
                              XGBoost
                                  │
                                  ▼
                               SHAP
                                  │
                                  ▼
                     Probable Contributing Factors
                                  │
                                  ▼
                         Production Feedback
                                  │
                                  ▼
                            Dashboard
```

## 🎯 Goal

To transform traditional visual inspection into an intelligent quality-control system by connecting **what the product looks like with what was happening during production**.
