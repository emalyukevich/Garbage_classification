# 🗑️ Garbage Classification API

> **Automatic image classification of waste using EfficientNet-B0, deployed as a REST API with FastAPI.**

---

## 📌 Project Overview

Garbage Classification API is a complete pet project focused on solving the problem of sustainable waste sorting using computer vision. A user can upload an image, and the model will classify it into one of the predefined categories (e.g., `plastic`, `glass`, `metal`, `paper`, etc.).

The project demonstrates the **entire lifecycle of an ML application** — from data preparation and model training to API deployment and prediction visualization.

---

## 🎯 Goals and Applications

- 💡 Learn and showcase skills in **Data Science + MLOps**
- ♻️ Potential use in **environmental projects** and **waste sorting systems**
- 📱 Can be integrated into mobile or IoT solutions for real-time garbage recognition
- 🧪 Great portfolio project for a Junior/Middle Data Scientist or ML Engineer

---

## 🧠 Demonstrated Skills

| Area                 | Skills                                                                      |
|----------------------|-----------------------------------------------------------------------------|
| **Data Science**     | Data preparation, EDA, augmentation, training EfficientNet-B0               |
| **Machine Learning** | Transfer Learning, Fine-tuning, EarlyStopping, performance metrics          |
| **Deployment**       | REST API development (FastAPI), Docker containerization, Uvicorn            |
| **Software Dev**     | Working with `requirements.txt`, clean project structure, error handling    |
| **Engineering**      | File processing, prediction overlay, logging, debugging                     |

---

## ⚙️ Technologies Used

- **Python 3.10**
- **PyTorch** — for model training and inference
- **EfficientNet-B0** — a CNN architecture with an excellent speed/accuracy tradeoff
- **Albumentations** — a powerful library for image augmentation
- **FastAPI** — a modern web framework for building REST APIs
- **Uvicorn** — an ASGI server for running FastAPI applications
- **Docker** — for packaging and deploying the project in a container
- **Pillow (PIL)** — for image handling and drawing predictions
- **requests** — for testing and interacting with the API

---

## 🖼️ Prediction Visualization

The project includes an interface at `/view_examples` that displays validation images with model predictions overlaid. An automated script `generate_examples.py` is also included, which:

1. Randomly selects images from `data/val/`
2. Sends them to the API
3. Overlays the predicted label on the image
4. Saves the result to the `examples/` folder

---

## 📦 Project Structure

```bash
Garbage_classification/
├── app/                   # FastAPI application
│   ├── main.py            # FastAPI entry point
│   ├── model.py           # Model loading and inference
│   ├── utils.py           # Utility functions
│   ├── config.py          # Paths and configuration
│   ├──	examples/          # Images with predictions
│   └── generate_examples.py # Script for generating predictions on images
│
├── data/                  # Dataset
├── models/                # Trained model files (.pth)
├── notebooks/             # Data analysis and experiments
├── src/                   # Model training source code
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker instructions
└── README.md              # Project documentation

```

---

## 🛠️ API Endpoints

| Method   | Endpoint         | Description                              |
|----------|------------------|------------------------------------------|
| `POST`   | `/predict`       | Image classification                     |
| `GET`    | `/view_examples` | View predictions on sample images        |

---

## 🚀 Getting Started

### 🔧 1. Install dependencies and run locally

```bash
# Clone the repository
git clone https://github.com/emalyukevich/Garbage_classification.git
cd garbage_classification

# (Optional but recommended) Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run FastAPI app
uvicorn app.main:app --host 0.0.0.0 --port 8000

Once started, open your browser at: http://127.0.0.1:8000/docs
The Swagger UI will be available for testing the API.

### 🐳 2. Run using Docker

```bash
# Build the container
docker build -t garbage-api .

# Run the container
docker run -d -p 8000:8000 --name garbage_container garbage-api

# Check if the container is running
docker ps

### 🧹 3. Clean up Docker (if needed)

```bash
# Stop and remove the container
docker stop garbage_container
docker rm garbage_container

# Remove the image
docker rmi garbage-api
