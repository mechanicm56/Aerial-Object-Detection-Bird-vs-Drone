# Aerial Object Classification & Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bird-vs-drone.streamlit.app/)

Deep Learning–based Bird vs Drone classification and optional YOLOv11 object detection.

## 📌 Overview
This project aims to identify aerial objects (Bird or Drone) using image classification and real-time detection. It uses a custom CNN, transfer learning models, and optionally YOLOv11. A Streamlit app is included for deployment.

## 🚀 Features
- Image classification (Bird vs Drone)
- YOLOv11 object detection (optional)
- Data preprocessing & augmentation
- Streamlit UI for deployment

## 🧠 Skills Learned
- Deep Learning  
- Computer Vision  
- Image Classification & Detection  
- TensorFlow/Keras or PyTorch  
- YOLOv11  
- Streamlit  

## 📂 Project Structure
project/
├── data/
├── dataset/
├── models/
├── notebooks/
├── app.py
├── README.md
└── requirements.txt

---

## 🧩 Methodology
### 1. **Data Preparation**
- Image cleaning  
- Augmentation (flip, rotation, brightness, zoom)

### 2. **Modeling**
- Custom CNN  
- Transfer learning (MobileNet, ResNet, EfficientNet)  
- YOLOv11 detection (optional)

### 3. **Evaluation**
- Accuracy, F1-score  
- Confusion Matrix  
- Sample prediction visualization  

### 4. **Deployment**
Streamlit dashboard for easy use.

---

## 🌟 What This Project Does
- Learns to classify images as Bird or Drone 
- Detects objects in videos/images using YOLOv11
- Lets you upload images using a friendly Streamlit interface  

---

## 📦 Requirements
Install everything using:

```bash
git clone <repo-url>
pip install -r requirements.txt
```