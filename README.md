# 🩺 AI-POWERED MEDICAL IMAGE ANALYSIS SYSTEM

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)]  
[![DeepLearning](https://img.shields.io/badge/DeepLearning-MobileNetV2-orange?style=flat-square)]  
[![Status](https://img.shields.io/badge/Status-Completed-green?style=flat-square)]

---

## ⚡ SYSTEM OVERVIEW

This project is an AI-powered medical image analysis system built using deep learning. It analyzes chest X-ray images and classifies them into two categories: **NORMAL** and **PNEUMONIA**. The system simulates real-world diagnostic assistance used in hospitals and radiology centers.

---

## 🔥 PROBLEM STATEMENT

Manual diagnosis of medical images is time-consuming and prone to human error. Radiologists often deal with large volumes of scans, which can lead to delayed or incorrect diagnosis. This project aims to automate disease detection using AI to assist healthcare professionals.

---

## 🏥 INDUSTRY RELEVANCE

AI-based medical imaging systems are widely used by healthcare companies like Google Health, IBM Watson Health, and Siemens Healthineers.

These systems:
- Assist doctors in faster diagnosis  
- Reduce human error  
- Improve early disease detection  
- Scale healthcare services efficiently  

---

## 🏗️ SYSTEM ARCHITECTURE

Chest X-ray Dataset → Image Loading → Preprocessing (Resize, Normalize, Augmentation) → Feature Extraction (MobileNetV2) → Model Training → Prediction Engine → Evaluation Metrics → Confusion Matrix → Visualization Output

---

## ⚙️ INSTALLATION & SETUP

git clone https://github.com/maheshbhakre/-AI-Powered-Medical-Image-Analysis.git  
cd -AI-Powered-Medical-Image-Analysis  

python -m venv venv  

venv\Scripts\activate   (Windows)  
source venv/bin/activate   (Linux/Mac)  

pip install -r requirements.txt  

---

## 💻 USAGE

python src/train.py  
python src/predict.py  
python src/evaluate.py  

### Dashboard

After training, start the Streamlit dashboard:

```bash
python -m streamlit run dashboard.py
```

Open `http://localhost:8501`. The dashboard uses the saved model at
`models/medical_model.h5`, preserves the training preprocessing (224x224 RGB input
scaled to 0-1), and reports model predictions as educational screening output only.

---

## ⚙️ TECH STACK

Python, TensorFlow/Keras, OpenCV, NumPy, Matplotlib, Seaborn, MobileNetV2 (Transfer Learning)

---

## 📊 DATASET

Chest X-Ray Pneumonia Dataset  

Classes:
- NORMAL  
- PNEUMONIA  

The dataset simulates real-world radiology data used for disease detection.

---

## 📈 RESULTS

Accuracy: ~89%  

- High recall for pneumonia detection (96%)  
- Model effectively detects disease cases  
- Some false positives (normal classified as pneumonia)  

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

### Accuracy Graph

![Accuracy](images/accuracy.png)

---

## 📁 PROJECT STRUCTURE

AI-MEDICAL-IMAGE-ANALYSIS/  
├── data/  
│   └── chest_xray/  
│       ├── test/  
│       ├── train/  
│       └── val/  
├── docs/  
├── images/  
│   ├── accuracy.png  
│   ├── confusion_matrix.png  
│   ├── evaluate.png  
│   ├── predicted.png  
│   ├── sample_xray.png  
│   └── training.png  
├── models/  
│   └── medical_model.h5  
├── notebook/  
├── outputs/  
│   ├── accuracy.png  
│   ├── confusion_matrix.png  
│   ├── graphs  
│   └── predictions  
├── src/  
│   ├── check_data.py  
│   ├── evaluate.py  
│   ├── model.py  
│   ├── predict.py  
│   ├── preprocessing.py  
│   ├── train.py  
│   └── utils.py  
├── main.py  
├── requirements.txt  
└── README.md  

---

## 📸 OUTPUT SCREENSHOTS

<p align="center">
<img src="images/x-ray_prediction.png" width="700"/>
</p>

<p align="center">
<img src="images/model_prediction.png" width="700"/>
</p>

<p align="center">
<img src="images/overview.png" width="700"/>
</p>

<p align="center">
<img src="images/evaluation.png" width="700"/>
</p>

<p align="center">
<img src="images/graph.png" width="700"/>
</p>

---

## 🚀 WHAT I LEARNED

End-to-end deep learning pipeline, medical image preprocessing, transfer learning using MobileNetV2, model training, evaluation using confusion matrix and classification report, and building a real-world healthcare AI simulation.

---

## ⚠️ LIMITATIONS

- Dataset is limited and simulated (not real hospital data)  
- Model may produce false positives  
- Performance can improve with larger and more balanced datasets  

---

## 👨‍💻 AUTHOR

Mahesh Bhakre  

---

## 🌐 CONNECT WITH ME

<a href="https://github.com/maheshbhakre">
<img src="https://img.shields.io/badge/GitHub-Profile-black?style=for-the-badge&logo=github">
</a>

<a href="https://www.linkedin.com/in/maheshbhakreds1242">
<img src="https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin">
</a>

<a href="https://www.instagram.com/mahesh_bhakre__2k06">
<img src="https://img.shields.io/badge/Instagram-Follow-purple?style=for-the-badge&logo=instagram">
</a>

<a href="https://saimfsd.github.io/mahesh-portfolio/">
<img src="https://img.shields.io/badge/Portfolio-Visit%20Website-orange?style=for-the-badge&logo=google-chrome">
</a>

---

## ⭐ NOTE

This project demonstrates a complete AI-powered medical image analysis system with real training, prediction, evaluation, and visualization. It simulates how AI assists doctors in disease detection and decision-making.c
