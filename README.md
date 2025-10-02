# 🔎 XAI-Driven Robustness Analysis of YOLO  

This project implements an **explainable and robust anomaly detection pipeline** using the **Ultralytics YOLOv11** model. The workflow spans **data augmentation, preprocessing, training, robustness testing, and explainability** with **Grad-CAM, Eigen-CAM, and Saliency Maps**, alongside **uncertainty estimation** for anomaly detection.  


---

## ⚙️ Workflow Overview  

### 1️⃣ Data Augmentation  
- **augmentation.ipynb** applies noise, blur, brightness changes, and flips.  
- Output stored in `augmented_data/`.  

### 2️⃣ Preprocessing  
- **preprocess.ipynb** ensures correct label formatting and image consistency.  

### 3️⃣ Dataset Splitting  
- Train/Val/Test (70/20/10) in `augmented_data_split/`.  

### 4️⃣ Model Training  
- **training.ipynb** trains YOLOv11m with early stopping.  
- Best weights saved at: `runs/detect/train*/weights/best.pt`.  

### 5️⃣ Evaluation & Explainability  
- **grad_cam.ipynb, eigen_cam.ipynb, saliency_map.ipynb** → highlight attention regions.  
- **robustness_testing.ipynb** → test on perturbed images.  
- **anomaly_detection.ipynb** → Monte Carlo Dropout for uncertainty estimation.  

---

## 📊 Training Results & Comparison  

We trained YOLOv11 models with different **input resolutions (416 vs 640)**:  

| Image Size | Precision (P) | Recall (R) | mAP50 | mAP50-95 | Notes |
|------------|---------------|------------|-------|----------|-------|
| **416x416** | 0.907         | 0.829      | 0.901 | 0.697    | Best balance, higher overall mAP |
| **640x640** | 0.886         | 0.818      | 0.877 | 0.667    | Slightly weaker, but handles larger objects better |

### 🔍 Class-wise Breakdown (example, 640x640):  
- **Van** → P=0.938, R=0.933, mAP50=0.973, mAP50-95=0.826  
- **Truck** → P=0.950, R=0.913, mAP50=0.964, mAP50-95=0.802  
- **Pedestrian** → P=0.943, R=0.942, mAP50=0.973, mAP50-95=0.835  

📌 **Interpretation:**  
- **416x416** achieved slightly higher metrics overall (best suited for balanced tasks).  
- **640x640** was slightly weaker on mAP, but more consistent with larger objects (like trucks/vans).  
- **Pedestrian detection** was strong across both, showing YOLOv11 generalizes well on smaller objects.  

---

