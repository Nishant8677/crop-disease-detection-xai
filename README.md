# 🌿 Crop Disease Detection with Explainable AI (Grad-CAM)

An end-to-end plant disease detection system built using Deep Learning (ResNet18) and enhanced with Explainable AI (Grad-CAM) to provide transparent and interpretable predictions from leaf images.

This project focuses not only on accuracy, but also on model trust, uncertainty awareness, and explainability, making it suitable for real-world agricultural decision support.

---

## 🚀 Key Features

- Multi-class plant disease classification (38 classes)
- ResNet18 with transfer learning (PyTorch)
- Top-k class probability visualization
- Low-confidence prediction warnings
- Explainable AI using Grad-CAM
- Interactive Grad-CAM intensity slider
- Downloadable Grad-CAM explanations
- Dark-mode safe Streamlit UI
- Baseline CNN comparison for justification
- Clear model limitations disclosure

---

## 🖼️ Demo

Add screenshots after running the app locally.

Recommended screenshots:
- App home screen
- Prediction with confidence bar
- Grad-CAM overlay visualization

---

## 🧠 Model Overview

Final Model:
- Architecture: ResNet18
- Framework: PyTorch
- Input size: 224 × 224 RGB
- Number of classes: 38
- Validation accuracy: ~96%

Baseline Model:
- Architecture: Simple CNN (2 convolution layers)
- Purpose: Performance comparison and architectural justification
- Accuracy: ~80–85%

A baseline CNN was trained to demonstrate that the final ResNet18 model provides significantly better generalization and performance.

---

## 🔍 Explainable AI (Grad-CAM)

Grad-CAM is used to visualize which regions of the leaf influenced the model’s decision.

Important notes:
- Grad-CAM provides coarse, semantic explanations
- It is not pixel-level segmentation
- Highlighted regions indicate decision influence, not guaranteed correctness

This helps verify that the model focuses on disease-relevant regions instead of background artifacts.

---

## 📂 Project Structure

crop-disease-detection-xai/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── utils/
│   ├── model.py
│   ├── preprocessing.py
│   └── gradcam.py
└── model/
    └── resnet18_plant_disease.pth (not included)

---

## ⚙️ Installation and Setup

1. Clone the repository:

git clone https://github.com/YOUR_USERNAME/crop-disease-detection-xai.git  
cd crop-disease-detection-xai

2. Create a virtual environment (optional but recommended):

python -m venv venv  
source venv/bin/activate   (Linux / macOS)  
venv\\Scripts\\activate    (Windows)

3. Install dependencies:

pip install -r requirements.txt

---

## 📥 Model Weights

The trained model weights are not included due to size constraints.

To use your trained model:
1. Train the model in Colab or locally
2. Download the resnet18_plant_disease.pth file
3. Place it inside the model/ directory

Expected path:
model/resnet18_plant_disease.pth

---

## ▶️ Running the Application

Run the Streamlit app using:

streamlit run app.py

Then open your browser at:
http://localhost:8501

---

## ⚠️ Model Limitations

- Trained on the PlantVillage dataset (lab-controlled images)
- Performance may vary on real-world field images
- Sensitive to lighting and camera quality
- Grad-CAM provides interpretability, not proof of correctness

---

## 🧠 Key Learnings

- Accuracy alone is insufficient for real-world ML systems
- Explainability helps identify shortcut learning
- Proper hook and gradient management is critical in PyTorch
- Transfer learning outperforms training from scratch for vision tasks
- UX design improves trust in ML-based applications

---

## 🎯 Interview One-Liner

I built a plant disease detection system using ResNet18 and validated its predictions using Grad-CAM, focusing on explainability, uncertainty awareness, and real-world deployment considerations.

---

## 📌 Future Improvements

- Fine-tuning on real-world field images
- Mobile-friendly deployment
- Disease severity estimation
- Multilingual farmer-oriented interface

---

## 📄 License

This project is for educational and research purposes only.
