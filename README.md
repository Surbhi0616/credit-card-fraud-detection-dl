# credit-card-fraud-detection-dl
Deep learning project for credit card fraud detection using DNN and Autoencoder.

# Credit Card Fraud Detection using Deep Learning

## 📘 Overview
This project implements two deep learning models for detecting fraudulent credit card transactions:
1. **Supervised Model:** Deep Neural Network (DNN)
2. **Unsupervised Model:** Autoencoder (Anomaly Detection)

Both models are compared in terms of AUC, Precision, Recall, and F1-score.

---

## 📊 Dataset
Dataset: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---

## 🧩 Libraries Used
- Python 3.10+
- PyTorch
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## 🧠 Execution Steps
1. Upload dataset in Colab.
2. Run all cells in order:
   - Data preprocessing
   - DNN training
   - Autoencoder training
   - Evaluation & visualization
3. Saved models:
   - `dnn_model.pth`
   - `autoencoder_model.pth`

---

## 🏁 Results

| Model | AUC | Precision | Recall | F1 |
|--------|-----|------------|--------|----|
| DNN | 0.99 | 0.93 | 0.88 | 0.90 |
| Autoencoder | 0.97 | 0.90 | 0.85 | 0.87 |

---

## 👩‍💻 Team Members
- **Surbhi** — Model design, Autoencoder implementation, result visualization  
- **[Teammate Name]** — Data preprocessing, DNN model training, documentation

---

## 🧾 Notes
- SHAP explainability was attempted but skipped due to long runtime.  
- Both models can be deployed using Flask + PyTorch for real-time detection.
