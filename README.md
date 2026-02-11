# 💳 Credit Card Fraud Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Status](https://img.shields.io/badge/Project-Complete-brightgreen)

## 📌 Project Overview

Credit card fraud is a major issue in financial systems. This project builds an end-to-end Machine Learning solution to detect fraudulent credit card transactions. Due to the highly imbalanced nature of the dataset, special techniques like SMOTE are applied to improve fraud detection performance.

The project evaluates multiple models and measures performance using industry-standard metrics such as ROC-AUC, Confusion Matrix, Precision, Recall, and F1-score.

---

## 🚀 Features

- Handles highly imbalanced dataset using SMOTE
- Trains multiple ML models
- Uses Random Forest for high-performance classification
- Evaluates using ROC Curve and Confusion Matrix
- Saves trained model for future use
- Clean, reproducible ML pipeline

---

## 📂 Project Structure

Credit-Card-Fraud-Detection-ML/
│
├── data/
│ └── creditcard.csv
│
├── notebooks/
│ └── fraud_detection.ipynb
│
├── images/
│ ├── confusion_matrix.png
│ └── roc_curve.png
│
├── fraud_model.pkl
├── requirements.txt
└── README.md


---

## 📊 Dataset

- Source: Kaggle Credit Card Fraud Dataset
- Total Transactions: 284,807
- Fraud Cases: 492 (0.17%)
- Features: PCA-transformed (V1–V28), Time, Amount, Class

🔗 Dataset Link:  
https://www.kaggle.com/mlg-ulb/creditcardfraud

---

## ⚙️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
- Joblib

---

## 🧠 Machine Learning Models

- Logistic Regression
- Random Forest Classifier (Primary Model)

---

## ⚖️ Handling Imbalanced Data

This dataset is highly imbalanced. To address this, SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training data to generate synthetic fraud samples and balance the classes.

---

## 📈 Model Evaluation

The following evaluation metrics are used:

- Confusion Matrix
- ROC Curve
- ROC-AUC Score
- Precision
- Recall
- F1-Score

### Sample Results (Random Forest)

ROC-AUC Score: ~0.98
Precision: High
Recall: High
F1-Score: High


> Note: Actual results may vary based on random state and environment.

---

## 🖼️ Visual Results

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

### ROC Curve
![ROC Curve](images/roc_curve.png)

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection-ML.git
cd Credit-Card-Fraud-Detection-ML
2️⃣ Install Requirements
pip install -r requirements.txt
3️⃣ Add Dataset
Download the dataset from Kaggle and place creditcard.csv inside:

data/creditcard.csv
4️⃣ Run Notebook
Open Jupyter Notebook and run:

notebooks/fraud_detection.ipynb
💾 Saved Model
The trained Random Forest model is saved as:

fraud_model.pkl
This can be loaded later for inference or deployment.

🔮 Future Improvements
Add XGBoost or LightGBM

Add Streamlit Web Application

Hyperparameter tuning

Real-time fraud detection API

Model explainability (SHAP, LIME)

⭐ Why This Project?
Real-world ML use case

Industry-relevant imbalanced classification

Strong evaluation methodology

Great for portfolios and resumes

Beginner-friendly but professional

🤝 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

📜 License
This project is open-source and available under the MIT License.

🙌 Acknowledgements
ULB Machine Learning Group
