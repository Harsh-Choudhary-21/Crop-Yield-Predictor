# 🌾 Crop Yield Prediction — Linear Regression with TensorFlow

This project implements a **Linear Regression model** in **TensorFlow/Keras** to predict **crop yields** using environmental and agricultural features.  
The dataset is sourced directly from **Kaggle** via [KaggleHub](https://github.com/Kaggle/kagglehub), ensuring reproducibility and easy integration.

---

## 🚀 Features
- Fetch dataset seamlessly from Kaggle using **KaggleHub**.
- Preprocess data with **Pandas** and **Scikit-learn**.
- Implement a **Linear Regression model** in **TensorFlow 2.x**.
- Train for up to **50 epochs** with **EarlyStopping** to avoid overfitting.
- Visualize:
  - Training **loss curves**
  - Predictions vs Actuals using **Matplotlib**

---

## 📊 Dataset
- **Source:** [Crop Yield Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)
- **Files Used:** `yield_df.csv`
- **Features:** Rainfall, pesticides, temperature, crop type, etc.
- **Target:** Crop yield

---

## 🛠️ Installation

Clone the repo:

```bash
git clone https://github.com/<your-username>/crop-yield-linear-regression.git
cd crop-yield-linear-regression
