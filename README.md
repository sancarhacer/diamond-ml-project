# 💎 Diamond Price Prediction Model (SVR Regression)

## Project Overview

This project implements a Machine Learning solution to predict the retail price of diamonds based on their physical attributes, such as carat weight, cut quality, color, clarity, and dimensions.

It was developed as a personal portfolio piece to master the end-to-end data science pipeline, focusing particularly on robust **data cleaning**, **outlier management**, and advanced regression with **Support Vector Regressor (SVR)**.

## 🚀 Key Learning Goals

During the development of this project, I focused on achieving proficiency in the following ML skills:

1.  **Data Preprocessing Mastery:** Implementing stringent data cleaning, including the removal of invalid records (zero dimensions in `x`, `y`, `z`).
2.  **Outlier Handling:** Utilizing visual EDA (`seaborn.scatterplot`) and statistical methods to identify and constrain extreme outliers in `depth`, `table`, and `y/z` dimensions.
3.  **Encoding and Scaling:** Applying **Label Encoding** for categorical features and **StandardScaler** for numerical feature standardization, a requirement for SVR.
4.  **Advanced Regression:** Successfully implementing **Support Vector Regressor (SVR)** and optimizing its parameters using **GridSearchCV**.
5.  **Model Persistence:** Saving the final model, encoder, and scaler objects together using `pickle` for easy deployment.

---

## ⚙️ Technical Details and Modeling Pipeline

### 1. Data Source and Initial Cleanup

* **Source:** Kaggle - Diamonds Dataset
* **Initial Action:** Dropped the initial index column (`Unnamed: 0`).
* **Data Cleaning Focus:** Removed records where `x`, `y`, or `z` dimensions were zero (7 records). Outliers in `depth` (45-75), `table` (40-80), `y` (<30), and `z` (2-30) were clipped/removed based on analysis.
* **Final Data Size:** 53,907 cleaned records.

### 2. Feature Engineering

| Feature | Type | Transformation |
| :--- | :--- | :--- |
| `cut`, `color`, `clarity` | Categorical | **Label Encoding** |
| `carat`, `depth`, `table`, etc. | Numerical | **StandardScaler** |

### 3. Model Training and Results

* **Algorithm:** **Support Vector Regressor (SVR)**
* **Optimization:** **GridSearchCV** was used to find the optimal C, gamma, and kernel.
* **Data Split:** 75% Training, 25% Testing (`random_state=15`).

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **$R^2$ Score** | **0.9452** | The model explains **94.5%** of the variance in diamond prices, indicating high prediction accuracy on the test set. |

### 4. Saved Model

The finalized model, along with the necessary preprocessing tools, is saved as:
* `26-diamond_model_complete.pkl`

---

## 💻 Installation and Execution

To run the analysis and training script locally:

1.  **Clone the repository:**
    ```bash
    git clone [Your GitHub Repo URL]
    cd diamond-price-predictor
    ```

2.  **Set up the virtual environment:**
    ```bash
    python -m venv .venv
    # Activate on macOS/Linux
    source .venv/bin/activate
    # Activate on Windows
    .\.venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirement.txt
    ```

4.  **Execute the training script:**
    ```bash
    python app.py
    ```

## 🛠️ Technologies Used

* **Python Libraries:** Pandas, NumPy, Matplotlib, Seaborn
* **ML Stack:** Scikit-learn (SVR, GridSearchCV, StandardScaler, LabelEncoder)
* **Serialization:** `pickle`

---

## 💡 Future Enhancements

Future steps to expand this project and further my ML skills include:

1.  **Comparative Modeling:** Evaluating the performance against boosting algorithms like LightGBM or XGBoost.
2.  **Model Deployment:** Building a user-friendly prediction interface using **Streamlit** or **Flask**.
3.  **Hyperparameter Tuning:** Implementing more advanced tuning methods (e.g., Random Search) for more efficient optimization.

---
