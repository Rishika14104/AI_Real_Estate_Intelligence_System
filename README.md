# 🏠 AI Real Estate Intelligence System

## 📌 Project Overview

The **AI Real Estate Intelligence System** is an end-to-end Machine Learning project that predicts property prices based on property characteristics and amenities. The project also provides an interactive Streamlit dashboard for data analysis, prediction, feature importance, SHAP explainability, and market insights.

---

## 🚀 Features

* 🏠 Property Price Prediction
* 📊 Interactive Dataset Analysis
* 📈 Feature Importance Visualization
* 📉 SHAP Explainable AI
* 📍 Market Insights Dashboard
* 💻 Interactive Streamlit Web Application

---

## 📂 Project Structure

```text
AI_Real_Estate_Intelligence_System/
│
├── app.py
├── AI_Real_Estate_Intelligence.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── cleaned_real_estate.csv
│   └── feature_engineered_dataset.csv
│
├── models/
│   ├── best_model.pkl
│   └── label_encoders.pkl
│
├── screenshots/
│
└── presentation/
```

---

## 📊 Dataset

The dataset contains various residential property attributes, including:

* Property Type
* City
* Locality
* Carpet Area
* Bedrooms
* Bathrooms
* Balconies
* Furnishing
* Floor Number
* Maintenance Charges
* Security Deposit
* Property Amenities
* Property Price (Target Variable)

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* CatBoost
* XGBoost
* LightGBM
* SHAP
* Plotly
* Matplotlib
* Streamlit
* Joblib

---

## 🤖 Machine Learning Models

The following regression models were trained and compared:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor
* CatBoost Regressor
* LightGBM Regressor

The best-performing model was saved as:

```text
models/best_model.pkl
```

---

## 📈 Workflow

```text
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Feature Engineering
   │
   ▼
Label Encoding
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
Best Model Selection
   │
   ▼
Model Saving
   │
   ▼
Streamlit Deployment
```

---

## 📷 Application Modules

### 🏠 Home

* Project overview
* KPI cards
* Price distribution
* Property distribution

### 📊 Dataset Analysis

* Dataset preview
* Missing values
* Correlation heatmap
* Statistical summary

### 🤖 Price Prediction

* Property price prediction using the trained ML model

### 📈 Feature Importance

* Top important features used by the model

### 📉 SHAP Explainability

* Model interpretation using SHAP

### 📍 Market Insights

* Average property prices
* City-wise analysis
* Bedroom-wise analysis

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Rishika14104/AI_Real_Estate_Intelligence_System.git
```

Navigate to the project directory:

```bash
cd AI_Real_Estate_Intelligence_System
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📸 Screenshots

Add screenshots of your application in the `screenshots/` folder.

Examples:

* Home Page
* Dataset Analysis
* Prediction Page
* Feature Importance
* SHAP Explainability
* Market Insights

---

## 📌 Future Enhancements

* Interactive Maps
* Advanced Property Search
* Recommendation System
* Real-Time Property Data Integration
* Cloud Deployment
* User Authentication

---

## 👩‍💻 Author

**Rishika Kosireddy**

Aspiring Data Scientist | Machine Learning Enthusiast | Python Developer

---

<img width="1920" height="1020" alt="Image" src="https://github.com/user-attachments/assets/c9d35f06-1b79-47e9-828e-9dbd31e736d7" />

<img width="1920" height="1020" alt="Image" src="https://github.com/user-attachments/assets/b3698e8e-c48d-454a-b049-36f9e5a1da63" />

<img width="1920" height="1020" alt="Image" src="https://github.com/user-attachments/assets/0fb97ad5-3798-46a1-8563-91757f099ce1" />

<img width="1920" height="1020" alt="Image" src="https://github.com/user-attachments/assets/acf6a20e-7cae-46ea-acd9-e382f1255749" />
