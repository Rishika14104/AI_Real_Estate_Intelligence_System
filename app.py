import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AI Real Estate Intelligence System",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 AI Real Estate Intelligence System")
st.markdown("---")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

# ----------------------------
# Load Encoders
# ----------------------------
@st.cache_resource
def load_encoders():
    return joblib.load("label_encoders.pkl")

encoders = load_encoders()

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("feature_engineered_dataset.csv")

df = load_data()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "🏠 Home",
        "📊 Dataset Analysis",
        "🤖 Price Prediction",
        "📈 Feature Importance",
        "📉 SHAP Explainability",
        "📍 Market Insights",
        "ℹ About"
    ]
)

# =====================================================
# HOME PAGE
# =====================================================

if page == "🏠 Home":

    st.title("🏠 AI Real Estate Intelligence System")

    st.markdown("""
    ### Welcome!

    This application predicts **Real Estate Property Prices**
    using Machine Learning and provides useful insights about
    the housing market.

    ### Features

    ✅ Property Price Prediction

    ✅ Dataset Analysis

    ✅ Feature Importance

    ✅ SHAP Explainability

    ✅ Market Insights

    ✅ Interactive Dashboard
    """)

    st.markdown("---")

    st.subheader("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Properties", len(df))
    col2.metric("Total Features", df.shape[1])
    col3.metric("Average Price", f"₹ {df['exactPrice'].mean():,.0f}")
    col4.metric("Maximum Price", f"₹ {df['exactPrice'].max():,.0f}")

    st.markdown("---")

    st.subheader("🏡 Property Price Distribution")

    fig = px.histogram(
        df,
        x="exactPrice",
        nbins=40,
        title="Distribution of Property Prices"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("🏙️ Properties by City")

    city_counts = df["city"].value_counts().reset_index()

    city_counts.columns = ["City", "Properties"]

    fig = px.bar(
        city_counts,
        x="City",
        y="Properties",
        color="Properties",
        title="Number of Properties in Each City"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("🏘️ Property Type Distribution")

    property_counts = df["propertyType"].value_counts().reset_index()

    property_counts.columns = ["Property Type", "Count"]

    fig = px.pie(
        property_counts,
        names="Property Type",
        values="Count",
        hole=0.45,
        title="Property Types"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.success("✅ Dashboard Loaded Successfully!")

# =====================================================
# DATASET ANALYSIS
# =====================================================

elif page == "📊 Dataset Analysis":

    st.title("📊 Dataset Analysis Dashboard")

    st.markdown("Explore the dataset using interactive visualizations.")

    st.markdown("---")

    # Dataset Preview
    st.subheader("📄 Dataset Preview")

    rows = st.slider(
        "Select number of rows",
        min_value=5,
        max_value=50,
        value=10
    )

    st.dataframe(df.head(rows))

    st.markdown("---")

    # Dataset Information
    st.subheader("📌 Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    st.markdown("---")

    # Missing Values
    st.subheader("❓ Missing Values")

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:

        st.success("✅ No Missing Values Found!")

    else:

        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Values": missing.values
        })

        fig = px.bar(
            missing_df,
            x="Column",
            y="Missing Values",
            color="Missing Values",
            title="Missing Values by Column"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Numerical Feature Distribution
    st.subheader("📈 Numerical Feature Distribution")

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    selected_numeric = st.selectbox(
        "Choose Numerical Feature",
        numeric_columns
    )

    fig = px.histogram(
        df,
        x=selected_numeric,
        nbins=40,
        title=f"{selected_numeric} Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Categorical Feature Distribution
    st.subheader("🏘️ Categorical Feature Distribution")

    categorical_columns = df.select_dtypes(include="object").columns.tolist()

    if len(categorical_columns) > 0:

        selected_cat = st.selectbox(
            "Choose Categorical Feature",
            categorical_columns
        )

        counts = (
            df[selected_cat]
            .value_counts()
            .reset_index()
        )

        counts.columns = [selected_cat, "Count"]

        fig = px.bar(
            counts,
            x=selected_cat,
            y="Count",
            color="Count",
            title=f"{selected_cat} Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:

        st.info("No categorical columns available (all features are encoded).")

    st.markdown("---")

    # Correlation Heatmap
    st.subheader("🔥 Correlation Heatmap")

    corr = df.select_dtypes(include=np.number).corr()

    fig = px.imshow(
        corr,
        text_auto=False,
        aspect="auto",
        color_continuous_scale="Viridis"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Summary Statistics
    st.subheader("📋 Summary Statistics")

    st.dataframe(df.describe())

# =====================================================
# PRICE PREDICTION
# =====================================================

elif page == "🤖 Price Prediction":

    st.title("🤖 House Price Prediction")

    st.write("Enter the property details below.")

    # --------------------------
    # User Inputs
    # --------------------------

    sqftPrice = st.number_input("Sqft Price", min_value=0.0, value=150.0)

    securityDeposit = st.number_input("Security Deposit", min_value=0.0, value=100000.0)

    propertyType = st.number_input("Property Type (Encoded)", value=0)

    noOfLifts = st.number_input("No. of Lifts", value=1)

    maintenanceChargesFrequency = st.number_input(
        "Maintenance Charges Frequency",
        value=0
    )

    maintenanceCharges = st.number_input(
        "Maintenance Charges",
        min_value=0.0,
        value=2000.0
    )

    locality = st.number_input("Locality (Encoded)", value=0)

    furnishing = st.number_input("Furnishing (Encoded)", value=0)

    flrNum = st.number_input("Floor Number", value=1)

    firstMonthCharges = st.number_input(
        "First Month Charges",
        value=0
    )

    facing = st.number_input("Facing (Encoded)", value=0)

    totalFlrNum = st.number_input("Total Floors", value=5)

    city = st.number_input("City (Encoded)", value=0)

    carpetAreaUnit = st.number_input(
        "Carpet Area Unit (Encoded)",
        value=0
    )

    carpetArea = st.number_input(
        "Carpet Area",
        min_value=100.0,
        value=1200.0
    )

    brokerage = st.number_input(
        "Brokerage",
        min_value=0.0,
        value=0.0
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=10,
        value=2
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=10,
        value=2
    )

    balconies = st.number_input(
        "Balconies",
        min_value=0,
        max_value=10,
        value=1
    )

    posted_year = st.number_input(
        "Posted Year",
        value=2025
    )

    posted_month = st.number_input(
        "Posted Month",
        min_value=1,
        max_value=12,
        value=6
    )

    st.markdown("---")

    st.subheader("Amenities")

    lift = st.checkbox("Lift")

    gym = st.checkbox("Gymnasium")

    swimming = st.checkbox("Swimming Pool")

    park = st.checkbox("Park")

    power = st.checkbox("Power Backup")

    cctv = st.checkbox("CCTV Camera")

    clubhouse = st.checkbox("Club House")

    security = st.checkbox("Security")

    # --------------------------
    # Prediction
    # --------------------------

    if st.button("Predict Price"):

        sample = pd.DataFrame(
            np.zeros((1, len(df.columns)-1)),
            columns=df.drop("exactPrice", axis=1).columns
        )

        sample["sqftPrice"] = sqftPrice
        sample["securityDeposit"] = securityDeposit
        sample["propertyType"] = propertyType
        sample["noOfLifts"] = noOfLifts
        sample["maintenanceChargesFrequency"] = maintenanceChargesFrequency
        sample["maintenanceCharges"] = maintenanceCharges
        sample["locality"] = locality
        sample["furnishing"] = furnishing
        sample["flrNum"] = flrNum
        sample["firstMonthCharges"] = firstMonthCharges
        sample["facing"] = facing
        sample["totalFlrNum"] = totalFlrNum
        sample["city"] = city
        sample["carpetAreaUnit"] = carpetAreaUnit
        sample["carpetArea"] = carpetArea
        sample["brokerage"] = brokerage
        sample["bedrooms"] = bedrooms
        sample["bathrooms"] = bathrooms
        sample["balconies"] = balconies

        if "Posted_Year" in sample.columns:
            sample["Posted_Year"] = posted_year

        if "Posted_Month" in sample.columns:
            sample["Posted_Month"] = posted_month

        if "Lift" in sample.columns:
            sample["Lift"] = int(lift)

        if "Gymnasium" in sample.columns:
            sample["Gymnasium"] = int(gym)

        if "Swimming_Pool" in sample.columns:
            sample["Swimming_Pool"] = int(swimming)

        if "Park" in sample.columns:
            sample["Park"] = int(park)

        if "Power_Back_Up" in sample.columns:
            sample["Power_Back_Up"] = int(power)

        if "CCTV_Camera" in sample.columns:
            sample["CCTV_Camera"] = int(cctv)

        if "Club_House" in sample.columns:
            sample["Club_House"] = int(clubhouse)

        if "Security" in sample.columns:
            sample["Security"] = int(security)

        prediction = model.predict(sample)[0]

        st.success(f"🏠 Estimated Property Price: ₹ {prediction:,.2f}")

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

elif page == "📈 Feature Importance":

    st.title("📈 Feature Importance")

    st.write("Top Important Features Used by the Machine Learning Model")

    if hasattr(model, "feature_importances_"):

        importance = pd.DataFrame({
            "Feature": df.drop("exactPrice", axis=1).columns,
            "Importance": model.feature_importances_
        })

        importance = importance.sort_values(
            by="Importance",
            ascending=False
        )

        st.dataframe(importance)

        fig = px.bar(
            importance.head(20),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 20 Important Features"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "This model does not support feature importance."
        )

# =====================================================
# SHAP EXPLAINABILITY
# =====================================================

elif page == "📉 SHAP Explainability":

    st.title("📉 SHAP Explainability")

    st.write(
        "SHAP explains how each feature contributes to the prediction."
    )

    try:

        X = df.drop("exactPrice", axis=1)

        sample = X.sample(200, random_state=42)

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(sample)

        fig, ax = plt.subplots(figsize=(10,6))

        shap.summary_plot(
            shap_values,
            sample,
            show=False
        )

        st.pyplot(fig)

    except Exception as e:

        st.error(str(e))

# =====================================================
# MARKET INSIGHTS
# =====================================================

elif page == "📍 Market Insights":

    st.title("📍 Market Insights")

    st.subheader("Average Property Price")

    avg_price = df["exactPrice"].mean()

    st.metric(
        "Average Price",
        f"₹ {avg_price:,.0f}"
    )

    st.markdown("---")

    st.subheader("Top Cities")

    city = (
        df.groupby("city")["exactPrice"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        city,
        x="city",
        y="exactPrice",
        color="exactPrice",
        title="Average Price by City"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Bedroom Analysis")

    bed = (
        df.groupby("bedrooms")["exactPrice"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        bed,
        x="bedrooms",
        y="exactPrice",
        markers=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# ABOUT
# =====================================================

elif page == "ℹ About":

    st.title("ℹ About Project")

    st.markdown("""
# 🏠 AI Real Estate Intelligence System

### Project Objective

Predict real estate prices using Machine Learning and provide market insights.

---

### Dataset

Real Estate Housing Dataset

---

### Machine Learning Models

- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- CatBoost
- LightGBM

---

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- XGBoost
- LightGBM
- SHAP
- Plotly
- Streamlit

---

### Developed By

**Kosireddy Rishika**

AI & Machine Learning Project
""")