
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Crop Yield Prediction", layout="wide")

#backgrond color formatting
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
   background-image: linear-gradient(to right top, #003800, #034706, #0b570b, #146710, #1d7715, #21811b, #248c20, #279626, #249c2d, #21a234, #1ba83b, #14ae42);
   
[data-testid="stSidebar"] {
  background-image: linear-gradient(to bottom, #0e2c0e, #134014, #185418, #1f6a1d, #268020, #268020, #268020, #268020, #1f6a1d, #185418, #134014, #0e2c0e);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Loading data
@st.cache_data
def load_data():
    df = pd.read_csv("crop_yield.csv")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=True)
    for col in ["Fertilizer_Used", "Irrigation_Used"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})
    return df

df = load_data()

# Sidebar 
st.sidebar.header("Filters")
crop_list = df["Crop"].unique()
crop = st.sidebar.selectbox("Select Crop", crop_list)
df_crop = df[df["Crop"] == crop]

region = st.sidebar.selectbox("Region", df_crop["Region"].unique())
soil = st.sidebar.selectbox("Soil Type", df_crop["Soil_Type"].unique())
weather = st.sidebar.selectbox("Weather Condition", df_crop["Weather_Condition"].unique())
# team info
st.sidebar.markdown("---")
st.sidebar.subheader("👨‍👩‍👧 Team Info")

team_data = {
    "Name": ["Nevid", "Badal", "Harsh"],
    "Role": ["ML Engineer", "Data Analyst", "Streamlit UI"],
    "LinkedIn": [
        "[Linkedin](https://www.linkedin.com/in/nevid786)",  
        "[Linkedin](https://www.linkedin.com/in/badal-617143366/)",        
        "[Linkedin](linkedin.com/in/harsh-gautam-h22)"              
    ]
    
}

team_df = pd.DataFrame(team_data,index=[1,2,3])

# Show table
st.sidebar.table(team_df)
col1, col2, col3 = st.sidebar.columns(3)

with col1:
    st.image("./images/Nevid.jpg", caption="Nevid", width=80)

with col2:
    st.image("./images/badal.jpg", caption="Badal", width=80)

with col3:
    st.image("./images/harsh.png", caption="Harsh", width=80)


# Model trainning 
MODEL_PATH = "crop_yield_model.pkl"

if os.path.exists(MODEL_PATH):
    model, X_test, y_test, y_pred = joblib.load(MODEL_PATH)
else:
    X = df.drop(columns=["Yield_tons_per_hectare", "Days_to_Harvest"])
    y = df["Yield_tons_per_hectare"]

    categorical = ["Region", "Soil_Type", "Crop", "Weather_Condition"]
    numeric = ["Rainfall_mm", "Temperature_Celsius", "Fertilizer_Used", "Irrigation_Used"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    joblib.dump((model, X_test, y_test, y_pred), MODEL_PATH)

# Tabs
tabs = st.tabs(["🔮 Prediction", "📊 EDA", "📈 Performance"])

# predicting section
with tabs[0]:
    st.header(" Crop Yield Prediction by Linear Regresson Model:")

    rainfall = st.slider("Rainfall (mm)", 0, 2000, 900)
    temp = st.slider("Temperature (°C)", 0, 50, 25)
    fertilizer = st.selectbox("Fertilizer Used", ["No", "Yes"])
    irrigation = st.selectbox("Irrigation Used", ["No", "Yes"])

    user_input = pd.DataFrame([{
        "Region": region,
        "Soil_Type": soil,
        "Crop": crop,
        "Rainfall_mm": rainfall,
        "Temperature_Celsius": temp,
        "Fertilizer_Used": 1 if fertilizer == "Yes" else 0,
        "Irrigation_Used": 1 if irrigation == "Yes" else 0,
        "Weather_Condition": weather
    }])

    if st.button(" Predict"):
        pred_yield = model.predict(user_input)[0]
        
        # Center the text using markdown
        st.markdown(
            f"<h3 style='text-align: center;'> Predicted Yield: {pred_yield:.2f} tons/hectare</h3>",
            unsafe_allow_html=True
        )
    
    


# Eda section
with tabs[1]:
    st.header("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Yield Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(df_crop["Yield_tons_per_hectare"], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Rainfall vs Yield")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(x="Rainfall_mm", y="Yield_tons_per_hectare", data=df_crop, ax=ax)
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Temperature vs Yield")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(x="Temperature_Celsius", y="Yield_tons_per_hectare", data=df_crop, ax=ax)
        st.pyplot(fig)

    with col4:
        st.subheader("Soil Type vs Yield")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(x="Soil_Type", y="Yield_tons_per_hectare", data=df_crop, ax=ax)
        st.pyplot(fig)

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Fertilizer vs Yield")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(x="Fertilizer_Used", y="Yield_tons_per_hectare",
                    data=df_crop.replace({0: "No", 1: "Yes"}), ax=ax)
        st.pyplot(fig)

    with col6:
        st.subheader("Irrigation vs Yield")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(x="Irrigation_Used", y="Yield_tons_per_hectare",
                    data=df_crop.replace({0: "No", 1: "Yes"}), ax=ax)
        st.pyplot(fig)

    st.subheader("Weather vs Yield")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="Weather_Condition", y="Yield_tons_per_hectare", data=df_crop, ax=ax)
    st.pyplot(fig)


#Performance tab
with tabs[2]:
    st.header("📈 Model Performance")

    from sklearn.metrics import mean_absolute_error

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.markdown(f"""
    - **MAE:** {mae:.2f}  
    - **RMSE:** {rmse:.2f}  
    - **R² Score:** {r2:.2f}  
    - **Total Rows in Dataset:** {df.shape[0]}  
    """)

    
    col1, col2, col3 = st.columns(3)

    # 1. Actual vs Predicted
    with col1:
        st.subheader("Actual vs Predicted")
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        sns.scatterplot(x=y_test, y=y_pred, ax=ax1, alpha=0.6, color="green")
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax1.set_xlabel("Actual Yield")
        ax1.set_ylabel("Predicted Yield")
        st.pyplot(fig1)

    # 2. Residuals plot
    with col2:
        st.subheader("Residuals")
        residuals = y_test - y_pred
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        sns.scatterplot(x=y_pred, y=residuals, ax=ax2, alpha=0.6, color="blue")
        ax2.axhline(0, color='red', linestyle='--')
        ax2.set_xlabel("Predicted Yield")
        ax2.set_ylabel("Residuals")
        st.pyplot(fig2)

    # 3. Error distribution
    with col3:
        st.subheader("Error Distribution")
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        sns.histplot(residuals, bins=20, kde=True, ax=ax3, color="purple")
        ax3.set_xlabel("Residual (Actual - Predicted)")
        st.pyplot(fig3)

    
