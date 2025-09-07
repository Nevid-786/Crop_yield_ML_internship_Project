
# import streamlit as st
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

df = pd.read_csv("crop_yield.csv")
print(df.shape)

def load_data():
    df = pd.read_csv("crop_yield.csv")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=True)
    for col in ["Fertilizer_Used", "Irrigation_Used"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})
    return df

df = load_data()

print(df.shape)