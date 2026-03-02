import os
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

st.title("Predicción de Uso de Arma en Delitos - CABA")

MODEL_PATH = "models/random_forest_model.pkl"

@st.cache_resource
def train_model():
    df = pd.read_csv("data/delitos_small.csv")

    # Usar solo una muestra para que sea más rápido
    df = df.sample(n=20000, random_state=42)

    X = df[["franja", "comuna", "uso_moto", "tipo"]]
    y = df["uso_arma"]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=30,  # menos árboles = más rápido
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    return model

# Intentar cargar modelo
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.warning("Modelo no encontrado. Entrenando modelo...")
    model = train_model()