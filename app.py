import os
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_small.pkl")

st.write("Buscando modelo en:", MODEL_PATH)
st.write("¿Existe?:", os.path.exists(MODEL_PATH))

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.success("Modelo cargado correctamente")
else:
    st.error("Modelo no encontrado")