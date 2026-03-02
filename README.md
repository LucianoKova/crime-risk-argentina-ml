# Crime Risk Argentina - ML Project

Machine Learning project focused on predicting weapon usage in urban crimes in Buenos Aires (CABA, 2022).

## 📊 Problem Statement

Can we predict whether a crime involves weapon usage based on:
- Type of crime
- Time of day
- Commune
- Motorcycle usage

## 🧠 Model

- Algorithm: Random Forest Classifier
- Target: `uso_arma`
- Dataset: CABA 2022 crime data

## 📈 Key Challenge

Severe class imbalance (weapon usage is minority class).

## 🖥️ Streamlit App

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py