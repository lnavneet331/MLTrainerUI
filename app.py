import streamlit as st
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

st.write("MLTrainerUI")

Xfile = st.file_uploader("Choose Feature file")
yfile = st.file_uploader("Choose Target file")

X = pd.read_csv(Xfile)
y = pd.read_csv(yfile)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    n_jobs=4
)

model.fit(X_train, y_train)

st.write("Model trained successfully")

st.write("Model evaluation")

st.write("Train score: ", model.score(X_train, y_train))

st.write("Test score: ", model.score(X_test, y_test))

st.write("Model saved successfully")

model.save_model("model.json")

st.write("Model loaded successfully")

