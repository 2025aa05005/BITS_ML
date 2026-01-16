import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Heart Disease Classification App")

# Load models
with open("model/saved_models.pkl", "rb") as f:
    models, scaler, metrics = pickle.load(f)

# Upload dataset
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]

st.subheader("Model Evaluation Metrics")
st.write(metrics[model_name])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    X = scaler.transform(data.drop("HeartDisease", axis=1))
    y_true = data["HeartDisease"]
    y_pred = model.predict(X)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y_true, y_pred))
