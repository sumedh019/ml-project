import streamlit as st
import requests

st.set_page_config(page_title="Pulsar Star Prediction")
st.title("Pulsar Star Classification")
st.write("---")


st.markdown("""
### About the Dataset
This dataset is from the [Kaggle Playground Series: Pulsar Classification](https://www.kaggle.com/competitions/playground-series-s3e10).  
It contains synthetic data generated from a deep learning model trained on pulsar signals. Each row represents a star observation, and features describe the shape, intensity, and statistics of the signal.

The goal is to predict whether a given observation corresponds to a pulsar.
""")

Mean_Integrated = st.number_input("Mean_Integrated")
SD = st.number_input("SD")
EK = st.number_input("EK")
Skewness = st.number_input("Skewness")
Mean_DMSNR_Curve = st.number_input("Mean_DMSNR_Curve")
SD_DMSNR_Curve = st.number_input("SD_DMSNR_Curve")
EK_DMSNR_Curve = st.number_input("EK_DMSNR_Curve")
Skewness_DMSNR_Curve = st.number_input("Skewness_DMSNR_Curve")

if st.button("Predict"):
    data = {
        'Mean_Integrated': Mean_Integrated,
        'SD': SD,
        'EK': EK,
        'Skewness': Skewness,
        'Mean_DMSNR_Curve': Mean_DMSNR_Curve,
        'SD_DMSNR_Curve': SD_DMSNR_Curve,
        'EK_DMSNR_Curve': EK_DMSNR_Curve,
        'Skewness_DMSNR_Curve': Skewness_DMSNR_Curve
    }
    response = requests.post("http://127.0.0.1:5000/predict", json=data)
    if response.status_code == 200:
        prediction_num = response.json()['prediction']
        prediction_text = "Pulsar Star" if prediction_num == 1 else "Not a Pulsar Star"
        st.success(f"Prediction: {prediction_text}")
    else:
        st.error("Error occurred while predicting!")
