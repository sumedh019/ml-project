import streamlit as st
import joblib
import pandas as pd

# --- Page setup ---
st.set_page_config(page_title="Pulsar Star Prediction", page_icon="✨")
st.title("🌌 Pulsar Star Classification")
st.write("---")

# --- About the Dataset ---
st.markdown("""
### 📘 About the Dataset
This dataset is from the [Kaggle Playground Series: Pulsar Classification](https://www.kaggle.com/competitions/playground-series-s3e10).  
It contains **synthetic data generated from a deep learning model trained on pulsar signals**.

Each row represents a star observation, and features describe the shape, intensity, and statistics of the signal.

**Goal:** Predict whether a given observation corresponds to a *Pulsar Star* or *Not a Pulsar Star*.
""")

# --- Load trained model ---
try:
    model_dict = joblib.load("model.pkl")
    model = model_dict["model"]
    model_columns = model_dict["columns"]
except Exception as e:
    st.error(f"⚠️ Error loading model.pkl: {e}")
    st.stop()

# --- Input fields ---
st.subheader("🔢 Enter Observation Features")
col1, col2 = st.columns(2)

with col1:
    Mean_Integrated = st.number_input("Mean Integrated", value=0.0)
    SD = st.number_input("Standard Deviation", value=0.0)
    EK = st.number_input("Excess Kurtosis", value=0.0)
    Skewness = st.number_input("Skewness", value=0.0)

with col2:
    Mean_DMSNR_Curve = st.number_input("Mean DMSNR Curve", value=0.0)
    SD_DMSNR_Curve = st.number_input("SD DMSNR Curve", value=0.0)
    EK_DMSNR_Curve = st.number_input("EK DMSNR Curve", value=0.0)
    Skewness_DMSNR_Curve = st.number_input("Skewness DMSNR Curve", value=0.0)

# --- Predict Button ---
if st.button("🚀 Predict"):
    try:
        # Create DataFrame from inputs
        input_data = {
            'Mean_Integrated': Mean_Integrated,
            'SD': SD,
            'EK': EK,
            'Skewness': Skewness,
            'Mean_DMSNR_Curve': Mean_DMSNR_Curve,
            'SD_DMSNR_Curve': SD_DMSNR_Curve,
            'EK_DMSNR_Curve': EK_DMSNR_Curve,
            'Skewness_DMSNR_Curve': Skewness_DMSNR_Curve
        }
        df = pd.DataFrame([input_data])[model_columns]

        # Make prediction
        prediction = model.predict(df)[0]
        prediction_text = "🌟 Pulsar Star" if prediction == 1 else "🪐 Not a Pulsar Star"

        # Display result
        st.success(f"**Prediction:** {prediction_text}")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
