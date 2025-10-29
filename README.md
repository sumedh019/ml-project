# Binary Classification with a Pulsar Dataset

A machine learning project that implements binary classification to identify pulsar stars using synthetic data from the [Kaggle Playground Series: Pulsar Classification](https://www.kaggle.com/competitions/playground-series-s3e10).

## Overview

This project demonstrates the complete machine learning pipeline for pulsar star classification, including:
- **Data Exploration & Preprocessing**: Comprehensive EDA with outlier detection and handling
- **Feature Engineering**: IQR-based outlier capping and SMOTE for class imbalance
- **Model Training**: Multiple algorithms (Random Forest, XGBoost, Logistic Regression)
- **Model Optimization**: Grid search for hyperparameter tuning
- **Web Deployment**: Flask API and Streamlit web interface

##  Dataset

The dataset contains synthetic pulsar signal data with the following features:
- `Mean_Integrated`: Mean of integrated profile
- `SD`: Standard deviation of integrated profile  
- `EK`: Excess kurtosis of integrated profile
- `Skewness`: Skewness of integrated profile
- `Mean_DMSNR_Curve`: Mean of DM-SNR curve
- `SD_DMSNR_Curve`: Standard deviation of DM-SNR curve
- `EK_DMSNR_Curve`: Excess kurtosis of DM-SNR curve
- `Skewness_DMSNR_Curve`: Skewness of DM-SNR curve
- `Class`: Target variable (0: Not a pulsar, 1: Pulsar)

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/xlr8-git/Binary-Classification-with-a-PulsarDataset.git
   cd Binary-Classification-with-a-PulsarDataset
   python -m venv venv
   # Activate virtual environment
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit web app**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Or run the Flask API**
   ```bash
   python app.py
   ```

## Project Structure

```
Binary-Classification-with-a-PulsarDataset/
├── app.py                                    # Flask API server
├── streamlit_app.py                         # Streamlit web interface
├── binary-classification-with-a-pulsardataset.ipynb  # Complete ML pipeline notebook
├── model.pkl                                # Trained Random Forest model
├── train.csv                                # Training dataset
├── requirements.txt                         # Python dependencies
├── README.md                               # Project documentation
└── LICENSE                                  # MIT License
```

##  Methodology

### Data Preprocessing
- **Outlier Detection**: IQR-based outlier identification and capping
- **Class Imbalance**: SMOTE (Synthetic Minority Oversampling Technique) for balanced training
- **Feature Scaling**: StandardScaler for algorithms requiring normalization

### Model Selection
Three algorithms were evaluated:
1. **Random Forest**: Best overall performance (98.6% accuracy)
2. **XGBoost**: Strong performance with gradient boosting (98.4% accuracy)  
3. **Logistic Regression**: Baseline model with feature scaling (91.4% accuracy)

### Hyperparameter Optimization
Grid search was performed on Random Forest with optimal parameters:
- `n_estimators`: 25
- `max_depth`: None
- `min_samples_split`: 2
- `min_samples_leaf`: 1
- `max_features`: 'sqrt'

##  Results

The optimized Random Forest model achieved:
- **Accuracy**: 98.6%
- **Precision**: 99% (Class 0), 90% (Class 1)
- **Recall**: 99% (Class 0), 96% (Class 1)
- **F1-Score**: 99% (Class 0), 93% (Class 1)

##  Web Interface

### Streamlit App
Interactive web interface for real-time predictions:
- Input form for all 8 features
- Real-time prediction display
- User-friendly interface

### Flask API
RESTful API for programmatic access:
- **Endpoint**: `POST /predict`
- **Input**: JSON with feature values
- **Output**: JSON with prediction (0 or 1)

Example API usage:
```python
import requests

data = {
    'Mean_Integrated': 120.5,
    'SD': 45.2,
    'EK': 0.15,
    'Skewness': 0.3,
    'Mean_DMSNR_Curve': 2.8,
    'SD_DMSNR_Curve': 18.5,
    'EK_DMSNR_Curve': 8.9,
    'Skewness_DMSNR_Curve': 95.2
}

response = requests.post("http://127.0.0.1:5000/predict", json=data)
prediction = response.json()['prediction']
```

## Development

### Running the Jupyter Notebook
```bash
jupyter notebook binary-classification-with-a-pulsardataset.ipynb
```

The notebook contains the complete analysis including:
- Exploratory Data Analysis (EDA)
- Data visualization and statistical analysis
- Outlier detection and treatment
- Model training and evaluation
- Hyperparameter optimization
- Model persistence

### Retraining the Model
To retrain the model with new data:
1. Update `train.csv` with new data
2. Run the notebook cells for model training
3. The new model will be saved as `model.pkl`

##  Requirements

See `requirements.txt` for complete dependency list. Key packages include:
- **Data Science**: pandas, numpy, scikit-learn
- **Machine Learning**: xgboost, imbalanced-learn
- **Visualization**: matplotlib, seaborn
- **Web Frameworks**: flask, streamlit
- **Utilities**: joblib, requests

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- [Kaggle Playground Series: Pulsar Classification](https://www.kaggle.com/competitions/playground-series-s3e10) for the dataset
- The open-source machine learning community for the excellent libraries used in this project

##  Contact

For questions or suggestions, please open an issue on GitHub or contact the maintainer.

---

⭐ If you found this project helpful, please give it a star!
