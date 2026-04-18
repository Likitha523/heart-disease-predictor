---
title: Heart Disease Predictor
emoji: ❤️
colorFrom: red
colorTo: pink
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# AI Heart Disease Prediction System

End-to-End Machine Learning web application for predicting Cardiovascular Disease using the `cardio_train.csv` dataset.

### Project Structure
- `train_model.py`: Script to download the dataset, preprocess it, perform SMOTE, train multiple models, tune Hyperparameters for XGBoost, and export the best model.
- `app.py`: Streamlit User Interface for prediction and SHAP explainability.
- `requirements.txt`: Python package dependencies.
- `heart_model.pkl` & `preprocessor.pkl`: Serialized model and scaler.
- `confusion_matrix.png`, `roc_curve.png`, `feature_importance.png`: Visual evaluation reports.

### Local Execution:
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
