 Advanced Time Series Forecasting with Deep Learning & Explainability

This project implements an end-to-end advanced time series forecasting pipeline using deep learning, Bayesian hyperparameter optimization, baseline statistical models, and model explainability techniques. The focus is not only on prediction accuracy, but also on robust evaluation and interpretability for multivariate, non-stationary time series data.

🚀 Project Objectives

Generate a multivariate, non-stationary synthetic time series dataset

Train a sequence-to-sequence LSTM model for multi-step forecasting

Apply rolling-window style evaluation and robust performance metrics

Optimize hyperparameters using Optuna (Bayesian Optimization)

Compare deep learning performance against a baseline ARIMA model

Interpret model predictions using SHAP explainability

🧠 Key Features

✔ 1200+ observations with 5 input features

✔ Non-stationary trend, seasonality, and volatility

✔ Multi-horizon forecasting (5-step ahead)

✔ LSTM-based deep learning model (PyTorch)

✔ Optuna-based hyperparameter tuning

✔ ARIMA baseline comparison

✔ SHAP explainability adapted for time-series models

✔ RMSE & MAE evaluation metrics

📂 Project Structure
├── advanced_time_series_forecasting.py
├── README.md

🧪 Dataset Description

The dataset is synthetically generated to simulate real-world time series characteristics:

Feature	Description
trend	Linear non-stationary trend
seasonality	Sinusoidal seasonal component
volatility	Increasing variance over time
feature4	Cosine-based periodic signal
feature5	Random noise feature
target	Weighted combination of features + noise

Total observations: 1200
Sequence length: 30 time steps
Forecast horizon: 5 future steps

🏗️ Model Architecture
🔹 Deep Learning Model

Model: LSTM (Sequence-to-Sequence)

Framework: PyTorch

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Output: Multi-step forecast (5 steps)

🔹 Baseline Model

ARIMA (2,1,2) using statsmodels

⚙️ Hyperparameter Optimization

Hyperparameters are tuned using Optuna, including:

Number of LSTM hidden units

Number of LSTM layers

Learning rate

Optimization Objective:
Minimize RMSE on the test set

📊 Evaluation Metrics

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

Performance is reported for:

Optimized LSTM model

Baseline ARIMA model

🔍 Explainability with SHAP

Since LSTM models require 3D input (samples × time × features) and SHAP supports 2D inputs, the time dimension is flattened for explainability.

Explainability Strategy:

Flatten time-series input for SHAP compatibility

Reshape data back to 3D inside the prediction wrapper

Use SHAP KernelExplainer to compute feature-time importance

This approach provides insight into which features and time steps most influence predictions.

📦 Installation

Install the required dependencies:

pip install numpy pandas torch scikit-learn optuna statsmodels shap

▶️ How to Run
python advanced_time_series_forecasting.py

📌 Sample Output
Deep Learning RMSE: 0.0XXX
Deep Learning MAE: 0.0XXX
ARIMA RMSE: 0.0XXX
SHAP explainability completed successfully

🎓 Academic Relevance

This project aligns with advanced coursework in:

Time Series Analysis

Deep Learning

Model Explainability (XAI)

Bayesian Optimization

It is suitable for:

University assignments

Capstone projects

Data science portfolios

Research prototypes

📜 License

This project is for educational and academic use.

🙌 Acknowledgements

PyTorch

Optuna

SHAP

Statsmodels

Scikit-learn
# Final-Project
