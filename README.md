# NYC Taxi Ride Volume: Anomaly Detection & Forecasting

This project uses LSTM-based deep learning models to analyze hourly NYC Yellow Taxi ride volume data from January to March 2025.

We built two time series modeling pipelines:
- **Unsupervised anomaly detection** using an LSTM autoencoder
- **Short-term forecasting** (1–6 hours ahead) using a sequence-to-sequence LSTM


## Data

- NYC Yellow Taxi data was downloaded directly from the NYC Taxi & Limousine Commission.
- We used `.parquet` files for January–March 2025.

## 🔍 1. Anomaly Detection (Notebook 1)

- Preprocesses hourly ride volume
- Trains an LSTM autoencoder to reconstruct normal traffic
- Computes reconstruction errors
- Flags and visualizes potential anomalies

## 📈 2. Forecasting (Notebook 2)

- Uses 24-hour windows to predict the next 6 hours
- Trains a sequence-to-sequence LSTM model
- Evaluates predictions using MSE and MAE
- Plots predicted vs. actual ride volume over time



