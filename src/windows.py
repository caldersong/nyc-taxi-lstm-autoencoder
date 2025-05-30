import numpy as np 

def create_sliding_windows(series, window_size):
    """
    Convert 1D array into overlapping sequences (aka windows) of fixed length.
    """
    X = []
    for i in range(len(series) - window_size):
        window = series[i : i + window_size]
        X.append(window)
    return np.array(X)

def create_forecasting_windows(series, input_size, forecast_horizon):
    """
    Create sliding windows for sequence-to-sequence forecasting.
    
    Parameters:
        series: 1D NumPy array
        input_size: number of past timesteps 
        forecast_horizon: number of steps to predict into the future

    Returns:
        X: input sequences (num_samples, input_size, 1)
        y: future sequences (num_samples, forecast_horizon, 1)
    """
    X, y = [], []
    total_length = input_size + forecast_horizon
    for i in range(len(series) - total_length):
        x_i = series[i : i + input_size]
        y_i = series[i + input_size : i + total_length]
        X.append(x_i)
        y.append(y_i)
    X = np.array(X).reshape(-1, input_size, 1)
    y = np.array(y).reshape(-1, forecast_horizon, 1)
    return X, y