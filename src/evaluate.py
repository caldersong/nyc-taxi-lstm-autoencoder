import torch 
import numpy as np 
import matplotlib.pyplot as plt 

def get_reconstruction_errors(model, X, device="cpu"):
    """
    Compute reconstruction erros for each sequence in X
    """
    model.eval()
    model = model.to(device)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(X_tensor)
        errors = torch.mean((output - X_tensor) ** 2, dim=(1, 2)) # MSE per sequence
    return errors.cpu().numpy()

def detect_anomalies(errors, threshold=None, quantile=0.95):
    """
    Return a binary mask of which sequences are anomalies
    """
    if threshold is None:
        threshold = np.quantile(errors, quantile)
    anomaly_mask = errors > threshold 
    return anomaly_mask, threshold

def plot_anomalies(timestamps, values, anomaly_mask, title="Anomaly Detection"):
    """
    Plot the ride volume with anomalies highlighted.
    """
   
    plt.figure(figsize=(15, 5))
    plt.plot(timestamps, values, label="Ride Count")
    plt.scatter(
        timestamps[anomaly_mask],
        values[anomaly_mask],
        color="red",
        label="Anomaly",
        marker="x"
    )
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Hourly Ride Count")
    plt.legend()
    plt.tight_layout()
    plt.show()