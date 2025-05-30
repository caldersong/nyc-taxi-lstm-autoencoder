import pandas as pd
import glob

def load_data(months=["01", "02", "03"], year="2025", data_path="/Users/chaebeensong/Documents/Projects/nyc-taxi-lstm-autoencoder/data/"):
    """
    Load and combine NYC Yellow Taxi data for specified months and year.
    Returns a DataFrame with hourly ride counts
    """

    files = [f"{data_path}yellow_tripdata_{year}-{month}.parquet" for month in months]
    dfs = [pd.read_parquet(file) for file in files]
    df = pd.concat(dfs, ignore_index=True)

    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df = df.dropna(subset=["tpep_pickup_datetime"])
    df = df[(df["tpep_pickup_datetime"] >= "2025-01-01") & (df["tpep_pickup_datetime"] <= "2025-03-31")]

    df.set_index("tpep_pickup_datetime", inplace=True)
    df = df.sort_index()

    ride_counts = df.resample("1H").size().to_frame(name="ride_count")
    return ride_counts 