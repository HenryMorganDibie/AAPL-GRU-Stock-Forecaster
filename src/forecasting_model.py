import pandas as pd
from darts.models import RNNModel
from darts.metrics import mape
from sklearn.preprocessing import StandardScaler 
from pathlib import Path
import pickle
import torch
import numpy as np
from darts import TimeSeries 


# --- Configuration ---
TICKER = 'AAPL'
# Prediction horizons (in trading days)
HORIZON_1D = 1
HORIZON_1W = 5    # ~1 week of trading days
HORIZON_1M = 21   # ~1 month of trading days

# --- Model Hyperparameters ---
INPUT_CHUNK = 30
OUTPUT_CHUNK = HORIZON_1M
HIDDEN_DIM = 50    
N_EPOCHS = 200     
BATCH_SIZE = 64     
DROPOUT = 0.0       
TRAINING_LEN = 40  
LEARNING_RATE = 1e-4 


# Define paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SERIES_PATH = PROJECT_ROOT / 'data' / 'processed' / f'{TICKER}_TS_train.pkl'
VAL_SERIES_PATH = PROJECT_ROOT / 'data' / 'processed' / f'{TICKER}_TS_val.pkl'
MODEL_PATH = PROJECT_ROOT / 'models' / f'darts_gru_{TICKER}_v1.pkl'


def load_data(train_path: Path, val_path: Path):
    """Loads the processed Darts TimeSeries objects."""
    try:
        with open(train_path, 'rb') as f:
            train_series = pickle.load(f)
        with open(val_path, 'rb') as f:
            val_series = pickle.load(f)
        return train_series, val_series
    except FileNotFoundError:
        print("!!! ERROR: Processed data not found. Run src/data_pipeline.py first.")
        return None, None

def train_and_forecast():
    """Trains the GRU model, makes predictions, and calculates MAPE."""
    train_series, val_series = load_data(TRAIN_SERIES_PATH, VAL_SERIES_PATH)
    
    if train_series is None or val_series is None:
        return
    
    # --- DATA CLEANING (Pandas fillna with method='...') ---
    print("-> Cleaning data: filling NaN/Inf values using direct Pandas methods...")
    
    # Using the working Pandas method for data cleaning
    train_series = TimeSeries.from_series(
        train_series.to_series().fillna(method='ffill').fillna(method='bfill')
    )
    val_series = TimeSeries.from_series(
        val_series.to_series().fillna(method='ffill').fillna(method='bfill')
    )
    
    # Check for remaining NaNs/Infs after cleaning (should be none)
    if np.any(np.isnan(train_series.values())) or np.any(np.isinf(train_series.values())):
        print("!!! FATAL ERROR: Data is entirely NaN or Inf even after cleaning. Check raw data source.")
        return
    # ---------------------------------------------

    # 1. CRITICAL FIX: DIRECT SCALING (With Pandas Intermediary)
    print("-> Scaling data using Scikit-learn StandardScaler directly...")
    
    # Convert TimeSeries to NumPy array and reshape to (N, 1) column vector
    # This shape is required for Scikit-learn fit/transform
    train_values = train_series.values().reshape(-1, 1)
    val_values = val_series.values().reshape(-1, 1)
    
    # Initialize and fit the Scikit-learn scaler
    scaler = StandardScaler() 
    scaler.fit(train_values)
    
    # Transform the values
    train_scaled_values = scaler.transform(train_values)
    val_scaled_values = scaler.transform(val_values)
    
    # Convert scaled NumPy arrays to Pandas Series (flattening back to 1D)
    # CRITICAL FIX: Use the Pandas Series intermediary to reliably create TimeSeries
    train_scaled_series_pd = pd.Series(train_scaled_values.flatten(), index=train_series.time_index)
    val_scaled_series_pd = pd.Series(val_scaled_values.flatten(), index=val_series.time_index)
    
    # Convert Pandas Series back to Darts TimeSeries
    train_scaled = TimeSeries.from_series(train_scaled_series_pd)
    val_scaled = TimeSeries.from_series(val_scaled_series_pd)
    
    # --- POST-SCALING SANITY CHECK ---
    train_vals = train_scaled.values().flatten()
    print(f"-> Scaled Data Mean: {np.mean(train_vals):.4f}, Std: {np.std(train_vals):.4f}")
    
    if np.any(np.isnan(train_vals)) or np.any(np.isinf(train_vals)):
        print("!!! ERROR: NaN or Inf values found in the SCALED data. Scaling failed.")
        return
    # ----------------------------------

    # 2. MODEL DEFINITION (GRU)
    print(f"-> Initializing RNNModel (GRU) with Input={INPUT_CHUNK}, Output={OUTPUT_CHUNK}, Epochs={N_EPOCHS}")
    
    model = RNNModel(
        model='GRU',
        input_chunk_length=INPUT_CHUNK,
        output_chunk_length=OUTPUT_CHUNK,
        hidden_dim=HIDDEN_DIM,
        n_rnn_layers=1,              
        dropout=DROPOUT, 
        batch_size=BATCH_SIZE, 
        n_epochs=N_EPOCHS,
        training_length=TRAINING_LEN,
        pl_trainer_kwargs={"accelerator": "cpu", "gradient_clip_val": 1.0},
        optimizer_kwargs={"lr": LEARNING_RATE, "eps": 1e-8}, 
        force_reset=True,
        random_state=42
    )

    # 3. TRAINING
    print("-> Starting model training...")
    # This should now execute and begin training
    model.fit(
        train_scaled,
        val_series=val_scaled,  
        verbose=True
    )
    print("-> Training complete.")
    
    # Save the trained model
    model.save(str(MODEL_PATH))
    print(f"-> Model saved to: {MODEL_PATH}")

    # 4. PREDICTION 
    # Prediction returns a TimeSeries object (N, 1)
    forecast_scaled_1m = model.predict(n=HORIZON_1M)
    
    # 5. INVERSE SCALING 
    # Get values (which are (N, 1)) and inverse scale
    forecast_values_1m = scaler.inverse_transform(forecast_scaled_1m.values())
    
    # Convert back to Darts TimeSeries via Pandas Series
    forecast_series_pd = pd.Series(forecast_values_1m.flatten(), index=forecast_scaled_1m.time_index)
    forecast_1m = TimeSeries.from_series(forecast_series_pd)
    
    actual_1m = val_series[:HORIZON_1M]
    
    # Extract shorter horizons
    forecast_1d = forecast_1m[:HORIZON_1D]
    forecast_1w = forecast_1m[:HORIZON_1W]
    actual_1d = val_series[:HORIZON_1D]
    actual_1w = val_series[:HORIZON_1W]
    
    # 6. EVALUATION (MAPE %)
    print("\n--- Model Evaluation (MAPE %) ---")
    mape_1d = mape(actual_1d, forecast_1d)
    mape_1w = mape(actual_1w, forecast_1w)
    mape_1m = mape(actual_1m, forecast_1m)
    
    print(f"MAPE (1 Day Ahead): {mape_1d:.4f}%")
    print(f"MAPE (1 Week Ahead): {mape_1w:.4f}%")
    print(f"MAPE (1 Month Ahead): {mape_1m:.4f}%")
    
    # Get the predicted closing prices for the report
    print("\n--- Predicted Closing Prices (from validation set start) ---")
    print(f"1-Day Forecast: {forecast_1d.values().flatten()[0]:.2f}")
    print(f"1-Week Forecast (Day 5): {forecast_1w.values().flatten()[-1]:.2f}")
    print(f"1-Month Forecast (Day 21): {forecast_1m.values().flatten()[-1]:.2f}")

    return {
        "mape_1d": mape_1d,
        "mape_1w": mape_1w,
        "mape_1m": mape_1m,
    }


if __name__ == "__main__":
    train_and_forecast()