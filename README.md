# 🍎 AAPL Stock Price Forecasting Project

## 🎯 Project Overview

This project implements an end-to-end Machine Learning solution for forecasting the closing price of Apple stock (AAPL) using a Gated Recurrent Unit (GRU) neural network. The entire pipeline, from raw data ingestion and cleaning to model training and evaluation, is containerized and follows a strict modular structure for reproducibility.

The final model successfully achieved a **Mean Absolute Percentage Error (MAPE)** of **0.27%** for the 1-Day forecast, validating the deep learning approach for time series prediction.

## 📁 Project Structure

The repository is organized following industry standards for data science projects, ensuring clear separation between code, data, models, and documentation.

<pre lang="markdown">
AAPL_Forecast_Project/
├── data/
│   ├── raw/                     # Contains the raw input data (e.g., AAPL_historical.csv)
│   └── processed/               # Stores the clean, serialized TimeSeries objects (.pkl files)
├── models/                      # Stores the final trained Darts/PyTorch models (.pkl files)
├── notebooks/                   # Contains exploratory and final reporting notebooks
├── src/                         # Source code for the production pipeline
│   ├── _init_.py                      
│   ├── data_pipeline.py         # Data fetching, cleaning, and TimeSeries preparation
│   └── forecasting_model.py     # Model definition, training, and evaluation
├── run_all.ps1                  # PowerShell script to execute the full forecasting pipeline
├── README.md                    # This file
└── requirements.txt             # List of project dependencies

</pre>

## 🛠️ Prerequisites

To run this project locally, you must have Python 3.8+ installed.

1.  **Clone the Repository:**
    ```bash
    git clone [YOUR_REPO_URL]
    cd AAPL_Forecast_Project
    ```

2.  **Set up the Environment:**
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate  # Windows
    # source .venv/bin/activate  # macOS/Linux
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 How to Run the Project

The entire project pipeline—data preparation, model training, and saving—can be executed from your PowerShell terminal using the dedicated automation script.

Note: Ensure your virtual environment (.venv) is activated before running.

### Single-Step Execution (Recommended)

```bash
.\run_all.ps1
```
This script will sequentially:

- Confirm the virtual environment is active.

- Execute src/data_pipeline.py.

- Execute src/forecasting_model.py.

- Output the final MAPE metrics to the console.


**Manual Execution (For Debugging or Development)**

**For development or debugging purposes, you can run the steps individually:**

### Step 1: Data Pipeline Execution

This script fetches the latest AAPL data, cleans it, and splits it into training and validation TimeSeries objects, saving them to `data/processed/`.

```bash
python src/data_pipeline.py
```

### Step 2: Model Training and Saving
This script loads the processed data, scales it, trains the optimized GRU model, performs inverse scaling, and calculates the final MAPE metrics before saving the model artifact to models/.

```bash
python src/forecasting_model.py
```

### Step 3: Final Reporting (Jupyter Notebooks)To view the detailed data analysis and model performance visualization, open the Jupyter notebooks:

- 01_EDA_Preprocessing.ipynb: Review the data cleaning process and initial time series plots

- 02_Model_Training_Evaluation.ipynb: Load the saved model, regenerate the final forecast, and analyze the results, including the key MAPE scores and the Actual vs. Predicted Price Plot.

### 📊 Key Results

The final GRU model provided a highly accurate forecast across short and medium horizons (evaluated against the last two years of data):

| Horizon               | Mean Absolute Percentage Error (MAPE) |
|------------------------|----------------------------------------|
| 1 Day Ahead            | 0.2726%                               |
| 1 Week Ahead (5 Days)  | 1.0403%                               |
| 1 Month Ahead (21 Days)| 4.1789%                               |
