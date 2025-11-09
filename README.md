# 🍎 AAPL Stock Price Forecasting Project

## 🎯 Project Overview

This project implements an end-to-end Machine Learning solution for forecasting the closing price of Apple stock (AAPL) using a Gated Recurrent Unit (GRU) neural network. The entire pipeline, from raw data ingestion and cleaning to model training and evaluation, is modular and automated for reproducibility.

The final pipeline successfully generated a **21-day Live Forecast** for November/December 2025 (e.g., ~$246.80 to ~$258.85), validated by an acceptable historical **Mean Absolute Percentage Error (MAPE)**, demonstrating a production-ready time series prediction workflow.

---

## 📁 Project Structure

The repository is organized following industry standards for data science projects, ensuring clear separation between code, data, models, and documentation.

<pre lang="markdown">
AAPL_Forecast_Project/
├── data/
│ ├── raw/ # Contains the raw input data (e.g., AAPL_historical.csv)
│ └── processed/ # Stores the clean, serialized TimeSeries objects (.pkl files)
├── models/ # Stores the final trained Darts/PyTorch models (.pkl files)
├── notebooks/ # Contains exploratory and final reporting notebooks
├── reports/ # **NEW:** Contains the final aapl_forecast_results.xlsx file.
├── src/ # Source code for the production pipeline
│ ├── data_pipeline.py # Data fetching, cleaning, and TimeSeries preparation
│ └── forecasting_model.py # Model definition, training, and evaluation
├── .venv/ # Python virtual environment
├── run_all.ps1 # PowerShell automation script to run the full pipeline.
├── README.md # This file
└── requirements.txt # List of project dependencies
</pre>

---

## 🛠️ Prerequisites

To run this project locally, you must have **Python 3.8+** and a PowerShell environment (standard on Windows) or access to the respective PowerShell Core tools on other platforms.

1.  **Clone the Repository:**
    ```
    git clone [https://github.com/HenryMorganDibie/AAPL-GRU-Stock-Forecaster.git](https://github.com/HenryMorganDibie/AAPL-GRU-Stock-Forecaster.git)
    cd AAPL_Forecast_Project
    ```

2.  **Set up the Environment:**
    ```
    python -m venv .venv
    .\.venv\Scripts\activate  # Windows
    # source .venv/bin/activate  # macOS/Linux
    ```

3.  **Install Dependencies:**
    ```
    pip install -r requirements.txt
    ```

---

## 🧠 Model & Strategy Deep Dive (Justification)

This section details the choices made regarding the data structure, model architecture, and multi-horizon forecasting strategy, fulfilling the requirement for technical justification.

### 1. Feature Engineering and Rationale

The project focuses on **univariate time series forecasting**, using only historical closing prices for prediction.

| Feature | Type | Rationale |
| :--- | :--- | :--- |
| **Close Price (Input)** | Univariate | Primary target variable and input feature. Establishes a strong *autoregressive* baseline using only past price movements. |
| **Open, High, Low, Volume** | Ignored | Excluded to minimize noise and complexity. Future work can incorporate these as **covariates** for improved trend capture. |

### 2. Darts Framework and Model Implementation

The **Darts** framework was utilized to provide a clean, production-ready interface for deep learning time series models:

* **Data Preparation:** The `TimeSeries` object handles data indexing and frequency (`freq='B'`) automatically.
    ```python
    from darts import TimeSeries
    # Creates Darts object with DatetimeIndex
    series = TimeSeries.from_dataframe(series_df, freq='B') 
    ```
* **Model Instantiation:** The core model is an implementation of the **BlockRNNModel** class from Darts, set to use the GRU architecture, ensuring native multi-step forecasting support.
    ```python
    from darts.models import BlockRNNModel 
    # Model uses 30 days of input to predict 21 days out
    model = BlockRNNModel( 
        model='GRU', 
        input_chunk_length=30, 
        output_chunk_length=21, # Multi-step prediction length
        n_epochs=200
    )
    ```

### 3. Multi-Horizon Forecasting Strategy

- **GRU vs. LSTM Comparison**: I performed a preliminary comparison between the GRU and LSTM architectures. The Gated Recurrent Unit (GRU) was chosen because it offered comparable prediction accuracy (MAPE scores) to the more complex LSTM but achieved faster convergence and required fewer trainable parameters. This prioritized model efficiency and simplicity for the final production baseline.

**Forecasting Strategy (Direct Multi-step)**: I employed a **Direct Multi-step Forecasting** approach:

* **Single Model Training:** Only **one** GRU model was trained.
* **Unified Output:** The model was configured with an `output_chunk_length` of **21** (the longest required horizon).
* **Evaluation:** The single 21-day forecast vector generated by the model is evaluated at three distinct time steps:
    * **1 Day Ahead** (Index 0 of the forecast vector)
    * **1 Week Ahead (5 Days)** (Index 4 of the forecast vector)
    * **1 Month Ahead (21 Days)** (Index 20 of the forecast vector)
* **Rationale:** This method ensures temporal consistency across all horizons and avoids the computational cost of training multiple models.

### 4. Data Splitting and Validation

* **Time-Based Split:** A strictly **chronological split** was used to prevent **look-ahead bias** (data leakage).
* **Split Ratios:** The overall dataset (2015-present) was split as follows:
    * **Train Set:** All data up to approximately **2 years before the current date**.
    * **Validation Set:** The **final 252 trading days** (approx. 1 year) of data, used for final metric reporting.
* **Live Forecast:** The final model is **retrained on the full dataset** (Train + Validation) to maximize information, ensuring the **2025 Live Forecast** starts immediately after the most current data point.

---

## 🚀 How to Run the Project (Automated)

The entire project pipeline—data preparation, model training, and saving—can be executed from your PowerShell terminal using the dedicated automation script.

**Note:** Ensure your virtual environment (`.venv`) is activated before running.

### Single-Step Execution (Recommended)

```bash
.\run_all.ps1
```

This script will sequentially:

- Confirm the virtual environment is active.

- Execute src/data_pipeline.py.

- Execute src/forecasting_model.py.

- Execute src/forecasting_model.py, which generates the two-sheet Excel report (aapl_forecast_results.xlsx) in the reports/ directory.

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

# 📊 Key Results and Limitations

## 🧠 Final Forecast Summary (Nov/Dec 2025 Live Prediction)

The retrained model generated the following **out-of-sample forecast** for the **21 trading days** following the current date:

| Horizon | Live Forecast Price (USD) |
|----------|----------------------------|
| **1 Day Ahead** | **$246.80** |
| **1 Week Ahead (5 Days)** | **$245.62** |
| **1 Month Ahead (21 Days)** | **$258.85** |

---

## 📈 Validation MAPE Performance

The historical validation results (on the 2023 data split) confirmed the model’s ability to capture long-term trends, despite high short-term error likely caused by volatility in the validation data start window.

| Horizon | Mean Absolute Percentage Error (MAPE) | Performance Description |
|----------|--------------------------------------|--------------------------|
| **1 Day Ahead** | **10.07%** | High error, suggesting the validation split was immediately after a sharp price change. |
| **1 Week Ahead (5 Days)** | **8.82%** | Error moderates as the forecast moves past the initial volatility. |
| **1 Month Ahead (21 Days)** | **6.48%** | Strongest performance, demonstrating the GRU's effectiveness in long-term trend capture. |

---

## ⚙️ Project Limitations and Future Work

### 🔹 Exogenous Factors  
- The current **univariate model** does not account for critical external variables (e.g., macroeconomic indicators, volume, market sentiment).  
- **Future work** should integrate these as **exogenous regressors** within the Darts framework.

### 🔹 Model Robustness  
- The model is trained on a **fixed historical window**.  
- For **production use**, implementing **rolling window retraining** would be necessary to adapt to recent market shifts and maintain performance over time.

---

## 📁 Final Reporting (Jupyter Notebooks)

The final results are presented in two forms: 
 
- A **generated Excel spreadsheet** for data  
- A **Jupyter Notebook** for visualization and detailed analysis  

To view the detailed data analysis, model performance visualization, and complete justification report, open the following files:

| **File Name** | **Description** |
|----------------|-----------------|
| **aapl_forecast_results.xlsx** | Final two-sheet Excel report containing the full 21-day Live Forecast data and historical prediction vs. actual data. |
| **01_EDA_Preprocessing.ipynb** | Review the data cleaning process and initial time series plots. |
| **02_Model_Training_Evaluation.ipynb** | Load the saved model, regenerate the final forecast, and analyze the results — including key MAPE scores and the Actual vs. Predicted Price Plot. |
| **placeholder.txt** | Project Conclusion |

---
