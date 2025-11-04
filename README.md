# 🍎 AAPL Stock Price Forecasting Project

## 🎯 Project Overview

This project implements an end-to-end Machine Learning solution for forecasting the closing price of Apple stock (AAPL) using a Gated Recurrent Unit (GRU) neural network. The entire pipeline, from raw data ingestion and cleaning to model training and evaluation, is containerized and follows a strict modular structure for reproducibility.

The final model successfully achieved a **Mean Absolute Percentage Error (MAPE)** of **0.27%** for the 1-Day forecast, validating the deep learning approach for time series prediction.

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
    ```bash
    git clone [https://github.com/HenryMorganDibie/AAPL-GRU-Stock-Forecaster.git](https://github.com/HenryMorganDibie/AAPL-GRU-Stock-Forecaster.git)
    cd AAPL_Forecast_Project
    ```

2.  **Set up the Environment:**
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\activate  # Windows
    # source .venv/bin/activate  # macOS/Linux
    ```

3.  **Install Dependencies:**
    ```powershell
    pip install -r requirements.txt
    ```

---

## 🧠 Model & Strategy Deep Dive (Justification)

This section details the choices made regarding the data structure, model architecture, and multi-horizon forecasting strategy, fulfilling the requirement for technical justification.

### 1. Data Structure and Darts Integration

* **Feature Choice (Univariate):** The model is based solely on the historical **Close price**. This *univariate* approach was chosen to establish a strong performance baseline, minimizing complexity before integrating noisy technical or external indicators (exogenous variables).
* **Darts Framework:** We utilize the **Darts** framework primarily because it provides a clean, unified interface for PyTorch-based Time Series models. Key Darts classes used are:
    * `TimeSeries.from_dataframe()`: Ensures the data index is a correct `DatetimeIndex` with a business-day frequency (`freq='B'`).
    * `Scaler`: Manages data normalization and inverse transformation automatically, crucial for neural networks.
    * `RNNModel` (GRU variant): Simplifies the training loop, checkpointing, and evaluation compared to native PyTorch.

### 2. Model Architecture and Horizon Strategy

* **GRU vs. LSTM:** The **Gated Recurrent Unit (GRU)** architecture was chosen over the more complex Long Short-Term Memory (LSTM) due to its **efficiency and simplicity**. Preliminary testing indicated that the GRU provided comparable performance on this univariate data but achieved **faster convergence** with **fewer trainable parameters**, making it ideal for a deployable baseline.
* **Forecasting Strategy (Direct Multi-Step):** The model is designed to handle multiple horizons (1-day, 1-week, 1-month) simultaneously using a **Direct Multi-step Forecasting** approach.
    * **Input Chunk Length:** 30 business days.
    * **Output Chunk Length:** 21 business days (the length of the longest horizon).
    * The model is trained once to predict **21 future steps**. We then evaluate the performance at the 1st step (1-day), the 5th step (1-week), and the 21st step (1-month). This unified approach is computationally efficient and maintains temporal consistency.

### 3. Data Splitting

The project employs a strict time-series split to prevent **look-ahead leakage**:
* **Train Set:** Data from **2015-01-01** up to the start of the validation period.
* **Validation Set:** The **last 252 trading days** (approximately one year of trading data) is reserved for validation and final performance evaluation. The evaluation plots (in the notebook) compare predictions only on this unseen future data.

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

### 📊 Key Results and Limitations

#### Final MAPE Performance
The model's error naturally increases as the forecast horizon extends — this is expected due to the error accumulation inherent in time-series forecasting.

| Horizon               | Mean Absolute Percentage Error (MAPE) | Performance Description                                           |
|------------------------|----------------------------------------|-------------------------------------------------------------------|
| 1 Day Ahead            | 0.2726%                               | Exceptional accuracy for short-term trading.                      |
| 1 Week Ahead (5 Days)  | 1.0403%                               | Highly reliable over a business week.                             |
| 1 Month Ahead (21 Days)| 4.1789%                               | Strong, maintaining error under the 5% threshold, demonstrating long-term predictive strength. |

---

### Project Limitations and Future Work

- **Exogenous Factors:**  
  The current univariate model does not account for critical external variables (e.g., macroeconomic indicators, volume, market sentiment).  
  Future work should integrate these as exogenous regressors within the Darts framework.

- **Model Robustness:**  
  The model is trained on a fixed historical window.  
  For production use, implementing **rolling window retraining** would be necessary to adapt to recent market shifts and maintain performance over time.

---

## 📁 Final Reporting (Jupyter Notebooks)

To view the detailed data analysis, model performance visualization, and complete justification report, open the following Jupyter notebooks:

1. **`01_EDA_Preprocessing.ipynb`**  
   Review the data cleaning process and initial time series plots.

2. **`02_Model_Training_Evaluation.ipynb`**  
   Load the saved model, regenerate the final forecast, and analyze the results — including key MAPE scores and the Actual vs. Predicted Price Plot.

3. **`placeholder.txt`**
   Project Conclusion
