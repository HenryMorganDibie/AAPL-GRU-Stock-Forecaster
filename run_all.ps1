# run_all.ps1
# This script executes the entire AAPL forecasting pipeline in sequence.
# NOTE: ASSUMES THE VIRTUAL ENVIRONMENT (.venv) IS ALREADY ACTIVE.

# --- Configuration ---
$PROJECT_ROOT = (Get-Item $PSScriptRoot).FullName
# Since environment is active, we just need the 'python' command
# The shell will resolve it to .venv/Scripts/python.exe
$PYTHON_EXE = "python"
$DATA_PIPELINE = "$PROJECT_ROOT\src\data_pipeline.py"
$FORECAST_MODEL = "$PROJECT_ROOT\src\forecasting_model.py"

Write-Host "--- Starting AAPL Forecasting Pipeline ---" -ForegroundColor Green

# 1. Skip Virtual Environment Activation (Already Active)
Write-Host "1. Virtual environment confirmed active. Skipping explicit activation." -ForegroundColor Yellow

# --- 2. Run Data Pipeline (Fetches, Cleans, and Saves Processed Data) ---
Write-Host "`n2. Executing data pipeline (src\data_pipeline.py) - OUTPUT BELOW:" -ForegroundColor Yellow
# Run the script and allow all output (stdout/stderr) to display immediately
& $PYTHON_EXE $DATA_PIPELINE *>&1

if ($LASTEXITCODE -ne 0) {
    Write-Error "Data pipeline failed with exit code $LASTEXITCODE. Aborting."
    exit $LASTEXITCODE
}

# --- 3. Run Forecasting Model (Trains GRU, Predicts, Saves Model) ---
Write-Host "`n3. Executing forecasting model (src\forecasting_model.py) - OUTPUT BELOW:" -ForegroundColor Yellow
# Run the script and allow all output (stdout/stderr) to display immediately
& $PYTHON_EXE $FORECAST_MODEL *>&1

if ($LASTEXITCODE -ne 0) {
    Write-Error "Forecasting model failed with exit code $LASTEXITCODE. Aborting."
    exit $LASTEXITCODE
}

# 4. Success Message and Next Steps
Write-Host "`n--- Pipeline Complete! ---" -ForegroundColor Green
Write-Host "Processed data and the final GRU model have been saved to the 'data/processed' and 'models' directories." -ForegroundColor Cyan
Write-Host "The final MAPE metrics should be visible in the output above." -ForegroundColor Cyan
Write-Host "Next, open the 'notebooks/02_Model_Training_Evaluation.ipynb' notebook to view the final report and visualization." -ForegroundColor Cyan

# Skip Deactivation (Best done manually outside the script)
Write-Host "----------------------------------------" -ForegroundColor Green