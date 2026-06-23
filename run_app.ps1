$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Creating local virtual environment ..."
    python -m venv .venv
}

Write-Host "Installing dependencies from reserve_agent/requirements.txt ..."
.\.venv\Scripts\python.exe -m pip install -r reserve_agent/requirements.txt

Write-Host "Starting Streamlit app ..."
.\.venv\Scripts\streamlit.exe run reserve_agent/app.py
