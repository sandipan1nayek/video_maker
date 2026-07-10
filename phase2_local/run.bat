@echo off
setlocal

:: Check if venv folder exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Run the Streamlit app
echo Starting Streamlit app...
streamlit run app.py

endlocal
