@echo off
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create virtual environment. Trying python3...
    python3 -m venv venv
)

echo.
echo Virtual environment created successfully!
echo.
echo To activate the environment, run:
echo venv\Scripts\activate
echo.
echo Then install dependencies with:
echo pip install -r requirements\requirements.txt
