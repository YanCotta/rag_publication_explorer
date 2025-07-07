#!/bin/bash
echo "Creating virtual environment..."
if command -v python3 &> /dev/null; then
    python3 -m venv venv
elif command -v python &> /dev/null; then
    python -m venv venv
else
    echo "Error: Python not found. Please install Python 3.7 or higher."
    exit 1
fi

echo ""
echo "Virtual environment created successfully!"
echo ""
echo "To activate the environment, run:"
echo "source venv/bin/activate"
echo ""
echo "Then install dependencies with:"
echo "pip install -r requirements/requirements.txt"
