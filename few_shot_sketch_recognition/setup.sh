#!/bin/bash
# Setup script for Few-Shot Sketch Recognition Framework
# This script creates a virtual environment and installs dependencies

set -e  # Exit on error

echo "=========================================="
echo "Few-Shot Sketch Recognition Framework"
echo "Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo ""
echo "Step 1: Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
    echo "To recreate it, delete the 'venv' folder first."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

echo ""
echo "Step 2: Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

echo ""
echo "Step 3: Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Step 4: Installing dependencies..."
pip install -r requirements.txt
echo "✓ All dependencies installed"

echo ""
echo "=========================================="
echo "Setup Complete! ✅"
echo "=========================================="
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To get started, try:"
echo "  python main.py --help"
echo ""
echo "Or follow the QUICKSTART.md guide!"
echo ""

