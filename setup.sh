#!/bin/bash

# Emotion Detection System Setup Script
# This script sets up the environment and installs dependencies

echo "🎭 Emotion Detection System Setup"
echo "================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "📋 Python version: $python_version"

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models data docs/results tests

# Create placeholder files
echo "📄 Creating placeholder files..."
touch models/.gitkeep
touch data/.gitkeep
touch tests/__init__.py

# Set permissions
echo "🔒 Setting permissions..."
chmod +x setup.sh

# Check installation
echo "✅ Checking installation..."
python -c "
import streamlit as st
import cv2
import sklearn
import pandas as pd
import numpy as np
print('✅ All dependencies installed successfully!')
print('Streamlit version:', st.__version__)
print('OpenCV version:', cv2.__version__)
print('Scikit-learn version:', sklearn.__version__)
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Download FER2013 dataset (optional, for training)"
echo "3. Run the app: streamlit run emotion_app.py"
echo ""
echo "📖 For more information, see README.md"
