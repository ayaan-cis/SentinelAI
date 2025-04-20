#!/bin/bash
# Setup script for SentinelAI project

# Create required directories
mkdir -p data
mkdir -p models

# Install required packages
# pip install -r requirements.txt
pip3 install -r requirements.txt  # For MacOS users, just uncomment

# Generate sample data
echo "Generating sample network flow data..."
# python generate_mock_data.py
python3 generate_mock_data.py  # For MacOS users, just uncomment

# Run the application
echo "Starting SentinelAI application..."
# streamlit run app.py
python3 -m streamlit run app.py # For MacOS users, just uncomment