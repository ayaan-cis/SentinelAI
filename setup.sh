#!/bin/bash
# Setup script for SentinelAI project

# Create required directories
mkdir -p data
mkdir -p models

# Install required packages
pip install -r requirements.txt

# Generate sample data
echo "Generating sample network flow data..."
python generate_mock_data.py

# Run the application
echo "Starting SentinelAI application..."
streamlit run app.py