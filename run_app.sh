#!/bin/bash

# Stock AI Assistant Streamlit Runner
# This script sets up the proper Python path and runs the streamlit app

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Set up Python path
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$PARENT_DIR")"

export PYTHONPATH="$PARENT_DIR:$PROJECT_ROOT:$PYTHONPATH"

echo "Starting Stock AI Assistant..."
echo "Setting PYTHONPATH to include:"
echo "  - $PARENT_DIR"
echo "  - $PROJECT_ROOT"
echo ""

# Change to the web directory and run streamlit
cd "$SCRIPT_DIR"
streamlit run streamlit_app.py

echo "Stock AI Assistant stopped."
