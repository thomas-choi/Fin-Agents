#!/bin/bash

# Path to your Python script
PYTHON_SCRIPT="/home/thomas/projects/Fin-Agents/POC/MA4Star_Strategy.py"

# Path to the CSV file containing tickers
TICKERS_CSV="hsi_components.csv"
# TICKERS_CSV="test.csv"

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Python3 is not installed. Please install it first."
    exit 1
fi

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Python script not found at: $PYTHON_SCRIPT"
    exit 1
fi

# Check if the CSV file exists
if [ ! -f "$TICKERS_CSV" ]; then
    echo "Tickers CSV file not found at: $TICKERS_CSV"
    exit 1
fi

# Read tickers from CSV file (assuming tickers are in the first column)
TICKERS=($(cut -d',' -f1 "$TICKERS_CSV"))

# Check if any tickers were read
if [ ${#TICKERS[@]} -eq 0 ]; then
    echo "No tickers found in $TICKERS_CSV"
    exit 1
fi


# Define argument lists (numeric values for batch_size and learning_rate)
ticker="Combined"
window_days="55"         
alpha="0.5"                
predict_days= "5"    
lookback="0"
batch_size="32"    
# Training parameters      
epochs="30"

# Print the current combination being executed
echo ">> $PYTHON_SCRIPT -t $ticker -w $window_days -a $alpha -e $epochs -p $predict_days -b $batch_size -l $lookback $@"
python3 "$PYTHON_SCRIPT" -t "$ticker" -w "$window_days" -a "$alpha" -e "$epochs" -p "$predict_days" -b "$batch_size" -l "$lookback" "$@"
if [ $? -eq 0 ]; then
    echo "Python script executed successfully for combination: $arg1 $arg2 $arg3"
else
    echo "Error executing Python script for combination: $arg1 $arg2 $arg3"
    exit 1
fi            
