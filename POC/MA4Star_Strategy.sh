#!/bin/bash

# Path to your Python script
PYTHON_SCRIPT="/home/thomas/projects/Fin-Agents/POC/MA4Star_Strategy.py"

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

# Define argument lists (numeric values for batch_size and learning_rate)
TICKERS=("^HSI")
WIN_DAYS_L=("30" "55" )          
ALPHA_L=("0.4" "0.5" "0.6")                 
EPOCHS_L=("30") 
PREDICT_DAYS_L=("2" "5" "8")    
LOOKBK_L=("2" "5")
BATCH_SIZE_L=("32")           

# Nested loops to iterate through all combinations
for ticker in "${TICKERS[@]}"; do
    for window_days in "${WIN_DAYS_L[@]}"; do
        for alpha in "${ALPHA_L[@]}"; do
            for epochs in "${EPOCHS_L[@]}"; do
                for predict_days in "${PREDICT_DAYS_L[@]}"; do
                    for batch_size in "${BATCH_SIZE_L[@]}"; do
                        for lookback in "${LOOKBK_L[@]}"; do
   
                            # Print the current combination being executed
                            echo ">> $PYTHON_SCRIPT -t $ticker -w $window_days -a $alpha -e $epochs -p $predict_days -b $batch_size -l"
                            python3 "$PYTHON_SCRIPT" -t "$ticker" -w "$window_days" -a "$alpha" -e "$epochs" -p "$predict_days" -b "$batch_size" -l "$lookback"
                            if [ $? -eq 0 ]; then
                                echo "Python script executed successfully for combination: $arg1 $arg2 $arg3"
                            else
                                echo "Error executing Python script for combination: $arg1 $arg2 $arg3"
                                exit 1
                            fi            
                        done
                    done
                done
            done
        done
    done
done
echo "All combinations executed."