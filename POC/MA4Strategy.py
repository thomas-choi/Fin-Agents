import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
import pickle
import gc
import argparse

result_file = "4MAs_result.csv"

# Step 1: Calculate EMAs and generate charts with price-based labeling
def calculate_emas(data, periods=[5, 10, 20, 30]):
    for period in periods:
        data[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
    return data

#
#   generate_candlestick_with_emas(ticker, predict_days, window_days, alpha, train_p, test_p)
# 
def generate_candlestick_with_emas(ticker, gap=5, days=20, alpha=0.5, train_name="train_charts", test_name="test_charts"):
    print(f"generate_candlestick_with_emas {ticker} with predict_days/gap={gap}, days={days}, alpha={alpha}, train_name={train_name}, test_name={test_name}")
    # Create directories for training and testing images
    train_dir = os.path.join("train", train_name)
    test_dir = os.path.join("test", test_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Download data
    try:
        data = yf.download(ticker, period="2y", progress=False)
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return [], [], [], []
    
    # Check if data is a valid DataFrame
    if not isinstance(data, pd.DataFrame):
        print(f"Error: Downloaded data for {ticker} is not a DataFrame, got type {type(data)}")
        return [], [], [], []
    
    # Check for required columns
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_cols):
        print(f"Error: DataFrame missing required columns: {required_cols}")
        return [], [], [], []
    
    data.columns = [col[0] for col in data.columns]
    print(data.info())
    # Ensure numeric data and handle NaN
    for col in required_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    
    # Check if enough data is available
    if len(data) < days + 1:
        print(f"Error: Insufficient data for {ticker}, got {len(data)} rows, need at least {days + 1}")
        return [], [], [], []
       
    # Calculate EMAs
    data = calculate_emas(data)
    
    data['EMStrend'] = None
    data['Labels'] = None
    charts, labels, EMTrends = [], [], []
    
    for i in range(len(data) - days - 1):
        lastdayptr = i+days
        chart_data = data.iloc[i:lastdayptr]
        if len(chart_data) != days:
            continue
            
        ap = [
            mpf.make_addplot(chart_data['EMA_5'], color='red', width=1),
            mpf.make_addplot(chart_data['EMA_10'], color='blue', width=1),
            mpf.make_addplot(chart_data['EMA_20'], color='green', width=1),
            mpf.make_addplot(chart_data['EMA_30'], color='yellow', width=1)
        ]

        start_d = chart_data.index[0].strftime('%Y%m%d')
        end_d = chart_data.index[-1].strftime('%Y%m%d')
        chart_path = f"chart_{ticker}_{i}_{start_d}_{end_d}.png"
        try:
            mpf.plot(chart_data, type='candle', addplot=ap, savefig=chart_path, warn_too_much_data=1000)
        except Exception as e:
            print(f"Error generating chart {chart_path}: {e}")
            continue
        
        # Label based on price change
        current_close = chart_data['Close'].iloc[-1]
        next_close = data['Close'].iloc[i+gap]
        sigma = chart_data['Close'].std()
        print("Close on [", chart_data.index[-1], "]=", current_close, "[", data.index[i+days], "]=", next_close)
        
        if next_close > current_close + alpha * sigma:
            labels.append("UP")
            data.at[data.index[lastdayptr], 'Labels'] = "UP"
            # data.iloc[lastdayptr]['Labels'] = "UP"
            # data['Labels'][lastdayptr] = "UP"
        elif next_close < current_close - alpha * sigma:
            labels.append("DOWN")
            data.at[data.index[lastdayptr], 'Labels'] = "DOWN"
            # data.iloc[lastdayptr]['Labels'] = "DOWN"
            # data['Labels'][lastdayptr] = "DOWN"
        else:
            labels.append("SIDEWAY")
            data.at[data.index[lastdayptr], 'Labels'] = "SIDEWAY"
            # data.iloc[lastdayptr]['Labels'] = "SIDEWAY"
            # data.at[lastdayptr, 'Labels'] = "SIDEWAY"
            # data['Labels'][lastdayptr] = "SIDEWAY"
        
        charts.append(chart_path)

        # Label based on EMA ordering on the last day
        last_day = data.iloc[lastdayptr]
        ema5 = last_day['EMA_5']
        ema10 = last_day['EMA_10']
        ema20 = last_day['EMA_20']
        ema30 = last_day['EMA_30']
        
        if ema5 > ema10 > ema20 > ema30:
            data.at[data.index[lastdayptr], 'EMStrend'] = "UP"
            # data.iloc[lastdayptr]['EMStrend'] = "UP"
            # data['EMStrend'][lastdayptr] = "UP"
        elif ema30 > ema20 > ema10 > ema5:
            data.at[data.index[lastdayptr], 'EMStrend'] = "DOWN"
            # data.iloc[lastdayptr]['EMStrend'] = "DOWN"
            # data.at[lastdayptr, 'EMStrend'] = "DOWN"
            # data['EMStrend'][lastdayptr] = "DOWN"
        else:
            data.at[data.index[lastdayptr], 'EMStrend'] = "SIDEWAY"
            # data.iloc[lastdayptr]['EMStrend'] = "SIDEWAY"
            # data.at[lastdayptr, 'EMStrend'] = "SIDEWAY"
            # data['EMStrend'][lastdayptr] = "SIDEWAY"

    # Split into training and testing sets
    train_charts, test_charts, train_labels, test_labels = train_test_split(
        charts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Move charts to respective directories and update paths
    train_chart_paths, test_chart_paths = [], []
    for chart, label in zip(train_charts, train_labels):
        new_path = os.path.join(train_dir, os.path.basename(chart))
        shutil.move(chart, new_path)
        train_chart_paths.append(new_path)
    
    for chart, label in zip(test_charts, test_labels):
        new_path = os.path.join(test_dir, os.path.basename(chart))
        shutil.move(chart, new_path)
        test_chart_paths.append(new_path)
    
    # Save metadata to CSV
    train_df = pd.DataFrame({"chart_path": train_chart_paths, "label": train_labels})
    test_df = pd.DataFrame({"chart_path": test_chart_paths, "label": test_labels})
    train_fullp = os.path.join("train_metadata", f"{train_name}_metadata.csv")
    print(train_fullp)
    train_df.to_csv(train_fullp, index=False)
    test_fullp = os.path.join("test_metadata", f"{test_name}_metadata.csv")
    print(test_fullp)
    test_df.to_csv(test_fullp, index=False)
    
    return train_charts, train_labels, test_charts, test_labels, data

# Step 2: Preprocess images
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).resize((224, 224)).convert('RGB')
        return np.array(img) / 255.0
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Step 3: Load data from CSV
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X, y = [], []
    for _, row in df.iterrows():
        img = preprocess_image(row['chart_path'])
        if img is not None:
            X.append(img)
            y.append(["UP", "DOWN", "SIDEWAY"].index(row['label']))
    return np.array(X), np.array(y)

# Step 4: Define multiple models
def build_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, "CNN"

def build_cnn_lstm_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Reshape((-1, 64)),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, "CNN-LSTM"

def build_dense_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, "Dense"

# Step 5: Plot and save epoch vs. accuracy
def plot_accuracy(ticker, history, model_name):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{ticker}_{model_name} Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{ticker}_{model_name}_accuracy_plot.png")
    plt.close()

# Step 6: Evaluate model performance
def evaluate_model(ticker, model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    return cal_accuracy(ticker, y_test, y_pred_classes, model_name)

def cal_accuracy(ticker, y_test, y_pred_classes, model_name):
    # print("cal_accuracy: y_test=", y_test, ",   y_pred_classes=", y_pred_classes)
    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred_classes, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_classes, average='weighted', zero_division=0)
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["UP", "DOWN", "SIDEWAY"], yticklabels=["UP", "DOWN", "SIDEWAY"])
    plt.title(f"{ticker}_{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{ticker}_{model_name}_confusion_matrix.png")
    plt.close()
    return accuracy, precision, recall, f1

# Step 7: Generate EMA-based trend labels
def EMSTrend(ticker, days=20):
    # Download data
    try:
        data = yf.download(ticker, period="2y", progress=False)
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return [], []
    
    # Check if data is a valid DataFrame
    if not isinstance(data, pd.DataFrame):
        print(f"Error: Downloaded data for {ticker} is not a DataFrame, got type {type(data)}")
        return [], []
    
    # Check for required columns
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_cols):
        print(f"Error: DataFrame missing required columns: {required_cols}")
        return [], []
    
    data.columns = [col[0] for col in data.columns]
    # Ensure numeric data and handle NaN
    for col in required_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    
    # Check if enough data is available
    if len(data) < days + 1:
        print(f"Error: Insufficient data for {ticker}, got {len(data)} rows, need at least {days + 1}")
        return [], []
       
    # Calculate EMAs
    data = calculate_emas(data)
    
    time_slots, labels = [], []
    
    for i in range(len(data) - days - 1):
        window_data = data.iloc[i:i+days]
        if len(window_data) != days:
            continue
            
        # Get the last two days for crossover detection
        ema_5 = window_data['EMA_5'].iloc[-2:]
        ema_20 = window_data['EMA_20'].iloc[-2:]
        
        if len(ema_5) < 2 or len(ema_20) < 2:
            continue  # Skip if not enough data for crossover detection
        
        # Store the end date of the window as the time slot
        time_slot = window_data.index[-1].strftime('%Y-%m-%d')
        
        # Check for crossover
        if ema_5.iloc[-2] <= ema_20.iloc[-2] and ema_5.iloc[-1] > ema_20.iloc[-1]:
            labels.append("UP")  # Golden Cross
        elif ema_5.iloc[-2] >= ema_20.iloc[-2] and ema_5.iloc[-1] < ema_20.iloc[-1]:
            labels.append("DOWN")  # Death Cross
        else:
            labels.append("SIDEWAY")  # No crossover
        
        time_slots.append(time_slot)
    
    return time_slots, labels

# Step 8: Main function
def main(ticker, window_days, alpha, epochs, predict_days, batch_size, result_df):

    # Generate and split data
    train_p = f"{ticker}_{window_days}_{alpha}_{epochs}_{predict_days}_{batch_size}"
    test_p = f"{ticker}_{window_days}_{alpha}_{epochs}_{predict_days}_{batch_size}"
    train_charts, train_labels, test_charts, test_labels, fulldata = 
    generate_candlestick_with_emas(ticker, predict_days, window_days, alpha, train_p, test_p)

    if not train_charts or not test_charts:
        print("No charts generated, exiting.")
        return result_df
        
    em_y = fulldata['EMStrend'].dropna()
    print(len(em_y))
    lb_y = fulldata['Labels'].dropna()
    print(len(lb_y))

    # Load training and testing data
    X_train, y_train = load_data(os.path.join("train_metadata", f"{train_p}_metadata.csv"))
    X_test, y_test = load_data(os.path.join("test_metadata", f"{test_p}_metadata.csv"))
    
    # print("Trainning data:  ", X_train, ",", y_train)
    # print("Testing data:  ", X_test, ",", y_test)
    
    if X_train.size == 0 or X_test.size == 0:
        print("No valid images processed, exiting.")
        return result_df
        
    # Define models to train
    models = [build_cnn_model(), build_cnn_lstm_model(), build_dense_model()]
    
    # Train and evaluate each model
    for model, model_name in models:
        # if model file exist, skip the process
        model_file = os.path.join("models", f"{ticker}_{model_name}_{window_days}_{alpha}_{epochs}_{predict_days}_{batch_size}.pkl")
        if os.path.exists(model_file):
            print(f"{model_file} model exist")
            continue
        print(f"\nTraining {model_name}...")
        history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, batch_size=batch_size, verbose=1)
        plot_accuracy(ticker, history, model_name)
        acc, prec, recall, f1 = evaluate_model(ticker, model, X_test, y_test, model_name)
        # Append the row
        new_row =[ticker,model_name,window_days, alpha, epochs, predict_days, batch_size,acc, prec, recall, f1]
        result_df.loc[len(result_df)] = new_row
        result_df.to_csv(result_file, index=False)
        with open(model_file, 'wb') as file:
            pickle.dump(model, file)  
        # Free memory
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    trend_mapping = {"UP": 1, "DOWN": 2, "SIDEWAY": 3}
    em_y = fulldata['EMStrend'].dropna().to_list()
    lb_y = fulldata['Labels'].dropna().to_list()
    emyy = [trend_mapping[em] for em in em_y]
    lbyy = [trend_mapping[em] for em in lb_y]

    acc, prec, recall, f1 = cal_accuracy(ticker, lbyy, emyy, model_name)
    new_row =[ticker,'4MAs',window_days, alpha, epochs, predict_days, batch_size,acc, prec, recall, f1]
    result_df.loc[len(result_df)] = new_row
    result_df.to_csv(result_file, index=False)
    return result_df

if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run TensorFlow model with specified arguments")
    parser.add_argument('-t', '--ticker', dest='ticker', type=str, help="ticker for predict", default='')
    parser.add_argument('-w', '--window_days', dest='window_days', type=int, help="window days for training", default=25)
    parser.add_argument('-a', '--alpha', dest='alpha', type=float, help="alpha for model training", default=0.5)
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=25, help="epochs for training")
    parser.add_argument('-b', '--batch_size', dest='batch_size', default=32, type=int, help="batch size for training")
    parser.add_argument('-p', '--predict_days', dest='predict_days', default=5, type=int, help="number of days from today to predict")

    args = parser.parse_args()
    # Access arguments
    ticker = args.ticker
    window_days = args.window_days
    alpha = args.alpha
    epochs = args.epochs
    predict_days = args.predict_days   
    batch_size = args.batch_size

    column_names=['ticker','model','window_days','alpha','epochs','predict_days','batch_size','accuracy','precision','recall','f1']
    if os.path.exists(result_file):
        print(f"Reload {result_file} for append result.")
        result = pd.read_csv(result_file)
    else:
        print(f"Create {result_file}")
        result=pd.DataFrame(columns=column_names)
  
    if len(ticker)> 0:
        result = main(ticker, window_days, alpha, epochs, predict_days, batch_size, result)
    else:
        print("No ticker provided, exiting.")

    print(result)