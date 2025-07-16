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
from datetime import date

result_file = "4MAs_train_result.csv"

LABEL_STAR = "EMStrend"
# LABEL_STAR = "Labels"
DATA_START_DATE = "2022-01-01"  # Start date for data download
DATA_END_DATE = "2024-12-31"  # End date for data download

UP_TREND = "UP"
DOWN_TREND = "DOWN"
SIDEWAY_TREND = "SIDEWAY"
LABEL_L = [UP_TREND, DOWN_TREND, SIDEWAY_TREND]

def data_label(ticker, window_days, alpha, predict_days, star_lookback):
    return f"{ticker}_{window_days}_{alpha}_{predict_days}_{star_lookback}"

def model_label(data_label_, model_name_, epochs_, batch_size_):
    return f"{data_label_}_{model_name_}_{epochs_}_{batch_size_}"

def retreive_model(model_label):
    model_file = os.path.join("models", f"{model_label}.pkl")
    if os.path.exists(model_file):
        with open(model_file, 'rb') as file:
            model = pickle.load(file)
        print(f"Model {model_label} loaded successfully.")
        return model
    else:
        print(f"Model {model_label} not found.")
        return None
  
# Step 1: Calculate EMAs and generate charts with price-based labeling
def calculate_emas(data, periods=[5, 10, 20, 30]):
    for period in periods:
        data[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
    return data

def detect_rising_star(data, idx, lookback):
    nback = lookback+2  # Number of candles to look back for the pattern
    if idx < nback or idx >= len(data):  # Need at least 7 prior candles (5 for trend + 2 for pattern)
        return False
    
    # Check for downtrend: previous 5 candlesticks' Labels must be "DOWN"
    if lookback > 0:
        prev_labels = data[LABEL_STAR].iloc[idx-nback:idx-2]
        if not all(label == DOWN_TREND for label in prev_labels if label is not None):
            return False
    
    # Check three candlestick pattern
    candle_1 = data.iloc[idx-2]
    candle_2 = data.iloc[idx-1]
    candle_3 = data.iloc[idx]
    
    # First candlestick: long down stick
    if not (candle_1['Close'] < candle_1['Open'] and 
            (candle_1['Open'] - candle_1['Close']) > 0.5 * (candle_1['High'] - candle_1['Low'])):
        return False
    
    # Second candlestick: open/close below first candle's low
    if not (candle_2['Open'] < candle_1['Low'] and candle_2['Close'] < candle_1['Low']):
        return False
    
    # Third candlestick: close higher than second's close, low higher than second's low
    if not (candle_3['Close'] > candle_2['Close'] and candle_3['Low'] > candle_2['Low']):
        return False
    
    print(f"Find rising star: lookback={lookback} at {data.index[idx]} with {candle_1['Close']}, {candle_2['Close']}, {candle_3['Close']}   ")
    return True

def detect_evening_star(data, idx, lookback):
    nback = lookback+2  # Number of candles to look back for the pattern
    if idx < nback or idx >= len(data):  # Need at least 7 prior candles (5 for trend + 2 for pattern)
        return False
    
    # Check for uptrend: previous 5 candlesticks' Labels must be "UP"
    if lookback > 0:
        prev_labels = data[LABEL_STAR].iloc[idx-nback:idx-2]
        if not all(label == UP_TREND for label in prev_labels if label is not None):
            return False
    
    # Check three candlestick pattern
    candle_1 = data.iloc[idx-2]
    candle_2 = data.iloc[idx-1]
    candle_3 = data.iloc[idx]
    
    # First candlestick: big up stick
    if not (candle_1['Close'] > candle_1['Open'] and 
            (candle_1['Close'] - candle_1['Open']) > 0.5 * (candle_1['High'] - candle_1['Low'])):
        return False
    
    # Second candlestick: high above first's high, low below first's low
    if not (candle_2['High'] > candle_1['High'] and candle_2['Low'] < candle_1['Low']):
        return False
    
    # Third candlestick: big down stick, high not above second's high, close below second's low
    if not (candle_3['Close'] < candle_3['Open'] and 
            candle_3['High'] <= candle_2['High'] and 
            candle_3['Close'] < candle_2['Low'] and
            (candle_3['Open'] - candle_3['Close']) > 0.5 * (candle_3['High'] - candle_3['Low'])):
        return False
    
    print(f"Find Evening star: lookback={lookback} at {data.index[idx]} with {candle_1['Close']}, {candle_2['Close']}, {candle_3['Close']}   ")    
    return True

def calulate_Alltrend(data_, prdict_days_, alpha_):
    labels = []
    for i in range(1, len(data_) - prdict_days_ - 1):
    
        if prdict_days_ > 0:
            lastdayptr = i+prdict_days_
            chart_data = data_.iloc[i:lastdayptr]
            if len(chart_data) != prdict_days_:
                continue
            
            # Label based on price change
            current_close = chart_data['Close'].iloc[0]
            next_close = chart_data['Close'].iloc[-1]
            sigma = chart_data['Close'].std()
            print("Close on [", chart_data.index[0], "]=", current_close, "[", data_.index[-1], "]=", next_close)

            if next_close > current_close + alpha_ * sigma:
                labels.append(UP_TREND)
                data_.at[data_.index[i], 'Labels'] = UP_TREND
            elif next_close < current_close - alpha_ * sigma:
                labels.append(DOWN_TREND)
                data_.at[data_.index[i], 'Labels'] = DOWN_TREND
            else:
                labels.append(SIDEWAY_TREND)
                data_.at[data_.index[i], 'Labels'] = SIDEWAY_TREND

        prev_day = data_.iloc[i-1]
        ema5 = prev_day['EMA_5']
        ema10 = prev_day['EMA_10']
        ema20 = prev_day['EMA_20']
        ema30 = prev_day['EMA_30']
        if ema5 > ema10 > ema20 > ema30:
            data_.at[data_.index[i], 'EMStrend'] = UP_TREND
        elif ema30 > ema20 > ema10 > ema5:
            data_.at[data_.index[i], 'EMStrend'] = DOWN_TREND
        else:
            data_.at[data_.index[i], 'EMStrend'] = SIDEWAY_TREND    

    return data_

def find_RE_star(data_, gap_, lookback_):
    for j in range(gap_+2, len(data_)):  # Start from 7 to ensure enough prior candles for trend check
        if detect_rising_star(data_, j, lookback_):
            data_.at[data_.index[j], 'RisingStar'] = data_['Low'].iloc[j] * 0.98  # Marker below candle
            print(f"Rising Star found at {data_.index[j]} with Low={data_['Low'].iloc[j]}")
        if detect_evening_star(data_, j, lookback_):
            data_.at[data_.index[j], 'EveningStar']  = data_['High'].iloc[j] * 1.02  # Marker above candle
            print(f"Evening Star found at {data_.index[j]} with High={data_['High'].iloc[j]}")

    # Debug: Check if markers are all NaN
    print(f"Rising Star markers: {data_['RisingStar'].notna().sum()} non-NaN values")
    print(f"Evening Star markers: {data_['EveningStar'].notna().sum()} non-NaN values")
    return data_

def combine_data(datalabel_, combinedlabel_):
    combined_filename=f"{combinedlabel_}_metadata.csv"
    print(f"combine_data with datalabel={datalabel_}, combined_filename={combined_filename}")
    column_names = ['chart_path', 'label']
    test_combined_fd = pd.read_csv(os.path.join("test_metadata", combined_filename), usecols=column_names) if os.path.exists(os.path.join("test_metadata", combined_filename)) else pd.DataFrame(columns=column_names)
    train_combined_fd = pd.read_csv(os.path.join("train_metadata", combined_filename), usecols=column_names) if os.path.exists(os.path.join("train_metadata", combined_filename)) else pd.DataFrame(columns=column_names)
    train_fullp = os.path.join("train_metadata", f"{datalabel_}_metadata.csv")
    test_fullp = os.path.join("test_metadata", f"{datalabel_}_metadata.csv")
    train_df = pd.read_csv(train_fullp)
    test_df = pd.read_csv(test_fullp)
    test_combined_fd = pd.concat([test_combined_fd, test_df], ignore_index=True)
    train_combined_fd = pd.concat([train_combined_fd, train_df], ignore_index=True) 
    test_combined_fd.to_csv(os.path.join("test_metadata", combined_filename), index=False)
    train_combined_fd.to_csv(os.path.join("train_metadata", combined_filename), index=False)

def download_setup_data(ticker, predict_days, window_days, alpha, lback, start_date="", end_date=""):
    print(f"download_setup_data {ticker} with predict_days={predict_days}, window_days={window_days}, alpha={alpha}, lookback={lback}, start_date={start_date}, end_date={end_date}")

    # Download data
    try:
        filename=os.path.join("data", f"{ticker}_{start_date}_{end_date}.csv")
        if os.path.exists(filename):
            print(f"Loading data from {filename}")
            data = pd.read_csv(filename, index_col=0, parse_dates=True)
        else:
            print(f"Downloading data for {ticker} from Yahoo Finance")
            # Download data for the last 4 years
            # period="4y" to ensure we have enough data for trend calculation
            if len(start_date)>0 and len(end_date)> 0:
                data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
            else:
                data = yf.download(ticker, period="4y", progress=False)
            data.columns = [col[0] for col in data.columns]
            print(data.info())
            data.to_csv(filename)
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None
    
    # Check if data is a valid DataFrame
    if not isinstance(data, pd.DataFrame):
        print(f"Error: Downloaded data for {ticker} is not a DataFrame, got type {type(data)}")
        return None
    
    # Check for required columns
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_cols):
        print(f"Error: DataFrame missing required columns: {required_cols}")
        return None
    
    # Ensure numeric data and handle NaN
    for col in required_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    
    # Check if enough data is available
    if len(data) < window_days + predict_days + 7:  # Increased to account for trend check
        print(f"Error: Insufficient data for {ticker}, got {len(data)} rows, need at least {window_days + predict_days + 7}")
        return None
       
    # Calculate EMAs
    data = calculate_emas(data)
    
    data['EMStrend'] = None
    data['Labels'] = None
    data['RisingStar'] = None
    data['EveningStar'] = None
    # Convert to numeric type with np.nan for None
    data['RisingStar'] = pd.to_numeric(data['RisingStar'], errors='coerce')
    data['EveningStar'] = pd.to_numeric(data['EveningStar'], errors='coerce')

    # Pre-compute Labels for the entire dataset

    data = calulate_Alltrend(data, predict_days, alpha)  # Ignore returned labels, use data['Labels']
  
    # find Rising Star and Evening Star patterns
    data = find_RE_star(data, predict_days, lback)

    return data

def generate_chart(data, ticker, predict_days, window_days, datalabel_):
    print(f"generate_chart {ticker} with predict_days={predict_days}, window_days={window_days}, datalabel={datalabel_}")
    chart_dir = os.path.join("chartsdata", datalabel_)
    os.makedirs(chart_dir, exist_ok=True)
    charts, chart_labels = [], []  # Separate list for chart labels

    data['predict_date'] = np.NAN
    # Generate charts and detect patterns
    for i in range(len(data) - window_days - predict_days):
        lastdayptr = i + window_days
        chart_data = data.iloc[i:lastdayptr]
        # print(chart_data)
        if len(chart_data) != window_days:
            print(f"Skipping window {i} due to insufficient or invalid data: {len(chart_data)} chart_data in {window_days} days")
            continue
            
        # Detect Rising Star and Evening Star patterns
        rising_star_markers = chart_data['RisingStar']
        evening_star_markers = chart_data['EveningStar']
        
        # Build additional plots list, only include scatter plots if markers exist
        ap = [
            mpf.make_addplot(chart_data['EMA_5'], color='red', width=1),
            mpf.make_addplot(chart_data['EMA_10'], color='blue', width=1),
            mpf.make_addplot(chart_data['EMA_20'], color='green', width=1),
            mpf.make_addplot(chart_data['EMA_30'], color='yellow', width=1)
        ]
        
        # Only add scatter plots if there are non-NaN markers
        if rising_star_markers.notna().any():
            # Ensure the series contains only numeric or np.nan values
            rising_star_markers = pd.to_numeric(rising_star_markers, errors='coerce')
            print(f"Adding Rising Star markers for window {i} is ")
            ap.append(mpf.make_addplot(rising_star_markers, type='scatter', marker='^', color='cyan', markersize=100))
        if evening_star_markers.notna().any():
            evening_star_markers = pd.to_numeric(evening_star_markers, errors='coerce')
            print(f"Adding Evening Star markers for window {i} is ")
            ap.append(mpf.make_addplot(evening_star_markers, type='scatter', marker='v', color='magenta', markersize=100))

        start_d = chart_data.index[0].strftime('%Y%m%d')
        end_d = chart_data.index[-1].strftime('%Y%m%d')
        predict_d = data.index[lastdayptr+1].strftime('%Y-%m-%d')
        # print(start_d, " , ", end_d, " , ", predict_d)
        chart_path = os.path.join(chart_dir, f"chart_{ticker}_{start_d}_{end_d}_{i}.png")
        try:
            fig, ax = mpf.plot(chart_data, type='candle', addplot=ap, savefig=chart_path, warn_too_much_data=1000, returnfig=True)
            plt.close(fig)  # Close the figure to free memory
            plt.close('all')  # Close all figures to prevent memory accumulation
        except Exception as e:
            print(f"Error generating chart {chart_path}: {e}")
            continue
        
        # Use pre-computed Label
        label = data.at[data.index[lastdayptr], 'Labels']
        if label is None:
            print(f"Skipping window {i} due to missing label")
            continue
        charts.append(chart_path)
        chart_labels.append(label)  # Append to chart_labels instead of labels
        data.at[data.index[lastdayptr], "predict_date"] = predict_d

    print(f"Generated {len(charts)} charts for {ticker}")
    if not charts:
        print(f"Error: No valid charts generated for {ticker}. Check data availability or chart generation logic.")
        return [], [], []
    return charts, chart_labels, data

def generate_candlestick_with_emas(ticker, gap, days, alpha, datalabel_, lback, start_date="", end_date=""):
    print(f"generate_candlestick_with_emas {ticker} with predict_days/gap={gap}, days={days}, alpha={alpha}, datalabel={datalabel_}, lookback={lback}, start_date={start_date}, end_date={end_date} ")
    if len(start_date)>0:
        start_d= start_date
    if len(end_date)>0:
        end_d = end_date
    data = download_setup_data(ticker, gap, days, alpha, lback, start_date, end_date)

    if data is None:
        return [], [], [], [], data

    chart_dir = os.path.join("chartdata", datalabel_)
    os.makedirs(chart_dir, exist_ok=True)

    charts, chart_labels, data = generate_chart(data, ticker, predict_days, window_days, datalabel_)

    # Split into training and testing sets
    print(f"charts={len(charts)}, chart_labels={len(chart_labels)}")
    try:
        train_charts, test_charts, train_labels, test_labels = train_test_split(
            charts, chart_labels, test_size=0.2, random_state=42, stratify=chart_labels
        )
    except ValueError as e:
        print(f"Error in train_test_split: {e}")
        return [], [], [], [], data
    
    # Save metadata to CSV
    train_df = pd.DataFrame({"chart_path": train_charts, "label": train_labels})
    test_df = pd.DataFrame({"chart_path": test_charts, "label": test_labels})
    train_fullp = os.path.join("train_metadata", f"{datalabel_}_metadata.csv")
    print(train_fullp)
    os.makedirs(os.path.dirname(train_fullp), exist_ok=True)
    train_df.to_csv(train_fullp, index=False)
    test_fullp = os.path.join("test_metadata", f"{datalabel_}_metadata.csv")
    print(test_fullp)
    os.makedirs(os.path.dirname(test_fullp), exist_ok=True)
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
def load_data(csv_path, limit=20000):
    try:
        df = pd.read_csv(csv_path).sample(frac=1, random_state=42).reset_index(drop=True)
        if len(df) > limit:
            df = df.head(limit)
        print(f"Loaded {len(df)} records from {csv_path}")
    except FileNotFoundError:
        print(f"Error: Metadata file {csv_path} not found.")
        return np.array([]), np.array([])
    X, y = [], []
    for _, row in df.iterrows():
        img = preprocess_image(row['chart_path'])
        if img is not None:
            X.append(img)
            y.append(LABEL_L.index(row['label']))
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
def plot_accuracy(ticker, history, model_name, model_label):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_label} Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("tperf", f"{model_label}_accuracy_plot.png"))
    plt.close()

# Step 6: Evaluate model performance
def evaluate_model(ticker, model, X_test, y_test, model_name, model_label):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    return cal_accuracy(ticker, y_test, y_pred_classes, model_name, model_label)

def cal_accuracy(ticker, y_test, y_pred_classes, model_name, model_label_):
    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred_classes, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_classes, average='weighted', zero_division=0)
    
    print(f"\n{model_label_} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABEL_L, yticklabels=LABEL_L)
    plt.title(f"{ticker}_{model_label_} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join("matrix", f"{ticker}_{model_label_}_confusion_matrix.png"))
    plt.close()
    return accuracy, precision, recall, f1

def prepare_data(ticker, window_days, alpha, predict_days, lookback_, data_label_, start_date, end_date):
    # Generate and split data
    fulldata_name = os.path.join("data", f"full_{data_label_}_{start_date}_{end_date}.csv")

    if os.path.exists(fulldata_name):
        print(f"{fulldata_name} already exists, skipping prepare data.")
        full_data = pd.read_csv(fulldata_name)
        return full_data
    
    train_charts, train_labels, test_charts, test_labels, fulldata = generate_candlestick_with_emas(ticker, predict_days, window_days, alpha, data_label_, lookback_, start_date, end_date)

    if not train_charts or not test_charts or fulldata is None:
        print("No charts generated or invalid data, exiting.")
        return None
        
    # dump the fulldata with all calculated EMAs and labels
    fulldata.to_csv(fulldata_name, index=True)
    print(f"Fulldata saved for {ticker} with {len(fulldata)} rows.")

    em_y = fulldata['EMStrend'].dropna()
    print(f"EMStrend labels: {len(em_y)}")
    lb_y = fulldata['Labels'].dropna()
    print(f"Price-based labels: {len(lb_y)}")

    return fulldata         

def model_training(ticker, window_days, alpha, epochs, predict_days, batch_size, result_df_, lookback_, data_label_, result_file_):
    print(f"model_training({ticker}, {window_days}, {alpha}, {epochs}, {predict_days}, {batch_size}, result_df, {lookback_}, {data_label_}, {result_file_}")
    # Load training and testing data
    X_train, y_train = load_data(os.path.join("train_metadata", f"{data_label_}_metadata.csv"), limit=5000)
    X_test, y_test = load_data(os.path.join("test_metadata", f"{data_label_}_metadata.csv"), limit=2000)
    print(f"Loaded {len(X_train)} training images and {len(X_test)} testing images for {ticker}.")
    if X_train.size == 0 or X_test.size == 0:
        print("No valid images loaded, exiting.")
        return result_df_

    # Define models to train
    # models = [build_cnn_model(), build_cnn_lstm_model(data_label_), build_dense_model()]
    models = [build_cnn_model(),  build_dense_model()]

    # Train and evaluate each model
    for model, model_name in models:
        # if model file exist, skip the process
        modellabel = model_label(data_label_, model_name, epochs, batch_size)
        model_file = os.path.join("models", f"{modellabel}.pkl")

        if os.path.exists(model_file):
            print(f"{model_file} already exists, skipping training.")
            continue
        print(f"\nTraining {model_name}...")
        try:
            history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, batch_size=batch_size, verbose=1)
            plot_accuracy(ticker, history, model_name, modellabel)
            acc, prec, recall, f1 = evaluate_model(ticker, model, X_test, y_test, model_name, modellabel)
            # Append the row
            new_row = [ticker, model_name, window_days, alpha, epochs, predict_days, batch_size, lookback_, acc, prec, recall, f1]
            result_df_.loc[len(result_df_)] = new_row
            result_df_.to_csv(result_file_, index=False)
            os.makedirs(os.path.dirname(model_file), exist_ok=True)
            with open(model_file, 'wb') as file:
                pickle.dump(model, file)
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue
        # Free memory
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    return result_df_

# Step 8: Main function
def main(ticker, window_days, alpha, epochs, predict_days, batch_size, result_df, lookback_, data_label_, start_date, end_date):

    result_df = model_training(ticker, window_days, alpha, epochs, predict_days, batch_size, result_df, lookback_, data_label_, result_file)

    fulldata_name = os.path.join("data", f"full_{data_label_}_{start_date}_{end_date}.csv")
    fulldata = pd.read_csv(fulldata_name)

    trend_mapping = {"UP": 1, "DOWN": 2, "SIDEWAY": 3}
    em_y = fulldata['EMStrend'].dropna().to_list()
    lb_y = fulldata['Labels'].dropna().to_list()
    emyy = [trend_mapping[em] for em in em_y]
    lbyy = [trend_mapping[em] for em in lb_y]

    try:
        m_label = model_label(data_label_, '4MAs', epochs, batch_size)
        acc, prec, recall, f1 = cal_accuracy(ticker, lbyy, emyy, '4MAs', m_label)
        new_row = [ticker, '4MAs', window_days, alpha, epochs, predict_days, batch_size, lookback_, acc, prec, recall, f1]
        result_df.loc[len(result_df)] = new_row
        result_df.to_csv(result_file, index=False)
    except Exception as e:
        print(f"Error calculating final accuracy: {e}")

    return result_df

  
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run TensorFlow model with specified arguments")
    parser.add_argument('-t', '--ticker', dest='ticker', type=str, help="Stock ticker for prediction", default='')
    parser.add_argument('-w', '--window_days', dest='window_days', type=int, help="Window days for training", default=30)
    parser.add_argument('-a', '--alpha', dest='alpha', type=float, help="Alpha parameter for training", default=0.5)
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, help="Number of training epochs", default=25)
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, help="Batch size for training", default=32)
    parser.add_argument('-p', '--predict_days', dest='predict_days', type=int, help="Number of days to predict", default=2)
    parser.add_argument('-l', '--lookback', dest='lookback', type=int, help="Number of days for identify Star", default=5)
    parser.add_argument('-d', '--data', dest='data', action='store_true', default=False)
    parser.add_argument('-c', '--combine', dest='combineLabel', type=str, help="Label for combined data", default='')
    parser.add_argument('-T', '--Training', dest='TrainLabel', type=str, help="Label for Training", default='')
    parser.add_argument('-ba', '--buildall', dest='buildall', action='store_true', help="Build all models from existing pre-data", default=False)

    args = parser.parse_args()
    # Access arguments
    ticker = args.ticker
    window_days = args.window_days
    alpha = args.alpha
    epochs = args.epochs
    predict_days = args.predict_days   
    batch_size = args.batch_size
    star_lookback = args.lookback
    combined_Label = args.combineLabel
    TrainLabel = args.TrainLabel
    if len(TrainLabel) > 0:
        result_file = f"{TrainLabel}_result.csv"
        print(f"Using training result file: {result_file}")

    column_names = ['ticker', 'model', 'window_days', 'alpha', 'epochs', 'predict_days', 'batch_size', "Lookback", 'accuracy', 'precision', 'recall', 'f1']
    if os.path.exists(result_file):
        print(f"Reloading {result_file} for appending results.")
        result = pd.read_csv(result_file)
    else:
        print(f"Creating {result_file}")
        result = pd.DataFrame(columns=column_names)
  
    if len(ticker) > 0:
        # parepare data
        data_label = data_label(ticker, window_days, alpha, predict_days, star_lookback)
        print(f"Data label: {data_label}")
        if args.data:
            fulldata = prepare_data(ticker, window_days, alpha, predict_days, star_lookback, data_label, DATA_START_DATE, DATA_END_DATE)
        elif args.buildall:
            result = main(ticker, window_days, alpha, epochs, predict_days, batch_size, result, star_lookback, data_label)
        elif len(combined_Label) > 0:
            combine_data(data_label, combined_Label)
        elif len(TrainLabel) > 0:
            model_training(TrainLabel, 0, 0, epochs, 0, batch_size, result, 0, TrainLabel, result_file)
        else:
            fulldata = prepare_data(ticker, window_days, alpha, predict_days, star_lookback, data_label, DATA_START_DATE, DATA_END_DATE)
            if fulldata is not None and len(fulldata)>0:
                result = main(ticker, window_days, alpha, epochs, predict_days, batch_size, result, star_lookback, data_label, DATA_START_DATE, DATA_END_DATE)
                print(result)
    else:
        print("No ticker provided, exiting.")

