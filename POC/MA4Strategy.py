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

# Step 1: Calculate EMAs and generate charts with price-based labeling
def calculate_emas(data, periods=[5, 10, 20, 30]):
    for period in periods:
        data[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
    return data

def generate_candlestick_with_emas(ticker, gap=5, days=20, alpha=0.5, train_dir="train_charts", test_dir="test_charts"):
    # Create directories for training and testing images
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
    train_df.to_csv(f"{train_dir}_metadata.csv", index=False)
    test_df.to_csv(f"{test_dir}_metadata.csv", index=False)
    
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
def main(ticker="AAPL", days=20, alpha=0.5):
    # Generate and split data for chart-based training
    train_charts, train_labels, test_charts, test_labels, EMSdata = generate_candlestick_with_emas(ticker, days, alpha)
    
    if not train_charts or not test_charts:
        print("No charts generated, exiting.")
        return
    
    # Load training and testing data
    X_train, y_train = load_data(f"{ticker}_train_metadata.csv")
    X_test, y_test = load_data(f"{ticker}_test_metadata.csv")
    
    if X_train.size == 0 or X_test.size == 0:
        print("No valid images processed, exiting.")
        return
    
    # Define models to train
    models = [build_cnn_model(), build_cnn_lstm_model(), build_dense_model()]
    
    # Train and evaluate each model
    for model, model_name in models:
        print(f"\nTraining {model_name}...")
        history = model.fit(X_train, y_train, epochs=15, validation_split=0.2, batch_size=32, verbose=1)
        plot_accuracy(ticker, history, model_name)
        evaluate_model(ticker, model, X_test, y_test, model_name)
    
    # Predict on a test chart (example)
    new_chart_path = test_charts[0] if test_charts else None
    if new_chart_path and os.path.exists(new_chart_path):
        new_chart = preprocess_image(new_chart_path)
        if new_chart is not None:
            for model, model_name in models:
                prediction = model.predict(np.array([new_chart]))
                trend = ["UP", "DOWN", "SIDEWAY"][np.argmax(prediction)]
                print(f"{model_name} predicted trend for {new_chart_path}: {trend} (Probabilities: {prediction[0]})")
    
    # Generate EMA-based trend labels using the new function
    time_slots, ema_labels = EMSTrend(ticker, days)
    if time_slots:
        print(f"\nEMA Trend Labels for {ticker}:")
        for slot, label in zip(time_slots, ema_labels):
            print(f"Date: {slot}, Trend: {label}")
    
    # Clean up chart directories (optional, comment out to keep images)
    # shutil.rmtree("train_charts")
    # shutil.rmtree("test_charts")
    # if os.path.exists(f"{ticker}_train_metadata.csv"):
    #     os.remove(f"{ticker}_train_metadata.csv")
    # if os.path.exists(f"{ticker}_test_metadata.csv"):
    #     os.remove(f"{ticker}_test_metadata.csv")

if __name__ == "__main__":
    ticker = "AAPL"
    days = 20
    alpha = 0.5
    main(ticker, days, alpha)