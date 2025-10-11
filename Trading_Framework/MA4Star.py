import numpy as np
import pandas as pd
from strategies import TradingStrategy
import mplfinance as mpf
import os
import logging
import matplotlib.pyplot as plt
import talib


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

chg_limit = 0.005 # Change limit for labeling
alpha = 0.5
LABEL_STAR = "MA4Star_L"
UP_TREND = 1.0
DOWN_TREND = -1.0
SIDEWAY_TREND = 0.0

def calculate_emas(data, periods=[5, 10, 20, 30]):
    for period in periods:
        # data[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
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
    
    logging.debug(f"Find rising star: lookback={lookback} at {data.index[idx]} with {candle_1['Close']}, {candle_2['Close']}, {candle_3['Close']}   ")
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
    
    logging.debug(f"Find Evening star: lookback={lookback} at {data.index[idx]} with {candle_1['Close']}, {candle_2['Close']}, {candle_3['Close']}   ")    
    return True

def find_RE_star(data_, gap_, lookback_):
    for j in range(gap_+2, len(data_)):  # Start from 7 to ensure enough prior candles for trend check
        if detect_rising_star(data_, j, lookback_):
            data_.at[data_.index[j], 'RisingStar'] = data_['Low'].iloc[j] * 0.98  # Marker below candle
            logging.debug(f"Rising Star found at {data_.index[j]} with Low={data_['Low'].iloc[j]}")
        if detect_evening_star(data_, j, lookback_):
            data_.at[data_.index[j], 'EveningStar']  = data_['High'].iloc[j] * 1.02  # Marker above candle
            logging.debug(f"Evening Star found at {data_.index[j]} with High={data_['High'].iloc[j]}")

    # Debug: Check if markers are all NaN
    logging.info(f"Rising Star markers: {data_['RisingStar'].notna().sum()} non-NaN values")
    logging.info(f"Evening Star markers: {data_['EveningStar'].notna().sum()} non-NaN values")
    return data_

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
            logging.debug(f"Close on [{chart_data.index[0]}]={current_close}, [{data_.index[-1]}]={next_close}")

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
            data_.at[data_.index[i], LABEL_STAR] = UP_TREND
        elif ema30 > ema20 > ema10 > ema5:
            data_.at[data_.index[i], LABEL_STAR] = DOWN_TREND
        else:
            data_.at[data_.index[i], LABEL_STAR] = SIDEWAY_TREND    

    return data_

class MA4Star(TradingStrategy):
    def __init__(self, ticker: str='unknown'):
        super().__init__()
        self.ticker = ticker
        logging.info(f"Initialized MA4Star with ticker: {self.ticker}")     

    def prepare_data(self, data: pd.DataFrame, n: int = 100) -> pd.DataFrame:
        """Prepare the data in DataFrame including signals and labels."""
        if len(data) < 30:  # Ensure enough data for MA200
            logging.warning(f"Data length {len(data)} is less than 200 days, skipping signal generation")
            return None
        # Compute MAs on the full available data
        data = data.copy()
        
        data[LABEL_STAR] = None
        data['Labels'] = None
        data['RisingStar'] = None
        data['EveningStar'] = None
        # Convert to numeric type with np.nan for None
        data['RisingStar'] = pd.to_numeric(data['RisingStar'], errors='coerce')
        data['EveningStar'] = pd.to_numeric(data['EveningStar'], errors='coerce')
        # Calculate EMAs
        data = calculate_emas(data)
        # find Rising Star and Evening Star patterns
        data = find_RE_star(data, 0, n)
        data = calulate_Alltrend(data, 1, alpha)
        return data

    def generate_signals(self, data: pd.DataFrame, n: int = 100) -> pd.DataFrame:
        signals = pd.Series(0, index=data.index)
        if len(data) < n:  # Ensure enough data for MA200
            logging.warning(f"Data length {len(data)} is less than 200 days, skipping signal generation")
            return signals
        rdata = calculate_emas(data)
        rdata = rdata.rename(columns={'EMStrend': 'MA4Star_L'})
        return rdata

    def generate_signals_image(self, data: pd.DataFrame, n: int, chart_dir: str) -> pd.DataFrame:
        logging.info(f"generate_chart {self.ticker} with chart_dir={chart_dir}, n={n}")
        os.makedirs(chart_dir, exist_ok=True)

        data['predict_date'] = np.NAN
        data['chart_path'] = None
        predict_days = 1  # Number of days to predict after the chart window
        # Generate charts and detect patterns
        for i in range(len(data) - n - predict_days):
            lastdayptr = i + n
            chart_data = data.iloc[i:lastdayptr]
            # print(chart_data)
            if len(chart_data) != n:
                logging.info(f"Skipping window {i} due to insufficient or invalid data: {len(chart_data)} chart_data in {n} days")
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
                logging.debug(f"Adding Rising Star markers for window {i} is ")
                ap.append(mpf.make_addplot(rising_star_markers, type='scatter', marker='^', color='cyan', markersize=100))
            if evening_star_markers.notna().any():
                evening_star_markers = pd.to_numeric(evening_star_markers, errors='coerce')
                logging.debug(f"Adding Evening Star markers for window {i} is ")
                ap.append(mpf.make_addplot(evening_star_markers, type='scatter', marker='v', color='magenta', markersize=100))

            start_d = chart_data.index[0].strftime('%Y%m%d')
            end_d = chart_data.index[-1].strftime('%Y%m%d')
            predict_d = data.index[lastdayptr+1].strftime('%Y-%m-%d')
            chart_path = os.path.join(chart_dir, f"{self.ticker}_MA4Star_{start_d}_{end_d}_{i}.png")
            try:
                fig, ax = mpf.plot(chart_data, type='candle', addplot=ap, savefig=chart_path, warn_too_much_data=1000, returnfig=True)
                plt.close(fig)  # Close the figure to free memory
                plt.close('all')  # Close all figures to prevent memory accumulation
            except Exception as e:
                logging.error(f"Error generating chart {chart_path}: {e}")
                continue
            
            # Use pre-computed Label
            label = data.at[data.index[lastdayptr], 'Labels']
            if label is None:
                print(f"Skipping window {i} due to missing label")
                continue
            data.at[data.index[lastdayptr], "predict_date"] = predict_d
            data.at[data.index[lastdayptr], "chart_path"] = chart_path

        return data

