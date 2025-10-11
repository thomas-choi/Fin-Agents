import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import tensorflow as tf
from PIL import Image
import logging
from strategies import TradingStrategy  # Import from separate strategies module
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dotenv import load_dotenv

load_dotenv()

# Configure logging - simple INFO level
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)


EXPORT_BASE = os.getenv("EXPORT_BASE", "./export")

# --- Data Handler ---
class DataHandler:
    def __init__(self, cache_dir: str = 'data'):
        self.cache_dir = os.path.join(EXPORT_BASE, cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        logging.debug(f" init {self.cache_dir}")

    def fetch_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data from yfinance with caching as CSV."""
        os.makedirs(self.cache_dir, exist_ok=True)
        logging.debug(f" init {self.cache_dir} for fetch_data ")
        cache_file = os.path.join(self.cache_dir, f"{ticker}_{start_date}_{end_date}.csv")
        logging.debug(f"Checking cache file {cache_file}")
        if os.path.exists(cache_file):
            logging.info(f"Loading cached data for {ticker} from {cache_file}")
            data = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
            return data
        
        logging.info(f"Downloading data for {ticker} from {start_date} to {end_date}")
        data = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False)
        if data.empty:
            raise ValueError(f"No data retrieved for {ticker}")
        
        # Flatten MultiIndex columns to single level (e.g., 'Close' instead of ('Close', 'AAPL'))
        # if isinstance(data.columns, pd.MultiIndex):
        #     data.columns = [col[0] for col in data.columns]
        
        data.to_csv(cache_file)
        logging.debug(f"Saved data to {cache_file}")
        return data

# --- Deep Learning Models ---
class DLModel:
    def __init__(self, model_type: str, input_shape: tuple):
        self.model_type = model_type
        self.model = self._build_model(input_shape)

    def _build_model(self, input_shape: tuple):
        """Build a deep learning model for image inputs."""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(1, activation='tanh')
        ])
        model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
        return model

    def train_with_generator(self, generator, steps_per_epoch, epochs: int = 50):
        """Train the deep learning model using a generator."""
        self.model.fit(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1)

    def predict_with_generator(self, generator, steps) -> np.ndarray:
        """Generate predictions using a generator."""
        preds = self.model.predict(generator, steps=steps)
        return np.sign(preds).flatten()

    def save(self, ticker: str, strategy_name: str, epochs: int, batch_size: int, lookback: int):
        """Save the trained model to disk with training parameters in filename."""
        model_dir = os.path.join(EXPORT_BASE, './models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{ticker}_{strategy_name}_{self.model_type}_epochs{epochs}_batch{batch_size}_lookback{lookback}.h5")
        save_model(self.model, model_path)
        logging.info(f"Saved model to {model_path}")

    def load(self, ticker: str, strategy_name: str, epochs: int, batch_size: int, lookback: int):
        """Load an existing model from disk."""
        model_dir = os.path.join(EXPORT_BASE, './models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{ticker}_{strategy_name}_{self.model_type}_epochs{epochs}_batch{batch_size}_lookback{lookback}.h5")
        if os.path.exists(model_path):
            try:
                self.model = load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
                logging.info(f"Loaded model from {model_path}")
                return True
            except Exception as e:
                logging.error(f"Error loading model from {model_path}: {e}")
                return False
        return False

# --- Backtest Module ---
class Backtest:
    def __init__(self, ticker: str, data: pd.DataFrame, strategies: list, n: int = 100, start_date: str = None, end_date: str = None):
        self.data = data
        self.ticker = ticker
        self.strategies = strategies
        self.n = n
        self.start_date = start_date
        self.end_date = end_date
        # Initialize results with sliced index if date range is provided
        if start_date and end_date:
            self.results = pd.DataFrame(index=data.loc[start_date:end_date].index)
        else:
            self.results = pd.DataFrame(index=data.index)

    def run(self):
        """Run backtest for all strategies, optionally within date range."""
        logging.debug(f"Running backtest for strategies: {[s.__class__.__name__ for s in self.strategies]} on data from {self.results.index[0]} to {self.results.index[-1]}")
        for strategy in self.strategies:
            signals_df = strategy.generate_signals(self.data, self.n)
            if self.start_date and self.end_date:
                signals_df = signals_df.loc[self.start_date:self.end_date]
            col = strategy.__class__.__name__
            if col == 'DLStrategy':
                logging.info(f"Using DLStrategy for {strategy.ticker} with model type {strategy.model_type}")
                signals = signals_df['predicts'].dropna()
                strategy.save_preprocessed_data(signals_df)
            else:
                signals = signals_df[f'{col}_L'].dropna()
            returns = self._calculate_returns(signals)
            self.results[f'{strategy.__class__.__name__}_Signal'] = signals
            self.results[f'{strategy.__class__.__name__}_Returns'] = returns
        self.results[f'{self.ticker}_Signal'] = 1.0  # Assuming a buy-and-hold strategy for the ticker
        self.results[f'{self.ticker}_Returns'] = self.data['Close'].pct_change().shift(-1).loc[self.start_date:self.end_date]

    def _calculate_returns(self, signals: pd.Series) -> pd.Series:
        """Calculate strategy returns based on signals."""
        returns = self.data['Close'].pct_change().shift(-1)  # Next day's return
        strategy_returns = signals * returns
        if self.start_date and self.end_date:
            strategy_returns = strategy_returns.loc[self.start_date:self.end_date]
        return strategy_returns

    def get_results(self) -> pd.DataFrame:
        """Return backtest results."""
        return self.results

# --- Deep Learning Strategy ---
class DLStrategy(TradingStrategy):
    def __init__(self, model_type: str, base_strategy: TradingStrategy, ticker: str, lookback: int = 100, epochs: int = 50, batch_size: int = 32):
        self.model_type = model_type
        self.base_strategy = base_strategy
        self.ticker = ticker
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.image_size = (224, 224)  # Standard size for CNN model input

    def save_preprocessed_data(self, data: pd.DataFrame):
        """Save preprocessed data to CSV for later use."""
        pre_data_dir = os.path.join(EXPORT_BASE, './preprocessdata')
        os.makedirs(pre_data_dir, exist_ok=True)
        start_date = data.index[0].strftime('%Y-%m-%d')
        end_date = data.index[-1].strftime('%Y-%m-%d')
        ppdata_opath = os.path.join(pre_data_dir, f"{self.ticker}_{self.base_strategy.__class__.__name__}_{start_date}_{end_date}.csv")
        data.to_csv(ppdata_opath, index=True)
        logging.info(f"Preprocessed data saved to {ppdata_opath}")  

    def prepare_data(self, data: pd.DataFrame, n: int = 100) -> pd.DataFrame:
        """Prepare the data in DataFrame including signals and labels."""
        return self.preprocess_data(data)

    def get_data_generator(self, data: pd.DataFrame, batch_size: int, is_train: bool = True, for_predict: bool = False):
        """Create a data generator for training or prediction."""
        print(f"Creating data generator... batch_size={batch_size}, is_train={is_train}, for_predict={for_predict}")
        print(data.head(2))
        dumpdf = data.dropna(subset=['chart_path']).reset_index()
        print(f"DataFrame after dropping NaNs in 'chart_path':")
        print(dumpdf.head(2))
        # Reset index and name it 'original_index' to avoid conflicts
        df = data.dropna(subset=['chart_path'])
        print(df.head(2))
        orig_indices = df.index.to_list()
        print(f"DataFrame size: {df.shape}")
        print(f"number of original indices: {len(orig_indices)}")
        df = df.reset_index()
        if not for_predict and 'Labels' not in df.columns:
            raise ValueError("Labels column missing for training")
        datagen = ImageDataGenerator(rescale=1./255)
        y_col = 'Labels' if not for_predict else None
        class_mode = 'raw' if not for_predict else None
        shuffle = is_train and not for_predict
        generator = datagen.flow_from_dataframe(
            dataframe=df,
            x_col='chart_path',
            y_col=y_col,
            target_size=self.image_size,
            color_mode='rgb',
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=42
        )
        return generator, orig_indices, len(df)

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Pre-process data for deep learning model, ensuring it has enough historical data."""
        if len(data) < self.lookback:
            logging.warning(f"Insufficient data for {self.ticker}: {len(data)} days available, need at least {self.lookback}")
            return pd.DataFrame()
        
        preprocess_data = self.base_strategy.prepare_data(data, self.lookback)
        preprocess_data = self.base_strategy.generate_signals_image(preprocess_data, self.lookback, chart_dir=os.path.join(EXPORT_BASE, 'chartdata'))
        self.save_preprocessed_data(preprocess_data)
        return preprocess_data

    def generate_signals(self, data: pd.DataFrame, n: int) -> pd.DataFrame:
        """Generate signals using the trained DL model, or fall back to base strategy."""
        if self.model is None:
            logging.warning(f"{self.model_type} model not trained for {self.ticker}, using base strategy signals")
            return self.base_strategy.generate_signals(data, self.lookback)
        infer_batch_size = 32
        gen, orig_indices, num_samples = self.get_data_generator(data, infer_batch_size, is_train=False, for_predict=True)
        if num_samples == 0:
            logging.warning(f"No valid image data for {self.ticker}, using base strategy signals")
            return self.base_strategy.generate_signals(data, self.lookback)
        steps = (num_samples + infer_batch_size - 1) // infer_batch_size  # Ceiling division
        predictions = self.model.predict_with_generator(gen, steps)
        data['predicts'] = np.NaN
        data.loc[orig_indices, 'predicts'] = predictions
        return data

    def generate_signals_image(self, data: pd.DataFrame, n: int, chart_dir: str) -> pd.DataFrame:
        """Delegate signals image generation to the base strategy with ticker."""
        return self.base_strategy.generate_signals_image(data, n, chart_dir=chart_dir)

    def train(self, data: pd.DataFrame, ticker: str):
        """Train the deep learning model or load if it exists, using training period data."""
        # Slice data to training period
        train_data = data.loc['2020-01-01':'2023-12-31']
        if len(train_data) < 200:
            logging.warning(f"Training data for {ticker} too short ({len(train_data)} days), skipping training")
            return
        self.model = DLModel(self.model_type, input_shape=(*self.image_size, 3))
        if self.model.load(ticker, self.base_strategy.__class__.__name__, self.epochs, self.batch_size, self.lookback):
            return  # Model loaded, skip training
        logging.info(f"Training {self.model_type} model for {ticker} with {self.base_strategy.__class__.__name__}")
        gen, _, num_samples = self.get_data_generator(train_data, self.batch_size, is_train=True, for_predict=False)
        if num_samples == 0:
            logging.warning(f"Skipping training for {ticker}: No valid image data")
            return
        steps = num_samples // self.batch_size
        if steps == 0:
            logging.warning(f"Batch size too large or insufficient samples for training {ticker}")
            return
        self.model.train_with_generator(gen, steps, self.epochs)
        self.model.save(ticker, f"{self.base_strategy.__class__.__name__}", self.epochs, self.batch_size, self.lookback)

# --- Dashboard ---
class TradingDashboard:
    def __init__(self, backtest_results: pd.DataFrame, height: int = 600, width: int = 800):
        self.results = backtest_results
        self.height = height
        self.width = width

    def display(self, pdf_output: str = None):
        """Display backtest results as a table and plot, with option to save to PDF."""
        # Display numerical results
        print("Backtest Results Summary:")
        summary = pd.DataFrame({
            'Strategy': [col.replace('_Returns', '') for col in self.results.columns if 'Returns' in col],
            'Total Return': [self.results[col].sum() for col in self.results.columns if 'Returns' in col],
            'Sharpe Ratio': [(self.results[col].mean() / self.results[col].std() * np.sqrt(252))
                             for col in self.results.columns if 'Returns' in col]
        })
        print(summary)

        # Plot cumulative returns
        fig = make_subplots(rows=1, cols=1, subplot_titles=['Cumulative Returns'])
        for col in self.results.columns:
            if 'Returns' in col:
                strategy_name = col.replace('_Returns', '')
                if strategy_name.endswith('_Signal'):
                    continue
                # Use descriptive legend names
                if strategy_name == 'DLStrategy':
                    legend_name = f"{strategy_name} (CNN)"
                elif strategy_name == self.results.columns[-1].replace('_Returns', ''):
                    legend_name = f"{strategy_name} (Buy & Hold)"
                else:
                    legend_name = strategy_name
                cum_returns = (1 + self.results[col]).cumprod() - 1
                fig.add_trace(
                    go.Scatter(x=self.results.index, y=cum_returns, name=legend_name),
                    row=1, col=1
                )
        fig.update_layout(
            title='Backtest Cumulative Returns',
            height=self.height,
            width=self.width,
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            legend_title='Strategies',
            showlegend=True
        )
        
        if pdf_output:
            # Ensure the export directory exists
            export_dir = os.path.join(IMPORT_BASE, 'plots')
            os.makedirs(export_dir, exist_ok=True)
            pdf_path = os.path.join(export_dir, pdf_output)
            fig.write_image(pdf_path, format='pdf')
            logging.info(f"Saved plot to {pdf_path}")
        
        fig.show()

# --- Trading Framework ---
class TradingFramework:
    def __init__(self, ticker: str, strategy: TradingStrategy, start_date: str = '2020-01-01', end_date: str = '2023-12-31',
                 validation_start: str = '2024-01-02', validation_end: str = '2025-07-15', n: int = 100):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.validation_start = validation_start
        self.validation_end = validation_end
        self.n = n
        self.data_handler = DataHandler()
        self.strategies = [strategy]
        self.dl_strategies = [
            DLStrategy('CNN', strategy, ticker, lookback=100, epochs=50, batch_size=32),
        ]
        self.preprocessdata = dict()

    def fetch_data(self):
        """Fetch historical data for the entire period."""
        logging.debug(f"Fetching data for {self.ticker} from {self.start_date} to {self.validation_end}")
        self.data = self.data_handler.fetch_data(self.ticker, self.start_date, self.validation_end)

    def train_dl_models(self):
        """Train deep learning models or load existing ones."""
        for dl_strategy in self.dl_strategies:
            if dl_strategy.base_strategy.__class__.__name__ in self.preprocessdata:
                logging.info(f"Using preprocessed data for {dl_strategy.base_strategy.__class__.__name__}") 
                in_data = self.preprocessdata[dl_strategy.base_strategy.__class__.__name__]
                dl_strategy.train(in_data, self.ticker)
                self.strategies.append(dl_strategy)
            else:
                logging.warning(f"No preprocessed data available for {dl_strategy.base_strategy.__class__.__name__}, skipping training")

    def preprocess_data(self):
        for dl_strategy in self.dl_strategies:
            pdata = dl_strategy.preprocess_data(self.data)
            self.preprocessdata[dl_strategy.base_strategy.__class__.__name__] = pdata
        return self.preprocessdata

    def run_backtest(self):
        """Run backtest for all strategies on validation period only."""
        logging.info(f"Running backtest on validation data ({self.validation_start} to {self.validation_end})")
        pdata = self.preprocessdata[self.strategies[0].__class__.__name__]
        self.validation_backtest = Backtest(self.ticker, pdata, self.strategies, self.n, self.validation_start, self.validation_end)
        self.validation_backtest.run()

    def display_results(self, height: int = 600, width: int = 800):
        """Display backtest results."""
        logging.debug("Displaying validation period results (2024-01-02 to 2025-07-15)")
        validation_dashboard = TradingDashboard(self.validation_backtest.get_results(), height=height, width=width)
        print("\nValidation Period Results (2024-01-02 to 2025-07-15):")
        validation_dashboard.display()

# --- Example Usage ---
if __name__ == "__main__":
    from MAStrategy import MAStrategy  # Import the moving average strategy 

    # Initialize framework
    framework = TradingFramework(ticker='AAPL', strategy=MAStrategy(ticker='AAPL'))
    
    # Fetch data
    framework.fetch_data()
    
    # Train deep learning models
    framework.train_dl_models()
    
    # Run backtest
    framework.run_backtest()
    
    # Display results
    framework.display_results()