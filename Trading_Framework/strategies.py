from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class TradingStrategy(ABC):
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, n: int = 100) -> pd.DataFrame:
        """Generate trading signals based on historical data."""
        pass

    @abstractmethod
    def generate_signals_image(self, data: pd.DataFrame, n: int, chart_dir: str) -> pd.DataFrame:
        """Generate signals image for deep learning model training."""
        pass

    @abstractmethod
    def prepare_data(self, data: pd.DataFrame, n: int = 100) -> pd.DataFrame:
        """Prepare the data in DataFrame including signals and labels."""
        pass