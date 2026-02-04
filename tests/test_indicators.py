import unittest
import numpy as np
import pandas as pd
from src.indicators import TechnicalAnalysis, calculate_rsi
from src.lstm_model import LSTMPredictor

class TestIndicators(unittest.TestCase):
    def test_rsi_calculation(self):
        data = pd.Series([100, 101, 102, 101, 100, 99, 98, 99, 100, 101] * 5)
        rsi = calculate_rsi(data)
        self.assertFalse(rsi.isna().all())
    
    def test_technical_analysis(self):
        df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109] * 10
        })
        ta = TechnicalAnalysis(df)
        result = ta.add_all_indicators()
        self.assertIn('RSI', result.columns)
        self.assertIn('MACD', result.columns)

class TestLSTM(unittest.TestCase):
    def test_model_build(self):
        model = LSTMPredictor()
        model.build_model((60, 1))
        self.assertIsNotNone(model.model)

if __name__ == '__main__':
    unittest.main()
