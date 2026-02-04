import pandas as pd
import numpy as np

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, period=20, std_dev=2):
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

class TechnicalAnalysis:
    def __init__(self, df):
        self.df = df
        
    def add_all_indicators(self):
        self.df['RSI'] = calculate_rsi(self.df['close'])
        self.df['MACD'], self.df['Signal'], self.df['Histogram'] = calculate_macd(self.df['close'])
        self.df['BB_Upper'], self.df['BB_Middle'], self.df['BB_Lower'] = calculate_bollinger_bands(self.df['close'])
        self.df['SMA_20'] = self.df['close'].rolling(window=20).mean()
        self.df['SMA_50'] = self.df['close'].rolling(window=50).mean()
        return self.df
    
    def generate_signal(self):
        latest = self.df.iloc[-1]
        signals = []
        
        # RSI Signal
        if latest['RSI'] < 30:
            signals.append('BUY')
        elif latest['RSI'] > 70:
            signals.append('SELL')
        
        # MACD Signal
        if latest['MACD'] > latest['Signal']:
            signals.append('BUY')
        elif latest['MACD'] < latest['Signal']:
            signals.append('SELL')
        
        # Bollinger Bands
        if latest['close'] < latest['BB_Lower']:
            signals.append('BUY')
        elif latest['close'] > latest['BB_Upper']:
            signals.append('SELL')
        
        # Count signals
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        
        if buy_count > sell_count:
            return 'BUY', buy_count / len(signals)
        elif sell_count > buy_count:
            return 'SELL', sell_count / len(signals)
        else:
            return 'HOLD', 0.5
