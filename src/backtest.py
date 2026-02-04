import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class BacktestResult:
    """Results from backtesting a strategy"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    
class BacktestEngine:
    """
    Backtesting engine for trading strategies
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []
        self.trades = []
        self.logger = logging.getLogger(__name__)
        
    def run_backtest(self, data: pd.DataFrame, strategy) -> BacktestResult:
        """
        Run backtest on historical data
        
        Args:
            data: DataFrame with OHLCV data
            strategy: Strategy object with generate_signals method
            
        Returns:
            BacktestResult with performance metrics
        """
        signals = strategy.generate_signals(data)
        
        for i, signal in enumerate(signals):
            if signal['action'] == 'BUY':
                self._open_position(signal, data.iloc[i])
            elif signal['action'] == 'SELL':
                self._close_position(signal, data.iloc[i])
        
        # Calculate metrics
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        win_rate = self._calculate_win_rate()
        sharpe = self._calculate_sharpe()
        max_dd = self._calculate_max_drawdown()
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            num_trades=len(self.trades)
        )
    
    def _open_position(self, signal, row):
        """Open a new position"""
        position = {
            'entry_price': row['close'],
            'size': signal.get('size', 0.1),
            'timestamp': row.name
        }
        self.positions.append(position)
        
    def _close_position(self, signal, row):
        """Close existing position"""
        if not self.positions:
            return
            
        position = self.positions.pop(0)
        pnl = (row['close'] - position['entry_price']) * position['size']
        self.balance += pnl
        
        self.trades.append({
            'entry': position['entry_price'],
            'exit': row['close'],
            'pnl': pnl,
            'timestamp': row.name
        })
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate percentage"""
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        return (wins / len(self.trades)) * 100
    
    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.trades) < 2:
            return 0.0
        returns = [t['pnl'] for t in self.trades]
        return np.mean(returns) / (np.std(returns) + 1e-9)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.trades:
            return 0.0
        peak = self.initial_balance
        max_dd = 0
        
        for trade in self.trades:
            self.balance += trade['pnl']
            if self.balance > peak:
                peak = self.balance
            dd = (peak - self.balance) / peak
            max_dd = max(max_dd, dd)
            
        return max_dd * 100
