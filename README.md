# AI Trading Signals

> Generate cryptocurrency trading signals from technical indicators and an LSTM price model, with a backtesting engine for evaluating strategies.

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![CI](https://github.com/gnanam1990/ai-trading-signals/actions/workflows/ci.yml/badge.svg)

## Overview

AI Trading Signals is a Python toolkit for experimenting with rule-based and
machine-learning trading signals on cryptocurrency price data. It combines a set
of classic technical indicators with an LSTM price-prediction model and a
backtesting engine for measuring strategy performance on historical data.

It is intended for developers and researchers exploring quantitative trading
ideas. It is an educational/research project, not financial advice and not a
turnkey live-trading bot — see [Status](#status) for what is and isn't wired up.

## Features

- Technical indicators: RSI, MACD, Bollinger Bands, and 20/50-period simple
  moving averages (`src/indicators.py`).
- Rule-based signal generation that combines indicator votes into a
  `BUY` / `SELL` / `HOLD` decision with a confidence score.
- LSTM price-prediction model built on Keras/TensorFlow, with helpers for data
  scaling, training, and prediction (`src/lstm_model.py`).
- Backtesting engine that runs a strategy over OHLCV data and reports total
  return, Sharpe ratio, max drawdown, win rate, and trade count
  (`src/backtest.py`).
- Unit tests for indicators and model construction.

## Tech stack

- Python 3.8+
- NumPy, pandas — data handling and indicator math
- scikit-learn — feature scaling (`MinMaxScaler`)
- TensorFlow / Keras — LSTM model
- python-binance, requests — market data / API access (declared dependencies)
- python-dotenv — environment configuration
- matplotlib, seaborn — plotting
- pytest, flake8 — testing and linting (CI)

## Architecture

- `src/indicators.py` — indicator functions and a `TechnicalAnalysis` class that
  computes indicators and produces a combined signal.
- `src/lstm_model.py` — `LSTMPredictor` for building, training, and predicting
  with the LSTM network.
- `src/backtest.py` — `BacktestEngine` and a `BacktestResult` dataclass that
  evaluate a strategy and compute performance metrics.
- `src/main.py` — `TradingBot` entry point that wires technical analysis (and the
  LSTM predictor) together.

## Getting started

### Prerequisites

- Python 3.8 or newer
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and fill in your values. The following variables
are defined for configuring data access, trading parameters, and the model.
Note that environment loading is not yet wired into the entry point (see
[Status](#status)); these names document the intended configuration surface.

| Variable | Purpose |
| --- | --- |
| `BINANCE_API_KEY` | Exchange API key for market data / trading |
| `BINANCE_SECRET_KEY` | Exchange API secret |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token for notifications (optional) |
| `TELEGRAM_CHAT_ID` | Telegram chat ID for notifications (optional) |
| `INITIAL_BALANCE` | Starting balance for backtests |
| `MAX_POSITION_SIZE` | Maximum position size as a fraction of balance |
| `LEVERAGE` | Leverage multiplier |
| `STOP_LOSS_PERCENT` | Stop-loss threshold (percent) |
| `TAKE_PROFIT_PERCENT` | Take-profit threshold (percent) |
| `LSTM_LOOKBACK` | Sequence length / lookback window for the LSTM |
| `PREDICTION_THRESHOLD` | Confidence threshold for acting on predictions |
| `USE_BINANCE` | Toggle Binance as a data source |
| `USE_HYPERLIQUID` | Toggle Hyperliquid as a data source |

Never commit your `.env` file or real secret values.

### Running

```bash
python -m src.main
```

This initializes the `TradingBot` and logs startup. The continuous monitoring
loop is currently a stub (see [Status](#status)).

## Usage

Compute indicators and generate a signal from a price DataFrame:

```python
import pandas as pd
from src.indicators import TechnicalAnalysis

df = pd.DataFrame({"close": [...]})  # OHLCV data with a 'close' column
ta = TechnicalAnalysis(df)
ta.add_all_indicators()
signal, confidence = ta.generate_signal()
print(signal, confidence)  # e.g. "BUY", 0.66
```

Train and predict with the LSTM model:

```python
import numpy as np
from src.lstm_model import LSTMPredictor

predictor = LSTMPredictor(sequence_length=60)
predictor.train(price_array, epochs=50, batch_size=32)
next_price = predictor.predict(recent_price_array)
```

Run a backtest with a strategy object that exposes `generate_signals(data)`:

```python
from src.backtest import BacktestEngine

engine = BacktestEngine(initial_balance=10000.0)
result = engine.run_backtest(data, strategy)
print(result.total_return, result.sharpe_ratio, result.max_drawdown)
```

A concrete strategy class is not bundled; supply your own object implementing
`generate_signals(data)` that returns a list of `{"action": "BUY"|"SELL", ...}`
dicts.

## Testing

```bash
python -m pytest tests/ -v
```

Tests cover indicator calculations (`TechnicalAnalysis`, `calculate_rsi`) and
LSTM model construction. CI also runs a flake8 lint pass over `src/`.

## Project structure

```
ai-trading-signals/
├── src/
│   ├── main.py          # TradingBot entry point
│   ├── indicators.py    # Technical indicators + signal generation
│   ├── lstm_model.py    # LSTM price predictor
│   └── backtest.py      # Backtesting engine
├── tests/
│   └── test_indicators.py
├── .env.example
├── requirements.txt
├── setup.cfg
└── .github/workflows/ci.yml
```

## Status

Early-stage / research scaffold. What is real today:

- Technical indicators and rule-based signal generation are implemented and
  tested.
- The LSTM model can be built, trained, and used for prediction, but it is not
  yet connected to the live signal flow in `src/main.py`.
- The backtesting engine is implemented, but it requires a user-supplied
  strategy object; no concrete strategy ships in the repo.

Not yet implemented:

- The main monitoring loop in `src/main.py` is a stub
  ("Main loop would go here") — running it only logs startup.
- Environment-variable loading, exchange/data-source connectivity, and Telegram
  notifications are not wired into the code despite the dependencies and config
  placeholders being present.
- Sentiment analysis is not present in this codebase.

This software is for educational and research purposes only and is not financial
advice. Trade at your own risk.

## License

MIT — see [LICENSE](LICENSE).
