import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMPredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        
    def build_model(self, input_shape):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        logger.info("LSTM model built successfully")
        return model
    
    def prepare_data(self, data):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        x_train, y_train = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            x_train.append(scaled_data[i-self.sequence_length:i, 0])
            y_train.append(scaled_data[i, 0])
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        return x_train, y_train
    
    def train(self, data, epochs=50, batch_size=32):
        x_train, y_train = self.prepare_data(data)
        
        if self.model is None:
            self.build_model((x_train.shape[1], 1))
        
        logger.info(f"Training model with {len(x_train)} samples")
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        logger.info("Model training completed")
    
    def predict(self, recent_data):
        scaled_data = self.scaler.transform(recent_data.reshape(-1, 1))
        x_test = []
        x_test.append(scaled_data[-self.sequence_length:, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        prediction = self.model.predict(x_test)
        return self.scaler.inverse_transform(prediction)[0, 0]
