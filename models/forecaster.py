import numpy as np
import tensorflow as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

def create_bilstm_model(input_shape, lstm_units=64, dropout_rate=0.2):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(lstm_units, return_sequences=True)),
        Dropout(dropout_rate),
        Bidirectional(LSTM(lstm_units // 2)),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_forecaster(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    os.makedirs('models', exist_ok=True)
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    print("Building Bi-LSTM Model...")
    model = create_bilstm_model(input_shape)
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('models/bilstm_model.keras', monitor='val_loss', save_best_only=True)
    
    print("Training Bi-LSTM Model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    return model, history

if __name__ == "__main__":
    # Test script locally
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline.etl import ETLPipeline
    
    pipeline = ETLPipeline('../iiot_smart_grid_dataset.csv')
    try:
        df = pipeline.load_data()
        df = pipeline.feature_engineering(df)
        X_train, X_test, y_train, y_test, _ = pipeline.prepare_data(df)
        
        # We can split a simple validation set
        val_size = int(len(X_train) * 0.1)
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        X_train_sub, y_train_sub = X_train[:-val_size], y_train[:-val_size]
        
        train_forecaster(X_train_sub, y_train_sub, X_val, y_val, epochs=10) # 10 epochs for quick local test
    except FileNotFoundError:
        print("Please ensure iiot_smart_grid_dataset.csv is in the parent directory.")
