import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

SEQUENCE_LENGTH = 10

# ==============================
# Technical Indicators
# ==============================
def add_technical_indicators(df):
    df = df.copy()
    # Make sure ADX computation returns Series
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['ADX'] = compute_adx(df, 14)  # always a Series
    macd_line, signal_line = compute_macd(df['Close'])
    df['MACD'] = macd_line
    df['MACD_signal'] = signal_line
    return df

def compute_adx(df, window=14):
    high = df['High']
    low = df['Low']
    close = df['Close']

    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()

    plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
    minus_di = abs(100 * (minus_dm.rolling(window).mean() / atr))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=window).mean()

    # Make sure it's a Series with the same index
    adx = pd.Series(adx, index=df.index, name='ADX')
    return adx

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(close, span_short=12, span_long=26, signal_span=9):
    exp1 = close.ewm(span=span_short, adjust=False).mean()
    exp2 = close.ewm(span=span_long, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
    return macd_line, signal_line


# ==============================
# Sequence Creation
# ==============================
def create_sequences(data, target, seq_len=SEQUENCE_LENGTH):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(target[i + seq_len])
    return np.array(X), np.array(y)

# ==============================
# Train LSTM
# ==============================
def train_lstm_model(df):
    df = add_technical_indicators(df)
    df['Target'] = df['Close'].shift(-1) / df['Close'] - 1
    df = df.dropna()

    features = ['Close', 'SMA20', 'SMA50', 'EMA20', 'EMA50', 'RSI', 'ADX', 'MACD', 'MACD_signal']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    target = df['Target'].values

    X, y = create_sequences(scaled_data, target)

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    if not os.path.exists("models"):
        os.makedirs("models")
    model.save("models/lstm_model.h5")
    with open("models/lstm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return model, scaler

# ==============================
# Predict Next-Day Return
# ==============================
def predict_next_lstm_return(model, scaler, df):
    df = add_technical_indicators(df)
    features = ['Close', 'SMA20', 'SMA50', 'EMA20', 'EMA50', 'RSI', 'ADX', 'MACD', 'MACD_signal']
    scaled_data = scaler.transform(df[features])
    last_sequence = scaled_data[-SEQUENCE_LENGTH:]
    X_input = np.expand_dims(last_sequence, axis=0)
    pred = model.predict(X_input, verbose=0)[0][0]
    return pred
