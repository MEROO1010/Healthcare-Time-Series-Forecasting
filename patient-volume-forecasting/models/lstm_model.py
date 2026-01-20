import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler



def prepare_lstm_data(series, window_size=14):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i])
        y.append(scaled[i])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler


def train_lstm(X, y):
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(X.shape[1], 1)))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)

    return model