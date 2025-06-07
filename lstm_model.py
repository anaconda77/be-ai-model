from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

scaler = MinMaxScaler()
SEQ_LEN = 10

def preprocessing(df):
    features = ['sentiment_influence', 'open', 'high', 'low', 'adjClose', 'volume']
    target   = 'close'
    df = df[ features + [target] ].dropna()
    return df
    
def get_train_scale_data(df):
    df = preprocessing(df)
    return scaler.fit_transform(df)

def get_test_scale_data(df):
    df = preprocessing(df)
    return scaler.transform(df)

def create_sequences(arr):

    X, y = [], []
    for i in range(len(arr) - SEQ_LEN):
        X.append(arr[i:i+SEQ_LEN, :-1])   # 모든 피처
        y.append(arr[i+SEQ_LEN, -1])      # Close만
    
    return np.array(X), np.array(y)

def compile_model(X_train):
    model = Sequential([
        LSTM(64, input_shape=(SEQ_LEN, X_train.shape[2])),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test)
    )
    
    return model


def predict_prices(model, X_test):
    return model.predict(X_test)