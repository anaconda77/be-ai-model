from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

SEQ_LEN = 7

def get_scale_data_with_fit(scaler, df):
    return scaler.fit_transform(df)

def get_scale_data(scaler, df):
    return scaler.transform(df)

def create_sequences_for_train(arr):

    X, y = [], []
    for i in range(len(arr) - SEQ_LEN):
        X.append(arr[i:i+SEQ_LEN, :-1])   # 모든 피처
        y.append(arr[i+SEQ_LEN, -1])      # Close만
    
    return np.array(X), np.array(y)
    
def create_sequences_for_prod(arr):

    return np.array([arr[:, :-1]]) 

def compile_model(X_train):
    model = Sequential([
        LSTM(64, 
             activation='tanh', 
             recurrent_activation='sigmoid',  # 표준 LSTM의 기본값
             input_shape=(SEQ_LEN, X_train.shape[2])),
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