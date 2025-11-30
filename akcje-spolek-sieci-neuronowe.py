import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import streamlit as st 
import datetime

WINDOW_SIZE = 40  
TRAIN_START_DATE = "2015-01-01"
TRAIN_END_DATE = "2021-01-01"
ANOMALY_THRESHOLD_PERCENTILE = 95 

def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:(i + window_size)])
    return np.array(sequences)

def create_forecasting_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size): 
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])     
    return np.array(X), np.array(y)

@st.cache_resource
def train_autoencoder(X_train):
    input_dim = X_train.shape[1] 
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(input_dim, activation='sigmoid') 
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, X_train, epochs=40, batch_size=32, verbose=0, validation_split=0.1)
    return model

@st.cache_resource
def train_lstm(X_train, y_train):
    time_steps = X_train.shape[1]
    
    model = keras.Sequential([
        layers.Input(shape=(time_steps, 1)),
        layers.LSTM(50, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=0, validation_split=0.1)
    return model

st.title("🧠 Hybrydowy Model AI: Wykrywanie Anomalii + Prognoza")
st.markdown("""
Ta aplikacja działa w dwóch krokach:
1. **Autoenkoder** skanuje rynek i wykrywa **anomalie** (nietypowe zachowania cen).
2. **LSTM** uczy się trendu, **ignorując** wykryte anomalie (szum), aby dać lepszą prognozę.
""")

ticker = st.text_input("Podaj ticker (np. NVDA, TSLA, BTC-USD):", "NVDA").upper()
forecast_days = st.slider("Dni do prognozy:", 30, 365, 90)

if st.button("Analizuj i Prognozuj"):
    with st.spinner(f"Pobieranie i przetwarzanie danych dla {ticker}..."):
        
        try:
            raw_data = yf.download(ticker, start=TRAIN_START_DATE, auto_adjust=True)
            if raw_data.empty:
                st.error("Brak danych. Sprawdź ticker.")
                st.stop()
                
            if isinstance(raw_data.columns, pd.MultiIndex):
                try:
                    df = raw_data.xs(ticker, axis=1, level=1)
                except KeyError:
                    df = raw_data
                if 'Close' not in df.columns:
                     df = df.iloc[:, 0].to_frame(name='Close')
                else:
                    df = df[['Close']]
            else:
                if 'Close' in raw_data.columns:
                    df = raw_data[['Close']]
                else:
                    df = raw_data.iloc[:, 0].to_frame(name='Close')

        except Exception as e:
            st.error(f"Błąd pobierania danych: {e}")
            st.stop()

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(df.values)
        
        st.info("KROK 1: Autoenkoder szuka anomalii w danych historycznych...")
        
        X_ae = create_sequences(data_scaled.flatten(), WINDOW_SIZE)
        
        autoencoder = train_autoencoder(X_ae)
        
        reconstructions = autoencoder.predict(X_ae, verbose=0)
        mae = np.mean(np.abs(reconstructions - X_ae), axis=1) 
        
        threshold = np.percentile(mae, ANOMALY_THRESHOLD_PERCENTILE)
        
        anomaly_indices = np.where(mae > threshold)[0] + WINDOW_SIZE
        
        clean_df = df.copy()
        clean_df['Is_Anomaly'] = False
        clean_df.iloc[anomaly_indices, clean_df.columns.get_loc('Is_Anomaly')] = True
        
        anomalies = clean_df[clean_df['Is_Anomaly']]
        st.write(f"🔍 Znaleziono **{len(anomalies)}** anomalii (dziwnych zachowań rynku).")
        
        clean_data_values = df['Close'].values.copy()
        for idx in anomaly_indices:
            clean_data_values[idx] = clean_data_values[idx-1]
            
        clean_scaled = scaler.fit_transform(clean_data_values.reshape(-1, 1))

        st.info("KROK 2: LSTM uczy się na oczyszczonych danych i generuje prognozę...")
        
        X_lstm, y_lstm = create_forecasting_sequences(clean_scaled, WINDOW_SIZE)
        
        X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)
        
        lstm_model = train_lstm(X_lstm, y_lstm)
        
        last_window = clean_scaled[-WINDOW_SIZE:]
        forecast = []
        
        for _ in range(forecast_days):
            pred_input = last_window.reshape(1, WINDOW_SIZE, 1)
            pred_val = lstm_model.predict(pred_input, verbose=0)[0, 0]
            forecast.append(pred_val)
            last_window = np.append(last_window[1:], pred_val)
            
        forecast_usd = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
        
        last_date = df.index[-1]
        forecast_dates = pd.date_range(start=last_date + datetime.timedelta(days=1), periods=forecast_days, freq='B')
        
        forecast_df = pd.DataFrame({'Prognoza': forecast_usd}, index=forecast_dates)
        
        st.subheader(f"Wynik Analizy Hybrydowej dla {ticker}")
        
        fig = plt.figure(figsize=(16, 8))
        
        plt.plot(df.index, df['Close'], label='Cena Rzeczywista', color='blue', alpha=0.6)
        
        plt.scatter(anomalies.index, anomalies['Close'], color='red', s=30, label='Wykryte Anomalie (Ignorowane przez LSTM)', zorder=5)
        
        plt.plot(forecast_df.index, forecast_df['Prognoza'], label='Prognoza LSTM', color='lime', linewidth=2, linestyle='--')
        
        plt.title(f"Detekcja Anomalii i Prognoza Ceny: {ticker}")
        plt.xlabel("Data")
        plt.ylabel("Cena USD")
        plt.legend()
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.subheader("Dane prognozowane")
        result_series = pd.concat([df['Close'].tail(5), forecast_df['Prognoza'].head(10)])
        result_df = pd.DataFrame(result_series)
        result_df.columns = ['Cena']
        st.dataframe(result_df)