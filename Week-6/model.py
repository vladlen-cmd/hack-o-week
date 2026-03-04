import pandas as pd, numpy as np, pickle, os, warnings
warnings.filterwarnings('ignore')

def train():
    df = pd.read_csv('sports_data.csv')
    # Prepare sequences for LSTM: use past 24 hours to predict next hour
    values = df['energy_kwh'].values
    SEQ = 24
    X, y = [], []
    for i in range(SEQ, len(values)):
        X.append(values[i - SEQ:i])
        y.append(values[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)

    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping

        model = Sequential([
            LSTM(32, input_shape=(SEQ, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(16),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_tr, y_tr, epochs=20, batch_size=32, validation_split=0.1,
                  callbacks=[EarlyStopping(patience=3, restore_best_weights=True)], verbose=0)

        preds = model.predict(X_te, verbose=0).flatten()
        model.save('lstm_model.keras')
        engine = 'LSTM'
    except ImportError:
        # Fallback: simple sklearn if tensorflow not available
        from sklearn.linear_model import LinearRegression
        X_tr_2d, X_te_2d = X_tr.reshape(X_tr.shape[0], -1), X_te.reshape(X_te.shape[0], -1)
        model = LinearRegression(); model.fit(X_tr_2d, y_tr)
        preds = model.predict(X_te_2d)
        with open('lstm_model.pkl', 'wb') as f: pickle.dump(model, f)
        engine = 'LinearRegression-fallback'

    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    metrics = {'r2': round(r2_score(y_te, preds), 4), 'rmse': round(np.sqrt(mean_squared_error(y_te, preds)), 4),
               'mae': round(mean_absolute_error(y_te, preds), 4), 'engine': engine, 'seq_length': SEQ}

    with open('model_meta.pkl', 'wb') as f:
        pickle.dump({'metrics': metrics, 'engine': engine}, f)

    # Save day-type analysis
    day_analysis = {}
    for dt in df['day_type'].unique():
        sub = df[df['day_type'] == dt]
        day_analysis[dt] = {'avg': round(float(sub['energy_kwh'].mean()), 2),
                            'max': round(float(sub['energy_kwh'].max()), 2),
                            'count': len(sub)}
    with open('day_analysis.pkl', 'wb') as f: pickle.dump(day_analysis, f)

    print(f"{engine} | R²={metrics['r2']} | RMSE={metrics['rmse']}")
    return metrics

if __name__ == '__main__':
    train()
