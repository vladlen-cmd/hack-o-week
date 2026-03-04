import pandas as pd, numpy as np, pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

def train():
    df = pd.read_csv('campus_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.weekday

    features = ['hour', 'day_of_week', 'is_weekend', 'solar_generation_kwh', 'temperature']
    X, y = df[features].values, df['total_electricity_kwh'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model 1: Linear Regression
    lr = LinearRegression(); lr.fit(X_tr, y_tr)
    lr_preds = lr.predict(X_te)

    # Model 2: Exponential Smoothing on residuals
    lr_train_preds = lr.predict(X_tr)
    residuals = y_tr - lr_train_preds
    alpha = 0.3
    smoothed = np.zeros(len(X_te))
    last_smooth = np.mean(residuals[-24:])
    for i in range(len(smoothed)):
        smoothed[i] = alpha * (y_te[i] - lr_preds[i]) + (1 - alpha) * last_smooth if i > 0 else last_smooth
        last_smooth = smoothed[i]

    ensemble_preds = lr_preds + smoothed * 0.5

    r2_lr = round(r2_score(y_te, lr_preds), 4)
    r2_ens = round(r2_score(y_te, ensemble_preds), 4)
    rmse = round(np.sqrt(mean_squared_error(y_te, ensemble_preds)), 4)

    # Carbon KPIs
    total_energy = float(df['total_electricity_kwh'].sum())
    total_solar = float(df['solar_generation_kwh'].sum())
    co2_per_kwh = 0.82
    carbon_saved = round(total_solar * co2_per_kwh, 2)
    carbon_total = round(total_energy * co2_per_kwh, 2)

    bundle = {'lr_model': lr, 'alpha': alpha, 'features': features,
              'metrics': {'r2_lr': r2_lr, 'r2_ensemble': r2_ens, 'rmse': rmse},
              'carbon': {'saved_kg': carbon_saved, 'total_kg': carbon_total,
                         'pct_saved': round(carbon_saved / max(carbon_total, 1) * 100, 1),
                         'solar_total_kwh': round(total_solar, 1),
                         'energy_total_kwh': round(total_energy, 1)}}

    with open('model.pkl', 'wb') as f: pickle.dump(bundle, f)
    print(f"Ensemble (LR + EWM) | R²_LR={r2_lr} | R²_Ensemble={r2_ens}")
    print(f"Carbon saved: {carbon_saved} kg ({bundle['carbon']['pct_saved']}%)")
    return bundle

if __name__ == '__main__':
    train()
