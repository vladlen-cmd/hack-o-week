import pandas as pd, numpy as np, pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

def train():
    df = pd.read_csv('cafeteria_data.csv')
    features = ['hour', 'day_of_week', 'is_weekend', 'temperature_c', 'weather', 'footfall']
    X, y = df[features].values, df['energy_kwh'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression(); model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    metrics = {'r2': round(r2_score(y_te, preds), 4), 'rmse': round(np.sqrt(mean_squared_error(y_te, preds)), 4),
               'mae': round(mean_absolute_error(y_te, preds), 4),
               'coefficients': {f: round(float(c), 4) for f, c in zip(features, model.coef_)}, 'features': features}
    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'metrics': metrics, 'features': features}, f)
    print(f"LinearRegression | R²={metrics['r2']} | RMSE={metrics['rmse']}")
    return model, metrics

if __name__ == '__main__':
    train()
