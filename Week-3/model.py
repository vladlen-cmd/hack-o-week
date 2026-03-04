import pandas as pd, numpy as np, pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

def train():
    df = pd.read_csv('library_data.csv')
    features = ['hour', 'day_of_week', 'is_weekend', 'is_exam', 'temperature_c', 'weather', 'occupancy']
    X = df[features].values
    y = df['energy_kwh'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = round(r2_score(y_test, preds), 4)
    rmse = round(np.sqrt(mean_squared_error(y_test, preds)), 4)
    mae = round(mean_absolute_error(y_test, preds), 4)
    coefficients = {f: round(float(c), 4) for f, c in zip(features, model.coef_)}

    metrics = {'r2': r2, 'rmse': rmse, 'mae': mae, 'intercept': round(float(model.intercept_), 4),
               'coefficients': coefficients, 'features': features}

    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'metrics': metrics, 'features': features}, f)

    print(f"LinearRegression | R²={r2} | RMSE={rmse} | MAE={mae}")
    print(f"Coefficients: {coefficients}")
    return model, metrics

if __name__ == '__main__':
    train()
