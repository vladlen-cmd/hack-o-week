import pandas as pd, numpy as np, pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

def train():
    df = pd.read_csv('hvac_data.csv')
    features = ['hour', 'day_of_week', 'is_weekend', 'occupancy', 'outdoor_temp_c', 'equipment_heat_w', 'zone']
    X, y = df[features].values, df['cooling_kwh'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    fi = {f: round(float(v), 4) for f, v in zip(features, model.feature_importances_)}
    metrics = {'r2': round(r2_score(y_te, preds), 4), 'rmse': round(np.sqrt(mean_squared_error(y_te, preds)), 4),
               'mae': round(mean_absolute_error(y_te, preds), 4), 'feature_importance': fi, 'features': features}
    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'metrics': metrics, 'features': features}, f)
    print(f"DecisionTree | R²={metrics['r2']} | RMSE={metrics['rmse']}")
    return model, metrics

if __name__ == '__main__':
    train()
