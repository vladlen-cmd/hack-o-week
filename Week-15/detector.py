import pandas as pd, numpy as np, pickle
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def train():
    df = pd.read_csv('heart_data.csv')
    features = ['heart_rate', 'resting_hr', 'hr_variability']
    X = df[features].values
    y_true = df['is_anomaly'].values

    # Isolation Forest on heart rate features
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X)
    scores = model.decision_function(X)
    preds = model.predict(X)
    y_pred = (preds == -1).astype(int)  # -1 = anomaly

    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 3),
        'recall': round(recall_score(y_true, y_pred, zero_division=0), 3),
        'f1': round(f1_score(y_true, y_pred, zero_division=0), 3),
        'tp': int(cm[1, 1]), 'fp': int(cm[0, 1]), 'fn': int(cm[1, 0]), 'tn': int(cm[0, 0]),
        'total_anomalies': int(y_pred.sum()), 'actual_anomalies': int(y_true.sum()),
        'features': features
    }

    df['anomaly_score'] = scores
    df['predicted_anomaly'] = y_pred
    df.to_csv('heart_data_scored.csv', index=False)

    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'metrics': metrics}, f)

    print(f"Isolation Forest | F1={metrics['f1']} | Precision={metrics['precision']} | Recall={metrics['recall']}")
    print(f"Detected: {metrics['total_anomalies']} | Actual: {metrics['actual_anomalies']}")
    return metrics

if __name__ == '__main__':
    train()
