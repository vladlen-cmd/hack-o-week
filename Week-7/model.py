import pandas as pd, numpy as np, pickle
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

def train():
    df = pd.read_csv('admin_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.weekday

    # Build hourly usage profiles per day
    df['date'] = df['timestamp'].dt.date.astype(str)
    profiles = df.pivot_table(index='date', columns='hour', values='electricity_kwh', aggfunc='mean').fillna(0)

    # K-Means clustering on daily profiles
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = km.fit_predict(profiles.values)
    profile_df = pd.DataFrame({'date': profiles.index, 'cluster': clusters})
    df = df.merge(profile_df, on='date', how='left')

    # Label clusters
    cluster_means = df.groupby('cluster')['electricity_kwh'].mean()
    label_map = {cluster_means.idxmax(): 'high_usage', cluster_means.idxmin(): 'low_usage'}
    for c in range(3):
        if c not in label_map: label_map[c] = 'medium_usage'
    df['cluster_label'] = df['cluster'].map(label_map)

    # Regression per cluster
    features = ['hour', 'day_of_week', 'is_weekend']
    models = {}; all_preds = []; all_actual = []
    for c in range(3):
        sub = df[df['cluster'] == c]
        X, y = sub[features].values, sub['electricity_kwh'].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        m = LinearRegression(); m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        models[c] = m; all_preds.extend(preds); all_actual.extend(y_te)

    r2 = round(r2_score(all_actual, all_preds), 4)
    rmse = round(np.sqrt(mean_squared_error(all_actual, all_preds)), 4)

    # Savings potential
    savings = {}
    for c in range(3):
        sub = df[df['cluster'] == c]
        savings[label_map[c]] = {'avg_kwh': round(float(sub['electricity_kwh'].mean()), 2),
                                  'count_days': int(sub['date'].nunique()),
                                  'pct': round(float(sub['date'].nunique()) / df['date'].nunique() * 100, 1)}
    potential = round(float(cluster_means.max() - cluster_means.min()), 2)

    bundle = {'kmeans': km, 'models': models, 'label_map': label_map,
              'metrics': {'r2': r2, 'rmse': rmse, 'n_clusters': 3, 'features': features},
              'savings': savings, 'potential_kwh': potential,
              'centers': km.cluster_centers_.tolist()}

    with open('model.pkl', 'wb') as f: pickle.dump(bundle, f)
    # Rename column for app.py consistency
    df.rename(columns={'electricity_kwh': 'energy_kwh'}, inplace=True)
    df.to_csv('admin_data_clustered.csv', index=False)
    print(f"KMeans(3) + LinearRegression | R²={r2} | Savings potential: {potential} kWh")
    for lbl, info in savings.items(): print(f"  {lbl}: {info['avg_kwh']} avg kWh, {info['count_days']} days ({info['pct']}%)")
    return bundle

if __name__ == '__main__':
    train()
