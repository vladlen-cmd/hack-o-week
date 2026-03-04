import pandas as pd, numpy as np
from datetime import datetime, timedelta

def generate_data(days=90, output='heart_data.csv'):
    """Generate heart rate data with injected anomalies for Isolation Forest detection."""
    start = datetime.now() - timedelta(days=days)
    data = []
    for i in range(days * 24):
        ts = start + timedelta(hours=i)
        h = ts.hour; is_sleep = int(h < 6 or h >= 23)
        # Normal heart rate: ~65-85 bpm, lower during sleep
        base_hr = 62 if is_sleep else 72
        hr = base_hr + np.random.normal(0, 5)
        resting_hr = 58 + np.random.normal(0, 3)
        hr_variability = 40 + np.random.normal(0, 8)

        # Inject anomalies (~3%)
        is_anomaly = 0
        if np.random.random() < 0.03:
            anomaly_type = np.random.choice(['tachycardia', 'bradycardia', 'irregular'])
            if anomaly_type == 'tachycardia': hr = np.random.uniform(110, 160)
            elif anomaly_type == 'bradycardia': hr = np.random.uniform(35, 50)
            else: hr_variability = np.random.uniform(80, 150)
            is_anomaly = 1

        data.append({'timestamp': ts, 'hour': h, 'heart_rate': round(max(30, hr), 1),
                     'resting_hr': round(max(45, resting_hr), 1),
                     'hr_variability': round(max(10, hr_variability), 1),
                     'is_sleep': is_sleep, 'is_anomaly': is_anomaly})

    df = pd.DataFrame(data); df.to_csv(output, index=False)
    anomaly_count = df['is_anomaly'].sum()
    print(f"Generated {len(df)} records | Anomalies: {anomaly_count} ({anomaly_count/len(df)*100:.1f}%)")
    print(f"Avg HR: {df['heart_rate'].mean():.1f} bpm | Range: {df['heart_rate'].min():.0f}-{df['heart_rate'].max():.0f}")
    return df

if __name__ == '__main__':
    generate_data()
