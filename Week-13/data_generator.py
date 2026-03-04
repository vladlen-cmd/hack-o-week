import pandas as pd, numpy as np
from datetime import datetime, timedelta

def generate_data(days=90, output='activity_data.csv'):
    """Generate daily activity data (steps, HR, calories, sleep) for dashboard."""
    start = datetime.now() - timedelta(days=days)
    data = []
    for i in range(days):
        dt = start + timedelta(days=i)
        dow = dt.weekday()
        steps = int(np.random.normal(8000 if dow < 5 else 10000, 2000))
        hr = int(np.random.normal(72, 8))
        calories = int(steps * 0.04 + np.random.normal(0, 50))
        sleep = round(np.random.normal(7 if dow < 5 else 8, 0.8), 1)
        data.append({'date': dt.strftime('%Y-%m-%d'), 'steps': max(1000, steps),
                     'heart_rate': max(55, min(100, hr)), 'calories': max(100, calories),
                     'sleep_hours': max(4, min(10, sleep))})
    df = pd.DataFrame(data); df.to_csv(output, index=False)
    print(f"Generated {len(df)} days | Avg steps: {df['steps'].mean():.0f} | Avg HR: {df['heart_rate'].mean():.0f}")
    return df

if __name__ == '__main__':
    generate_data()
