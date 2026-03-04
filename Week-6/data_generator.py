import pandas as pd, numpy as np
from datetime import datetime, timedelta

def generate_data(days=90, output='sports_data.csv'):
    """Sports facility data with day types: weekday, weekend, event."""
    start = datetime.now() - timedelta(days=days)
    event_days = np.random.choice(range(days), size=20, replace=False)
    data = []
    for i in range(days * 24):
        ts = start + timedelta(hours=i)
        h, dow, dn = ts.hour, ts.weekday(), (ts - start).days
        is_wknd = int(dow >= 5)
        is_event = int(dn in event_days)
        day_type = 'event' if is_event else ('weekend' if is_wknd else 'weekday')
        if is_event and 18 <= h <= 22:
            energy = 80 + 30 * np.sin(np.pi * (h - 18) / 4) + np.random.normal(0, 5)
        elif 18 <= h <= 22:
            energy = 40 + 15 * np.sin(np.pi * (h - 18) / 4) + np.random.normal(0, 3)
        elif 6 <= h <= 17:
            energy = 20 + (10 if is_wknd else 5) + np.random.normal(0, 2)
        else:
            energy = 8 + np.random.normal(0, 1)
        energy = max(3, round(energy, 2))
        data.append({'timestamp': ts, 'hour': h, 'day_of_week': dow, 'is_weekend': is_wknd,
                     'is_event': is_event, 'day_type': day_type, 'energy_kwh': energy})
    df = pd.DataFrame(data); df.to_csv(output, index=False)
    print(f"Generated {len(df)} records | Events: {len(event_days)} days")
    for dt in ['weekday','weekend','event']:
        print(f"  {dt}: {df[df['day_type']==dt]['energy_kwh'].mean():.1f} avg kWh")
    return df

if __name__ == '__main__':
    generate_data()
