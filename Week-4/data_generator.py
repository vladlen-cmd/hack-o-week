import pandas as pd, numpy as np
from datetime import datetime, timedelta

def generate_data(days=90, output='cafeteria_data.csv'):
    """Cafeteria energy data: lunch-hour surges driven by temperature/weather."""
    start = datetime.now() - timedelta(days=days)
    data = []
    for i in range(days * 24):
        ts = start + timedelta(hours=i)
        h, dow = ts.hour, ts.weekday()
        is_wknd = int(dow >= 5)
        temp = 26 + 7 * np.sin(2 * np.pi * ((ts - start).days - 30) / 90) + 4 * np.sin(np.pi * (h - 6) / 12) + np.random.normal(0, 2)
        weather = np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15])
        footfall = 0
        if not is_wknd and 7 <= h <= 21:
            if 7 <= h <= 9: footfall = 200 + np.random.normal(0, 30)   # breakfast
            elif 11 <= h <= 14: footfall = 500 + 150 * np.sin(np.pi * (h - 11) / 3) + np.random.normal(0, 40)  # lunch surge
            elif 18 <= h <= 20: footfall = 350 + np.random.normal(0, 35)  # dinner
            else: footfall = 80 + np.random.normal(0, 15)
        elif is_wknd and 10 <= h <= 19:
            footfall = 150 + np.random.normal(0, 25)
        footfall = max(0, int(footfall))
        if weather == 2: footfall = int(footfall * 1.15)

        energy = 20 + 0.08 * footfall + 0.35 * max(0, temp - 24) + (5 if weather == 2 else 0)
        if 11 <= h <= 14 and not is_wknd: energy += 12  # cooking surge
        energy = max(10, round(energy + np.random.normal(0, 2.5), 2))

        data.append({'timestamp': ts, 'hour': h, 'day_of_week': dow, 'is_weekend': is_wknd,
                     'temperature_c': round(temp, 1), 'weather': weather, 'footfall': footfall, 'energy_kwh': energy})
    df = pd.DataFrame(data)
    df.to_csv(output, index=False)
    print(f"Generated {len(df)} records | Lunch surge avg: {df[(df['hour'].between(11,14)) & (df['is_weekend']==0)]['energy_kwh'].mean():.1f} kWh")
    return df

if __name__ == '__main__':
    generate_data()
