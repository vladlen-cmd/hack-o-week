import pandas as pd, numpy as np
from datetime import datetime, timedelta

def generate_data(days=90, output='library_data.csv'):
    """Generate library energy data with lunch-hour surges, temperature/weather influence."""
    start = datetime.now() - timedelta(days=days)
    data = []
    for i in range(days * 24):
        ts = start + timedelta(hours=i)
        h, dow = ts.hour, ts.weekday()
        is_wknd = int(dow >= 5)
        is_exam = int(60 <= (ts - start).days <= 75)

        # Temperature (seasonal + daily cycle)
        temp = 28 + 8 * np.sin(2 * np.pi * ((ts - start).days - 30) / 90) + 5 * np.sin(np.pi * (h - 6) / 12) + np.random.normal(0, 2)
        weather = np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15])  # 0=clear,1=cloudy,2=rain
        occupancy = 0
        if not is_wknd:
            if 8 <= h <= 20:
                occupancy = 150 + 80 * np.sin(np.pi * (h - 8) / 12)
                if 12 <= h <= 14: occupancy += 60  # lunch surge
                if is_exam: occupancy += 100
                if weather == 2: occupancy += 30  # rain drives people indoors
            elif 20 < h <= 23:
                occupancy = 60 + (30 if is_exam else 0)
        else:
            if 10 <= h <= 18: occupancy = 50 + np.random.normal(0, 10)
        occupancy = max(0, int(occupancy + np.random.normal(0, 12)))

        # Energy: driven by occupancy, temperature (cooling), weather
        energy = 15 + 0.12 * occupancy + 0.4 * max(0, temp - 25) + (3 if weather == 2 else 0)
        if 12 <= h <= 14 and not is_wknd: energy += 8  # lunch-hour surge
        if is_exam: energy += 5
        energy = max(8, round(energy + np.random.normal(0, 2), 2))

        data.append({'timestamp': ts, 'hour': h, 'day_of_week': dow, 'is_weekend': is_wknd,
                     'is_exam': is_exam, 'temperature_c': round(temp, 1), 'weather': weather,
                     'occupancy': occupancy, 'energy_kwh': energy})
    df = pd.DataFrame(data)
    df.to_csv(output, index=False)
    print(f"Generated {len(df)} records | Avg energy: {df['energy_kwh'].mean():.1f} kWh")
    print(f"Lunch surge (12-14h weekday): {df[(df['hour'].between(12,14)) & (df['is_weekend']==0)]['energy_kwh'].mean():.1f} kWh avg")
    return df

if __name__ == '__main__':
    generate_data()
