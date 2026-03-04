import pandas as pd, numpy as np
from datetime import datetime, timedelta

def generate_laundry_data(days=45, start_date=None):
    if start_date is None: start_date = datetime.now() - timedelta(days=days)
    data = []
    for i in range(days * 24):
        ts = start_date + timedelta(hours=i)
        h, dow, dn = ts.hour, ts.weekday(), (ts - start_date).days
        is_wknd = int(dow >= 5)
        # Machines in use (hostel has 20 washers + 20 dryers)
        if is_wknd:
            if 9 <= h <= 13: machines = 16
            elif 14 <= h <= 18: machines = 14
            elif 19 <= h <= 22: machines = 10
            else: machines = 2
        else:
            if 7 <= h <= 9: machines = 8
            elif 17 <= h <= 21: machines = 15  # evening peak after classes
            elif 21 <= h <= 23: machines = 10
            elif 10 <= h <= 16: machines = 5
            else: machines = 1
        # Sunday is THE laundry day
        if dow == 6 and 9 <= h <= 18: machines = int(machines * 1.4)
        machines = max(0, min(40, int(machines + np.random.normal(0, machines * 0.12))))
        water_temp = 40 if machines > 10 else 30  # hot wash for heavy loads
        # Electricity
        washer_load = machines * 0.5  # 500W per machine
        dryer_load = max(0, machines - 3) * 0.8  # dryers lag behind
        hot_water = machines * 0.15 * (water_temp / 40)
        base = 1.5  # lighting, ventilation
        elec = max(0.5, base + washer_load + dryer_load + hot_water + np.random.normal(0, 0.3))
        data.append({'timestamp': ts, 'machines_active': machines, 'is_weekend': is_wknd,
                     'water_temp': water_temp, 'electricity_kwh': round(elec, 2)})
    return pd.DataFrame(data)

if __name__ == '__main__':
    df = generate_laundry_data(45); df.to_csv('laundry_data.csv', index=False)
    sun = df[df['timestamp'].apply(lambda x: pd.to_datetime(x).dayofweek == 6)]
    print(f"✓ {len(df)} records | Sunday avg: {sun['electricity_kwh'].mean():.2f} | Overall avg: {df['electricity_kwh'].mean():.2f} kWh")
