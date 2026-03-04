import pandas as pd, numpy as np
from datetime import datetime, timedelta

def generate_data(days=90, output='hvac_data.csv'):
    """HVAC data with 4 zones, occupancy/temperature features for cooling prediction."""
    start = datetime.now() - timedelta(days=days)
    zones = ['Lab-A', 'Lab-B', 'Server-Room', 'Conference']
    data = []
    for i in range(days * 24):
        ts = start + timedelta(hours=i)
        h, dow = ts.hour, ts.weekday()
        is_wknd = int(dow >= 5)
        outdoor = 28 + 8 * np.sin(2 * np.pi * ((ts - start).days) / 90) + 5 * np.sin(np.pi * (h - 6) / 12) + np.random.normal(0, 2)
        for zi, zone in enumerate(zones):
            if zone == 'Server-Room':
                occ = np.random.randint(1, 5); equip = 2000 + np.random.normal(0, 200)
            elif is_wknd:
                occ = np.random.randint(0, 3); equip = 100 + np.random.normal(0, 30)
            elif 8 <= h <= 18:
                occ = np.random.randint(5, 30); equip = 400 + 200 * np.sin(np.pi * (h - 8) / 10) + np.random.normal(0, 50)
            else:
                occ = np.random.randint(0, 3); equip = 80 + np.random.normal(0, 20)
            cooling = 3 + 0.15 * occ + 0.3 * max(0, outdoor - 22) + 0.002 * equip + np.random.normal(0, 0.8)
            if zone == 'Server-Room': cooling += 5
            cooling = max(1, round(cooling, 2))
            data.append({'timestamp': ts, 'hour': h, 'day_of_week': dow, 'is_weekend': is_wknd,
                         'zone': zi, 'zone_name': zone, 'occupancy': occ, 'outdoor_temp_c': round(outdoor, 1),
                         'equipment_heat_w': round(max(0, equip), 1), 'cooling_kwh': cooling})
    df = pd.DataFrame(data); df.to_csv(output, index=False)
    print(f"Generated {len(df)} records across {len(zones)} zones")
    for z in zones: print(f"  {z}: {df[df['zone_name']==z]['cooling_kwh'].mean():.1f} avg kWh")
    return df

if __name__ == '__main__':
    generate_data()
