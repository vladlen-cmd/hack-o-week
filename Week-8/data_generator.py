import pandas as pd, numpy as np
from datetime import datetime, timedelta

def generate_parking_data(days=60, start_date=None):
    if start_date is None: start_date = datetime.now() - timedelta(days=days)
    data = []
    for i in range(days * 24):
        ts = start_date + timedelta(hours=i)
        h, dow, dn = ts.hour, ts.weekday(), (ts - start_date).days
        is_wknd = int(dow >= 5)
        # Sunset/sunrise approx
        sunrise = 6 + int(np.sin(2*np.pi*dn/365)*0.5)
        sunset = 18 + int(np.sin(2*np.pi*dn/365)*1)
        is_dark = int(h < sunrise or h >= sunset)
        # Vehicles
        if is_wknd:
            vehicles = 15 if 10 <= h <= 16 else 5
        elif 8 <= h <= 17:
            vehicles = 120 if 9 <= h <= 16 else 70
        elif 17 < h <= 20:
            vehicles = 40
        else:
            vehicles = 8
        has_event = int(dow == 4 and 18 <= h <= 22 and dn % 7 < 1)
        if has_event: vehicles = int(vehicles * 2)
        vehicles = max(0, int(vehicles + np.random.normal(0, vehicles * 0.1)))
        # Electricity
        base = 2.0  # security, cameras
        if is_dark:
            zone_lights = 8.0  # main zones
            pathway = 3.0
            entry = 2.5
        else:
            zone_lights = 0.5
            pathway = 0.3
            entry = 0.5
        motion = vehicles * 0.015  # motion-activated extras
        if has_event: zone_lights *= 1.3
        elec = max(1.5, base + zone_lights + pathway + entry + motion + np.random.normal(0, 0.3))
        data.append({'timestamp':ts,'vehicles':vehicles,'is_weekend':is_wknd,'is_dark':is_dark,
                     'sunrise_hour':sunrise,'sunset_hour':sunset,'has_event':int(has_event),
                     'electricity_kwh':round(elec,2)})
    return pd.DataFrame(data)

if __name__=='__main__':
    df=generate_parking_data(60); df.to_csv('parking_data.csv',index=False)
    print(f"✓ {len(df)} records | Dark avg: {df[df['is_dark']==1]['electricity_kwh'].mean():.2f} | Light avg: {df[df['is_dark']==0]['electricity_kwh'].mean():.2f}")
