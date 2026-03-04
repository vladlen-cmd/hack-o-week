import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sports_data(days=60, start_date=None):
    """
    Generate synthetic sports facility energy data focused on evening/night usage.
    Covers gym, basketball courts, swimming pool, and floodlit outdoor fields.
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)

    facilities = ['Gym', 'Courts', 'Pool', 'Fields']
    timestamps = [start_date + timedelta(hours=i) for i in range(days * 24)]
    data = []

    for ts in timestamps:
        hour = ts.hour
        dow = ts.weekday()
        day_num = (ts - start_date).days
        is_weekend = int(dow >= 5)
        is_night = int(hour >= 18 or hour < 6)

        # --- FACILITY USAGE (number of users) ---
        if is_weekend:
            if 8 <= hour <= 12:
                base_users = 50
            elif 14 <= hour <= 20:
                base_users = 80
            else:
                base_users = 10
        else:
            if 6 <= hour <= 8:
                base_users = 25  # morning session
            elif 16 <= hour <= 21:
                base_users = 90  # evening peak
            elif 9 <= hour <= 15:
                base_users = 20
            else:
                base_users = 5

        # Events boost (tournament every 2 weeks on Saturday evening)
        has_event = (dow == 5 and 17 <= hour <= 22 and day_num % 14 < 1)
        if has_event:
            base_users = int(base_users * 2.5)

        users = max(0, int(base_users + np.random.normal(0, base_users * 0.1)))

        # --- FLOODLIGHT STATUS ---
        floodlights_on = int(18 <= hour <= 22 and users > 15)

        # --- ELECTRICITY MODEL ---
        base_load = 5.0  # always-on: security, CCTV, emergency

        # Lighting
        if 6 <= hour <= 18:
            lighting = 3.0  # natural + indoor
        elif floodlights_on:
            lighting = 12.0  # floodlights are heavy!
        elif 18 <= hour <= 22:
            lighting = 6.0  # indoor only
        else:
            lighting = 1.5

        # HVAC for gym & pool
        if 6 <= hour <= 22:
            hvac = 4.0 + users * 0.03
        else:
            hvac = 2.0

        # Equipment (treadmills, pool pumps, scoreboards)
        equip = users * 0.06 + (3.0 if 6 <= hour <= 22 else 1.0)

        # Pool pump runs 24/7 but heating is active hours
        pool = 2.0 + (3.0 if 6 <= hour <= 22 else 0.0)

        electricity = base_load + lighting + hvac + equip + pool
        if has_event:
            electricity *= 1.15  # extra displays, sound systems
        electricity = max(4.0, electricity + np.random.normal(0, 0.6))

        data.append({
            'timestamp': ts, 'users': users, 'is_weekend': is_weekend,
            'is_night': is_night, 'floodlights_on': floodlights_on,
            'has_event': int(has_event),
            'electricity_kwh': round(electricity, 2)
        })

    return pd.DataFrame(data)

if __name__ == '__main__':
    print("=" * 60)
    print("Sports Facility Data Generator")
    print("=" * 60)
    df = generate_sports_data(days=60)
    df.to_csv('sports_data.csv', index=False)
    night = df[df['is_night'] == 1]
    print(f"✓ Generated {len(df)} records")
    print(f"✓ Night usage records: {len(night)}")
    print(f"✓ Avg electricity (night): {night['electricity_kwh'].mean():.2f} kWh")
    print(f"✓ Avg electricity (day): {df[df['is_night']==0]['electricity_kwh'].mean():.2f} kWh")
    print(f"✓ Saved to sports_data.csv")
