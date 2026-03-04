import pandas as pd, numpy as np, json, os
from datetime import datetime, timedelta

def simulate_wearable_stream(users=5, days=14, output='wearable_data.json'):
    """
    Simulate wearable device data for multiple users.
    Generates heart rate, steps, SpO2, skin temp, sleep, and calories.
    """
    device_types = ['FitBand Pro', 'HealthWatch X', 'PulseTrack 3', 'VitalRing', 'BioSync']
    records = []
    start = datetime.now() - timedelta(days=days)

    for uid in range(1, users + 1):
        device = device_types[(uid - 1) % len(device_types)]
        age = np.random.randint(18, 45)
        resting_hr = 60 + age * 0.2 + np.random.normal(0, 3)

        for d in range(days):
            for h in range(24):
                ts = start + timedelta(days=d, hours=h, minutes=np.random.randint(0, 60))
                # --- Heart Rate ---
                if 0 <= h < 6:  # sleeping
                    hr = resting_hr - 8 + np.random.normal(0, 2)
                elif 6 <= h <= 8 or 17 <= h <= 19:  # exercise window
                    hr = resting_hr + 40 + np.random.normal(0, 10)
                elif 9 <= h <= 17:  # work
                    hr = resting_hr + 10 + np.random.normal(0, 5)
                else:
                    hr = resting_hr + 5 + np.random.normal(0, 3)
                hr = max(45, min(180, hr))

                # --- Steps ---
                if 0 <= h < 6:
                    steps = np.random.randint(0, 10)
                elif h in [7, 8, 12, 13, 17, 18]:
                    steps = np.random.randint(300, 1200)
                else:
                    steps = np.random.randint(20, 200)

                # --- SpO2 ---
                spo2 = round(max(90, min(100, 97.5 + np.random.normal(0, 0.6))), 1)

                # --- Skin Temperature ---
                skin_temp = round(36.2 + 0.5 * np.sin(2 * np.pi * (h - 4) / 24) + np.random.normal(0, 0.2), 1)

                # --- Sleep (boolean — are they sleeping?) ---
                is_sleeping = int(h < 6 or h >= 23)

                # --- Calories burned (kcal) ---
                cals = round(steps * 0.04 + hr * 0.08 + np.random.uniform(5, 20), 1)

                # --- Battery level simulation ---
                battery = max(5, int(100 - (h * 3.5) + np.random.normal(0, 2)))

                records.append({
                    'timestamp': ts.isoformat(),
                    'user_id': f'user_{uid:03d}',
                    'device_type': device,
                    'heart_rate_bpm': round(hr, 1),
                    'steps': int(steps),
                    'spo2_pct': spo2,
                    'skin_temp_c': skin_temp,
                    'is_sleeping': is_sleeping,
                    'calories_kcal': cals,
                    'battery_pct': battery
                })

    # Save as JSON (simulating real ingest format)
    with open(output, 'w') as f:
        json.dump(records, f, indent=2)

    # Also save CSV for easier analysis
    df = pd.DataFrame(records)
    df.to_csv(output.replace('.json', '.csv'), index=False)

    print(f"✓ Generated {len(records)} records for {users} users over {days} days")
    print(f"✓ Devices: {', '.join(device_types[:users])}")
    print(f"✓ Avg HR: {df['heart_rate_bpm'].mean():.1f} bpm")
    print(f"✓ Avg daily steps: {df.groupby('user_id')['steps'].sum().mean() / days:.0f}")
    print(f"✓ Saved to {output} and {output.replace('.json', '.csv')}")
    return df

if __name__ == '__main__':
    simulate_wearable_stream(users=5, days=14)
