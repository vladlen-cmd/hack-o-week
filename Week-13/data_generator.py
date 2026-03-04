import pandas as pd, numpy as np, json
from datetime import datetime, timedelta

def generate_dashboard_data(days=90, output='dashboard_data.csv'):
    """
    Generate multi-source campus data for a comprehensive visualization dashboard.
    Covers energy, network, footfall, air quality, and event attendance.
    """
    start = datetime.now() - timedelta(days=days)
    data = []

    for i in range(days * 24):
        ts = start + timedelta(hours=i)
        h, dow, dn = ts.hour, ts.weekday(), (ts - start).days
        is_wknd = int(dow >= 5)

        # Energy (kWh)
        if is_wknd:
            energy = 30 + 10 * np.sin(np.pi * (h - 6) / 12) if 6 <= h <= 18 else 20
        elif 8 <= h <= 18:
            energy = 70 + 20 * np.sin(np.pi * (h - 8) / 10)
        else:
            energy = 25
        energy = max(15, energy + np.random.normal(0, 3))

        # Network bandwidth (Mbps)
        if is_wknd:
            bandwidth = 200 + np.random.normal(0, 30)
        elif 9 <= h <= 17:
            bandwidth = 800 + 200 * np.sin(np.pi * (h - 9) / 8) + np.random.normal(0, 50)
        elif 18 <= h <= 23:
            bandwidth = 600 + np.random.normal(0, 40)  # streaming hours
        else:
            bandwidth = 150 + np.random.normal(0, 20)
        bandwidth = max(50, bandwidth)

        # Footfall (total campus)
        if is_wknd:
            footfall = 500 if 10 <= h <= 16 else 100
        elif 8 <= h <= 18:
            footfall = 3000 + 1000 * np.sin(np.pi * (h - 8) / 10)
        elif 18 <= h <= 22:
            footfall = 1500
        else:
            footfall = 200
        footfall = max(50, int(footfall + np.random.normal(0, footfall * 0.08)))

        # Air Quality Index (AQI)
        base_aqi = 60 + 20 * np.sin(2 * np.pi * dn / 90) + 15 * np.sin(2 * np.pi * (h - 8) / 24)
        if 8 <= h <= 10 or 17 <= h <= 19:
            base_aqi += 25  # rush hour pollution
        aqi = max(20, min(300, round(base_aqi + np.random.normal(0, 8))))

        # AQI category
        if aqi <= 50: aqi_cat = 'Good'
        elif aqi <= 100: aqi_cat = 'Moderate'
        elif aqi <= 150: aqi_cat = 'Unhealthy-SG'
        else: aqi_cat = 'Unhealthy'

        # Event attendance (0 if no event)
        has_event = bool(dow in [2, 4] and 16 <= h <= 19) or bool(dow == 5 and 18 <= h <= 21)
        event_attendance = int(np.random.randint(80, 300)) if has_event else 0

        data.append({
            'timestamp': ts, 'energy_kwh': round(energy, 2),
            'bandwidth_mbps': round(bandwidth, 1), 'footfall': footfall,
            'aqi': aqi, 'aqi_category': aqi_cat,
            'event_attendance': event_attendance, 'is_weekend': is_wknd
        })

    df = pd.DataFrame(data)
    df.to_csv(output, index=False)
    print(f"✓ Generated {len(df)} records over {days} days")
    print(f"✓ Energy: {df['energy_kwh'].mean():.1f} avg kWh | Bandwidth: {df['bandwidth_mbps'].mean():.0f} avg Mbps")
    print(f"✓ AQI range: {df['aqi'].min()} – {df['aqi'].max()} | Events: {(df['event_attendance'] > 0).sum()}")
    return df

if __name__ == '__main__':
    generate_dashboard_data(90)
