import pandas as pd, numpy as np
from datetime import datetime, timedelta

def generate_anomaly_data(days=60, output='sensor_data.csv'):
    """
    Generate time-series sensor data with injected anomalies.
    Simulates campus power grid, server room temp, and network traffic.
    """
    start = datetime.now() - timedelta(days=days)
    data = []
    anomaly_days = np.random.choice(range(5, days - 5), size=12, replace=False)

    for i in range(days * 24):
        ts = start + timedelta(hours=i)
        h, dn = ts.hour, (ts - start).days

        # --- Power (kWh) ---
        power_base = 40 + 25 * np.sin(np.pi * (h - 6) / 12) if 6 <= h <= 18 else 20
        power = power_base + np.random.normal(0, 2)

        # --- Server Room Temp (°C) ---
        temp_base = 22 + 3 * np.sin(np.pi * (h - 8) / 16)
        temp = temp_base + np.random.normal(0, 0.5)

        # --- Network (Mbps) ---
        if 9 <= h <= 17:
            net_base = 400 + 200 * np.sin(np.pi * (h - 9) / 8)
        elif 18 <= h <= 23:
            net_base = 300
        else:
            net_base = 80
        network = net_base + np.random.normal(0, 20)

        # Inject anomalies
        is_anomaly = False
        anomaly_type = 'none'
        if dn in anomaly_days:
            if h == 14:  # power spike
                power += np.random.uniform(30, 60)
                is_anomaly = True
                anomaly_type = 'power_spike'
            if h == 3:  # server temp surge (cooling failure)
                temp += np.random.uniform(8, 15)
                is_anomaly = True
                anomaly_type = 'temp_surge'
            if h == 11:  # network drop
                network *= np.random.uniform(0.05, 0.2)
                is_anomaly = True
                anomaly_type = 'network_drop'

        # Additional random anomalies (rare)
        if np.random.random() < 0.005:
            choice = np.random.choice(['power', 'temp', 'network'])
            if choice == 'power':
                power += np.random.uniform(25, 50)
            elif choice == 'temp':
                temp += np.random.uniform(6, 12)
            else:
                network *= np.random.uniform(0.1, 0.3)
            is_anomaly = True
            anomaly_type = f'random_{choice}'

        data.append({
            'timestamp': ts, 'power_kwh': round(max(5, power), 2),
            'server_temp_c': round(max(15, temp), 2),
            'network_mbps': round(max(5, network), 1),
            'is_anomaly': int(is_anomaly), 'anomaly_type': anomaly_type
        })

    df = pd.DataFrame(data)
    df.to_csv(output, index=False)
    n_anom = df['is_anomaly'].sum()
    print(f"✓ Generated {len(df)} records with {n_anom} anomalies ({n_anom/len(df)*100:.1f}%)")
    return df

if __name__ == '__main__':
    generate_anomaly_data(60)
