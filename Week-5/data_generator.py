import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_hvac_data(days=75, start_date=None):
    """
    Generate synthetic HVAC energy data for university labs.
    Labs have equipment generating heat, variable occupancy, and outdoor temp.
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)

    timestamps = [start_date + timedelta(hours=i) for i in range(days * 24)]
    data = []

    for ts in timestamps:
        hour = ts.hour
        dow = ts.weekday()
        day_num = (ts - start_date).days
        is_weekend = int(dow >= 5)

        # --- LAB OCCUPANCY ---
        if is_weekend:
            if 10 <= hour <= 16:
                base_occ = 8
            else:
                base_occ = 1
        elif 9 <= hour <= 17:
            base_occ = 35 if 10 <= hour <= 15 else 22
        elif 18 <= hour <= 21:
            base_occ = 15  # grad students
        else:
            base_occ = 2
        occupancy = max(0, int(base_occ + np.random.normal(0, base_occ * 0.12)))

        # --- EQUIPMENT HEAT LOAD (kW) ---
        # Servers, centrifuges, ovens, 3D printers etc.
        if 8 <= hour <= 18 and not is_weekend:
            equip_heat = 6.0 + np.random.uniform(0, 3.0)
        else:
            equip_heat = 2.5 + np.random.uniform(0, 0.5)
        equip_heat = round(equip_heat, 2)

        # --- OUTDOOR TEMP ---
        seasonal = 28 + 8 * np.sin(2 * np.pi * day_num / 75)
        daily = 6 * np.sin(2 * np.pi * (hour - 6) / 24)
        temp_outside = round(seasonal + daily + np.random.normal(0, 1.5), 1)

        # --- HVAC SETPOINT (target indoor temp) ---
        if 7 <= hour <= 21 and not is_weekend:
            setpoint = 22.0
        else:
            setpoint = 26.0  # energy-saving mode

        # --- HVAC ELECTRICITY MODEL ---
        temp_diff = max(0, temp_outside - setpoint)
        hvac_cooling = 3.0 + temp_diff * 0.6
        hvac_equip_comp = equip_heat * 0.4  # compensate equipment heat
        hvac_occupancy = occupancy * 0.05
        hvac_fan = 2.0 if 7 <= hour <= 21 else 0.8

        total_hvac = hvac_cooling + hvac_equip_comp + hvac_occupancy + hvac_fan
        total_hvac = max(1.5, total_hvac + np.random.normal(0, 0.4))

        data.append({
            'timestamp': ts,
            'occupancy': occupancy,
            'equipment_heat_kw': equip_heat,
            'temp_outside': temp_outside,
            'setpoint': setpoint,
            'is_weekend': is_weekend,
            'hvac_kwh': round(total_hvac, 2)
        })

    return pd.DataFrame(data)


if __name__ == '__main__':
    print("=" * 60)
    print("HVAC Lab Data Generator")
    print("=" * 60)
    df = generate_hvac_data(days=75)
    out = 'hvac_data.csv'
    df.to_csv(out, index=False)
    print(f"✓ Generated {len(df)} records ({len(df)//24} days)")
    print(f"✓ HVAC range: {df['hvac_kwh'].min():.2f} – {df['hvac_kwh'].max():.2f} kWh")
    print(f"✓ Saved to {out}")
