import pandas as pd, numpy as np
from datetime import datetime, timedelta

def generate_campus_data(days=120, start_date=None):
    """Generate campus-wide sustainability data across multiple building categories."""
    if start_date is None: start_date = datetime.now() - timedelta(days=days)
    buildings = {
        'Library': {'base': 10, 'peak_mult': 2.5, 'weekend_mult': 0.4},
        'Labs': {'base': 8, 'peak_mult': 2.0, 'weekend_mult': 0.3},
        'Admin': {'base': 6, 'peak_mult': 2.2, 'weekend_mult': 0.25},
        'Cafeteria': {'base': 12, 'peak_mult': 3.0, 'weekend_mult': 0.5},
        'Hostels': {'base': 5, 'peak_mult': 1.8, 'weekend_mult': 0.9},
        'Sports': {'base': 4, 'peak_mult': 2.5, 'weekend_mult': 0.7},
        'Parking': {'base': 3, 'peak_mult': 1.5, 'weekend_mult': 0.3},
    }
    data = []
    for i in range(days * 24):
        ts = start_date + timedelta(hours=i)
        h, dow, dn = ts.hour, ts.weekday(), (ts - start_date).days
        is_wknd = int(dow >= 5)
        temp = round(26 + 8*np.sin(2*np.pi*dn/120) + 5*np.sin(2*np.pi*(h-6)/24) + np.random.normal(0,1.2), 1)
        # Solar generation (campus has rooftop panels)
        if 7 <= h <= 17:
            solar = max(0, round(15 * np.sin(np.pi*(h-7)/10) * (0.7+0.3*np.random.random()) * (1 - 0.3*(temp>35)), 2))
        else:
            solar = 0
        # Water consumption (liters)
        if is_wknd:
            water = max(100, int(800 + np.random.normal(0, 100)))
        elif 8 <= h <= 18:
            water = max(200, int(3000 + np.random.normal(0, 300)))
        else:
            water = max(100, int(500 + np.random.normal(0, 80)))

        total_elec = 0
        for bname, bp in buildings.items():
            if 8 <= h <= 18 and not is_wknd:
                e = bp['base'] * bp['peak_mult']
            elif is_wknd:
                e = bp['base'] * bp['weekend_mult']
            else:
                e = bp['base'] * 0.5
            e += max(0, temp-24)*0.2 + np.random.normal(0, 0.3)
            total_elec += max(0.5, e)

        co2 = round(total_elec * 0.82, 2)  # kg CO2 per kWh (India grid)
        co2_saved = round(solar * 0.82, 2)
        total_elec = round(total_elec, 2)
        sustainability_score = round(min(100, max(0, 50 + (solar/max(total_elec,1))*100 - (total_elec/80)*20 + np.random.normal(0,3))), 1)

        data.append({'timestamp':ts,'total_electricity_kwh':total_elec,'solar_generation_kwh':round(solar,2),
                     'water_consumption_l':water,'co2_emissions_kg':co2,'co2_saved_kg':co2_saved,
                     'temperature':temp,'is_weekend':is_wknd,'sustainability_score':sustainability_score})
    return pd.DataFrame(data)

if __name__=='__main__':
    df=generate_campus_data(120); df.to_csv('campus_data.csv',index=False)
    print(f"✓ {len(df)} records | Avg score: {df['sustainability_score'].mean():.1f}")
    print(f"✓ Total CO2: {df['co2_emissions_kg'].sum():.0f} kg | Saved: {df['co2_saved_kg'].sum():.0f} kg")
