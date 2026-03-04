import pandas as pd, numpy as np
from datetime import datetime, timedelta

def generate_admin_data(days=90, start_date=None):
    if start_date is None: start_date = datetime.now() - timedelta(days=days)
    data = []
    for i in range(days * 24):
        ts = start_date + timedelta(hours=i)
        h, dow, dn = ts.hour, ts.weekday(), (ts - start_date).days
        is_wknd = int(dow >= 5)
        # Occupancy
        if is_wknd:
            occ = 3 if 10 <= h <= 14 else 1
        elif 8 <= h <= 17:
            occ = 60 if 10 <= h <= 15 else 40
        elif 7 <= h <= 8 or 17 <= h <= 19:
            occ = 20
        else:
            occ = 2
        occ = max(0, int(occ + np.random.normal(0, occ * 0.1)))
        temp = round(26 + 7 * np.sin(2*np.pi*dn/90) + 5*np.sin(2*np.pi*(h-6)/24) + np.random.normal(0,1.2), 1)
        # Electricity
        base = 6.0
        lighting = 4.0 if 7 <= h <= 19 else (2.0 if is_wknd else 1.0)
        hvac = 2.5 + max(0, temp - 24) * 0.5 if 7 <= h <= 19 and not is_wknd else 1.5
        it_load = 3.5 if 8 <= h <= 18 and not is_wknd else 1.5
        occ_load = occ * 0.04
        elec = max(3.0, base + lighting + hvac + it_load + occ_load + np.random.normal(0, 0.4))
        data.append({'timestamp': ts, 'occupancy': occ, 'is_weekend': is_wknd, 'temp_outside': temp, 'electricity_kwh': round(elec, 2)})
    return pd.DataFrame(data)

if __name__ == '__main__':
    df = generate_admin_data(90)
    df.to_csv('admin_data.csv', index=False)
    wknd = df[df['is_weekend']==1]; wkdy = df[df['is_weekend']==0]
    print(f"✓ {len(df)} records | Weekend avg: {wknd['electricity_kwh'].mean():.2f} | Weekday avg: {wkdy['electricity_kwh'].mean():.2f} kWh")
    print(f"✓ Dip: {((wkdy['electricity_kwh'].mean()-wknd['electricity_kwh'].mean())/wkdy['electricity_kwh'].mean()*100):.1f}% lower on weekends")
