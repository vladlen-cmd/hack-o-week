import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_cafeteria_data(days=60, start_date=None):
    """
    Generate synthetic cafeteria electricity consumption data.
    
    Simulates a university cafeteria with distinct meal-period spikes,
    weather-dependent HVAC, and weekend/weekday patterns.
    
    Args:
        days: Number of days of data to generate
        start_date: Starting date (defaults to N days ago)
    
    Returns:
        DataFrame with timestamp, footfall, meal_period, is_weekend,
                       temperature, electricity_kwh
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)
    
    timestamps = [start_date + timedelta(hours=i) for i in range(days * 24)]
    data = []
    
    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.weekday()
        day_number = (ts - start_date).days
        is_weekend = int(day_of_week >= 5)
        
        # --- MEAL PERIOD ---
        # 0 = off-hours, 1 = breakfast, 2 = lunch, 3 = snacks, 4 = dinner
        if 7 <= hour <= 9:
            meal_period = 1
        elif 11 <= hour <= 14:
            meal_period = 2
        elif 15 <= hour <= 17:
            meal_period = 3
        elif 19 <= hour <= 21:
            meal_period = 4
        else:
            meal_period = 0
        
        # --- FOOTFALL MODEL ---
        if meal_period == 0:
            # Off-hours: cleaning staff, prep
            base_footfall = 5
        elif meal_period == 1:  # Breakfast
            if is_weekend:
                base_footfall = 40
            else:
                base_footfall = 80 if hour == 8 else 55
        elif meal_period == 2:  # Lunch (busiest)
            if is_weekend:
                base_footfall = 100
            else:
                peak = 200 if 12 <= hour <= 13 else 140
                base_footfall = peak
        elif meal_period == 3:  # Snacks
            base_footfall = 60 if not is_weekend else 35
        elif meal_period == 4:  # Dinner
            if is_weekend:
                base_footfall = 90
            else:
                base_footfall = 160 if hour == 20 else 120
        
        # Add special event spike (once a week on Wednesday, a popular meal)
        if day_of_week == 2 and meal_period == 2:
            base_footfall = int(base_footfall * 1.25)
        
        footfall = max(0, int(base_footfall + np.random.normal(0, base_footfall * 0.1)))
        
        # --- TEMPERATURE MODEL ---
        seasonal = 25 + 6 * np.sin(2 * np.pi * day_number / 60)
        daily_cycle = 6 * np.sin(2 * np.pi * (hour - 6) / 24)
        temperature = round(seasonal + daily_cycle + np.random.normal(0, 1.2), 1)
        
        # --- ELECTRICITY MODEL ---
        # Base load: refrigerators, freezers, emergency lighting
        base_load = 12.0  # kWh — cafeteria equipment runs 24/7
        
        # Cooking equipment load (ovens, grills, fryers, steamers)
        if meal_period == 0:
            cooking_load = 1.0  # Minimal prep
        elif meal_period == 1:
            cooking_load = 8.0
        elif meal_period == 2:
            cooking_load = 15.0  # Lunch is heaviest
        elif meal_period == 3:
            cooking_load = 5.0
        elif meal_period == 4:
            cooking_load = 12.0
        
        # Ventilation & exhaust (tied to cooking)
        ventilation = cooking_load * 0.3
        
        # Lighting
        if 6 <= hour <= 22:
            lighting = 3.5
        else:
            lighting = 0.8
        
        # HVAC (temperature dependent — cafeteria has high heat from cooking)
        temp_diff = abs(temperature - 24)
        hvac_load = 3.0 + temp_diff * 0.35
        if meal_period in [2, 4]:
            hvac_load *= 1.4  # Extra cooling during heavy cooking
        
        # Footfall-dependent (dish washing, POS systems, displays)
        footfall_load = footfall * 0.02
        
        # Total
        electricity = base_load + cooking_load + ventilation + lighting + hvac_load + footfall_load
        electricity = max(5.0, electricity + np.random.normal(0, 0.8))
        
        data.append({
            'timestamp': ts,
            'footfall': footfall,
            'meal_period': meal_period,
            'is_weekend': is_weekend,
            'temperature': temperature,
            'electricity_kwh': round(electricity, 2)
        })
    
    return pd.DataFrame(data)


if __name__ == '__main__':
    print("=" * 60)
    print("Cafeteria Load Data Generator")
    print("=" * 60)
    
    df = generate_cafeteria_data(days=60)
    
    output_file = 'cafeteria_data.csv'
    df.to_csv(output_file, index=False)
    
    meal_names = {0: 'Off-hours', 1: 'Breakfast', 2: 'Lunch', 3: 'Snacks', 4: 'Dinner'}
    
    print(f"✓ Generated {len(df)} records ({len(df) // 24} days)")
    print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"✓ Footfall range: {df['footfall'].min()} – {df['footfall'].max()}")
    print(f"✓ Electricity range: {df['electricity_kwh'].min():.2f} – {df['electricity_kwh'].max():.2f} kWh")
    
    print("\n✓ Average electricity by meal period:")
    for mp, name in meal_names.items():
        avg = df[df['meal_period'] == mp]['electricity_kwh'].mean()
        print(f"  {name:12s}: {avg:.2f} kWh")
    
    print(f"\n✓ Saved to {output_file}")
    
    print("\nSample data:")
    print(df.head(10).to_string(index=False))
