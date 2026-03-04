import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_classroom_data(days=30, start_date=None):
    """
    Generate synthetic Wi-Fi occupancy and electricity consumption data.
    
    Args:
        days: Number of days to generate data for
        start_date: Starting date (defaults to 30 days ago)
    
    Returns:
        DataFrame with timestamp, occupancy, and electricity columns
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)
    
    # Generate hourly timestamps
    timestamps = [start_date + timedelta(hours=i) for i in range(days * 24)]
    
    data = []
    
    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.weekday()  # 0=Monday, 6=Sunday
        
        # Base occupancy pattern (number of devices connected)
        if day_of_week >= 5:  # Weekend
            base_occupancy = 5
        elif 9 <= hour <= 17:  # Class hours on weekdays
            # Peak during mid-day
            if 11 <= hour <= 14:
                base_occupancy = 45
            elif 9 <= hour <= 11 or 14 <= hour <= 17:
                base_occupancy = 35
            else:
                base_occupancy = 25
        elif 8 <= hour <= 9 or 17 <= hour <= 19:  # Transition hours
            base_occupancy = 20
        else:  # Night/early morning
            base_occupancy = 3
        
        # Add random variation
        occupancy = max(0, int(base_occupancy + np.random.normal(0, base_occupancy * 0.15)))
        
        # Calculate electricity consumption (kWh)
        # Base load (lights, HVAC, equipment) + occupancy-dependent load
        base_load = 2.5  # kWh base consumption
        
        # Occupancy-dependent load (computers, projectors, etc.)
        occupancy_load = occupancy * 0.15  # ~150W per device
        
        # HVAC increases during class hours
        hvac_load = 0
        if 8 <= hour <= 18 and day_of_week < 5:
            hvac_load = 3.0
        
        # Total electricity with noise
        electricity = base_load + occupancy_load + hvac_load
        electricity = max(0, electricity + np.random.normal(0, 0.3))
        
        data.append({
            'timestamp': ts,
            'occupancy': occupancy,
            'electricity_kwh': round(electricity, 2)
        })
    
    df = pd.DataFrame(data)
    return df

if __name__ == '__main__':
    print("Generating synthetic classroom data...")
    df = generate_classroom_data(days=30)
    
    # Save to CSV
    output_file = 'classroom_data.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✓ Generated {len(df)} records")
    print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"✓ Occupancy range: {df['occupancy'].min()} to {df['occupancy'].max()} devices")
    print(f"✓ Electricity range: {df['electricity_kwh'].min():.2f} to {df['electricity_kwh'].max():.2f} kWh")
    print(f"✓ Saved to {output_file}")
    
    # Show sample
    print("\nSample data:")
    print(df.head(10))
