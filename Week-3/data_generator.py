import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_library_data(days=90, start_date=None):
    """
    Generate synthetic library energy consumption data with exam periods.
    
    Simulates a university library with distinct energy patterns during
    normal operations vs exam periods (midterms, finals).
    
    Args:
        days: Number of days to generate data for
        start_date: Starting date (defaults to 90 days ago)
    
    Returns:
        DataFrame with timestamp, occupancy, is_exam_period, temperature_outside, electricity_kwh
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)
    
    # Define exam periods (roughly: midterms around day 30-37, finals around day 75-85)
    exam_ranges = [
        (30, 37),   # Midterm exams
        (75, 85),   # Final exams
    ]
    
    timestamps = [start_date + timedelta(hours=i) for i in range(days * 24)]
    data = []
    
    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.weekday()
        day_number = (ts - start_date).days
        
        # Determine if we're in an exam period
        is_exam = any(start <= day_number <= end for start, end in exam_ranges)
        
        # --- OCCUPANCY MODEL ---
        if is_exam:
            # During exams: library is packed, stays open late
            if 0 <= hour < 6:
                # Late-night study: still many students
                base_occupancy = 60
            elif 6 <= hour < 8:
                base_occupancy = 40
            elif 8 <= hour < 12:
                base_occupancy = 150
            elif 12 <= hour < 14:
                # Lunch dip but still busy
                base_occupancy = 120
            elif 14 <= hour < 20:
                # Afternoon/evening peak during exams
                base_occupancy = 180
            elif 20 <= hour < 23:
                # Late evening still very busy
                base_occupancy = 140
            else:
                base_occupancy = 80
            
            # Weekends during exams are still busy
            if day_of_week >= 5:
                base_occupancy = int(base_occupancy * 0.85)
        else:
            # Normal periods
            if day_of_week >= 5:  # Weekend
                if 10 <= hour <= 18:
                    base_occupancy = 30
                else:
                    base_occupancy = 8
            elif 0 <= hour < 7:
                base_occupancy = 5
            elif 7 <= hour < 9:
                base_occupancy = 35
            elif 9 <= hour < 12:
                base_occupancy = 70
            elif 12 <= hour < 14:
                base_occupancy = 55
            elif 14 <= hour < 18:
                base_occupancy = 80
            elif 18 <= hour < 21:
                base_occupancy = 50
            else:
                base_occupancy = 15
        
        # Add noise
        occupancy = max(0, int(base_occupancy + np.random.normal(0, base_occupancy * 0.12)))
        
        # --- TEMPERATURE MODEL ---
        # Simulate outdoor temperature with seasonal drift and daily cycle
        seasonal_temp = 22 + 8 * np.sin(2 * np.pi * day_number / 90)
        daily_cycle = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
        temperature = seasonal_temp + daily_cycle + np.random.normal(0, 1.5)
        temperature = round(temperature, 1)
        
        # --- ELECTRICITY MODEL ---
        # Base load: lighting, servers, emergency systems
        base_load = 8.0  # kWh
        
        # Lighting load (depends on hour)
        if 7 <= hour <= 22:
            lighting = 4.5
        elif is_exam and (0 <= hour < 7 or hour > 22):
            lighting = 3.0  # Reduced but still on during exam late-night
        else:
            lighting = 1.0  # Minimal overnight
        
        # Occupancy-dependent load (computers, charging stations, elevators)
        occupancy_load = occupancy * 0.08  # ~80W per person average
        
        # HVAC load (temperature-dependent)
        temp_diff = abs(temperature - 22)  # Target indoor temp 22°C
        hvac_load = 2.0 + temp_diff * 0.4
        if is_exam:
            hvac_load *= 1.3  # More bodies = more cooling needed
        
        # Total with some noise
        electricity = base_load + lighting + occupancy_load + hvac_load
        electricity = max(2.0, electricity + np.random.normal(0, 0.5))
        
        data.append({
            'timestamp': ts,
            'occupancy': occupancy,
            'is_exam_period': int(is_exam),
            'temperature_outside': temperature,
            'electricity_kwh': round(electricity, 2)
        })
    
    return pd.DataFrame(data)


if __name__ == '__main__':
    print("=" * 60)
    print("Library Energy Data Generator")
    print("=" * 60)
    
    df = generate_library_data(days=90)
    
    output_file = 'library_data.csv'
    df.to_csv(output_file, index=False)
    
    exam_days = df[df['is_exam_period'] == 1]
    normal_days = df[df['is_exam_period'] == 0]
    
    print(f"✓ Generated {len(df)} records ({len(df) // 24} days)")
    print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"✓ Exam period records: {len(exam_days)} ({len(exam_days) // 24} days)")
    print(f"✓ Normal period records: {len(normal_days)} ({len(normal_days) // 24} days)")
    print(f"✓ Occupancy range: {df['occupancy'].min()} – {df['occupancy'].max()}")
    print(f"✓ Electricity range: {df['electricity_kwh'].min():.2f} – {df['electricity_kwh'].max():.2f} kWh")
    print(f"✓ Avg electricity (exam): {exam_days['electricity_kwh'].mean():.2f} kWh")
    print(f"✓ Avg electricity (normal): {normal_days['electricity_kwh'].mean():.2f} kWh")
    print(f"✓ Saved to {output_file}")
    
    print("\nSample data:")
    print(df.head(10).to_string(index=False))
