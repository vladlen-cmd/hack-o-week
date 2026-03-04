"""
Convert extracted PDF data to proper CSV format for the dashboard
"""
import pandas as pd
from datetime import datetime

# Read the raw extracted data
df_raw = pd.read_csv('data/dorm_energy_7days_raw.csv')

# Rename columns appropriately
df_raw.columns = ['datetime', 'hour', 'energy_kwh']

# Convert datetime string to proper datetime objects
df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])

# Convert kWh to MW (assuming 1 kWh per hour = 0.001 MW average over that hour)
# Or we can keep it in kWh and change the units display
df_raw['load_mw'] = df_raw['energy_kwh'] / 1000  # Convert kWh to MWh (which is MW average)

# Select only datetime and load_mw columns for compatibility
df_final = df_raw[['datetime', 'load_mw']].copy()

# Save to CSV
output_file = 'data/dorm_energy_7days.csv'
df_final.to_csv(output_file, index=False)

print(f"✅ Processed data saved to: {output_file}")
print(f"\nDataset Summary:")
print(f"  Records: {len(df_final)}")
print(f"  Date Range: {df_final['datetime'].min()} to {df_final['datetime'].max()}")
print(f"  Load Range: {df_final['load_mw'].min():.4f} - {df_final['load_mw'].max():.4f} MW")
print(f"\nFirst few rows:")
print(df_final.head(10))
print(f"\nLast few rows:")
print(df_final.tail(10))
