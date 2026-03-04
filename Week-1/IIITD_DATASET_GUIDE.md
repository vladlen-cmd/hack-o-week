# IIITD Dataset Integration - Quick Reference

## What Changed

Your Peak Hour Electricity Analysis project has been updated to use the **I-BLEND dataset** from IIIT Delhi instead of the Kaggle dataset.

## Dataset Information

- **Source**: Nature Scientific Data - I-BLEND Dataset
- **URL**: https://springernature.figshare.com/articles/dataset/Energy_dataset_of_IIITD/6007637
- **Size**: 561 MB (compressed)
- **Duration**: 52 months of continuous data
- **Buildings**: 7 buildings (Building1.csv to Building7.csv)
- **Sampling**: 1-minute intervals
- **Format**: CSV with columns: Unix timestamp, Power (watts), Current, Voltage, Frequency, PowerFactor

## Files Modified

1. **`src/data_processor.py`**
   - Added support for IIITD dataset format (Unix timestamps)
   - Automatic timezone conversion (Asia/Kolkata)
   - Power unit conversion (watts → MW)

2. **`run_dashboard.py`**
   - Updated data path to `data/Building1.csv`

3. **`README.md`**
   - Updated documentation with IIITD dataset information
   - Added download instructions
   - Updated expected results

## Download Status

A download is currently in progress in the background. You can check its status or download manually:

### Option 1: Wait for Background Download
The download is running in the terminal. It's downloading to:
`/Users/vlad/Downloads/Hack-O-Week/January/Week-1/data/iiitd_energy_dataset.zip`

Once complete, extract it:
```bash
cd /Users/vlad/Downloads/Hack-O-Week/January/Week-1/data
unzip iiitd_energy_dataset.zip
```

### Option 2: Manual Download (Faster)
1. Open browser and go to: https://springernature.figshare.com/ndownloader/files/10797959
2. Download will start automatically
3. Move the downloaded file to the `data/` directory
4. Extract: `unzip iiitd_energy_dataset.zip`

### Option 3: Use Synthetic Data (No Download Needed)
The application will automatically generate synthetic data if the dataset is not found. Just run:
```bash
python run_dashboard.py
```

## Using the Dataset

After extraction, you'll have 7 CSV files:
- `Building1.csv` - Academic building
- `Building2.csv` - Academic building
- `Building3.csv` - Academic building
- `Building4.csv` - Residential building
- `Building5.csv` - Residential building
- `Building6.csv` - Residential building
- `Building7.csv` - Residential building

The code is configured to use `Building1.csv` by default. To use a different building, edit `run_dashboard.py` line 27:
```python
data_path = "data/Building2.csv"  # Change to any building
```

## Running the Dashboard

```bash
python run_dashboard.py
```

The application will:
1. Try to load the IIITD dataset
2. Fall back to synthetic data if not found
3. Launch the dashboard at http://localhost:8050

## Dataset Citation

Rashid, H., Singh, P., Stankovic, V. & Stankovic, L. I-BLEND, a campus-scale commercial and residential buildings electrical energy dataset. Sci Data 6, 190015 (2019). https://doi.org/10.1038/sdata.2019.15
