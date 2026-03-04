"""
Extract dorm energy consumption data from PDF and convert to CSV
"""
import pdfplumber
import pandas as pd
import re
from datetime import datetime, timedelta

def extract_data_from_pdf(pdf_path):
    """Extract energy consumption data from PDF"""
    all_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            print(f"\n=== Page {page_num} ===")
            print(text[:500])  # Print first 500 chars to see structure
            
            # Try to extract tables
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    all_data.extend(table)
    
    return all_data

def parse_extracted_data(raw_data, output_csv='data/dorm_energy_7days.csv'):
    """Parse and save the extracted data"""
    # Filter out header rows and empty rows
    data_rows = []
    for row in raw_data:
        if row and len(row) >= 2:
            # Skip header rows
            if 'Date' in str(row[0]) or 'Time' in str(row[0]) or 'Timestamp' in str(row[0]):
                continue
            data_rows.append(row)
    
    # Create DataFrame based on detected structure
    if data_rows:
        df = pd.DataFrame(data_rows)
        print(f"\nExtracted DataFrame shape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head(10))
        print(f"\nColumn names: {df.columns.tolist()}")
        
        # Save raw extraction
        df.to_csv(output_csv.replace('.csv', '_raw.csv'), index=False)
        print(f"\nRaw data saved to: {output_csv.replace('.csv', '_raw.csv')}")
    
    return data_rows

if __name__ == "__main__":
    pdf_path = "/Users/vlad/Downloads/Dorm_Energy_Consumption_7_Days.pdf"
    
    print("Extracting data from PDF...")
    raw_data = extract_data_from_pdf(pdf_path)
    
    print(f"\n\nTotal rows extracted: {len(raw_data)}")
    
    if raw_data:
        parse_extracted_data(raw_data)
    else:
        print("No data extracted. The PDF might be image-based or have a different structure.")
