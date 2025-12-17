
import pandas as pd
import os

DATA_PATH = 'data/historical_data.csv'

def parse_date(row):
    try:
        p_str = str(row['periode_data'])
        year = int(p_str[:4])
        month = int(p_str[4:6])
        day = int(row['tanggal'])
        return pd.Timestamp(year=year, month=month, day=day)
    except:
        return pd.NaT

print(f"Reading {DATA_PATH}...")
try:
    df = pd.read_csv(DATA_PATH)
    print("Columns:", df.columns.tolist())
    print("First 5 rows raw:")
    print(df[['periode_data', 'tanggal', 'pm_duakomalima']].head())
    
    df['date'] = df.apply(parse_date, axis=1)
    df = df.dropna(subset=['date']).sort_values('date')
    
    pollutants = ['pm_sepuluh', 'pm_duakomalima']
    for col in pollutants:
        if col in df.columns:
            # Check for non-numeric before conversion
            non_numeric = df[pd.to_numeric(df[col], errors='coerce').isna()]
            if not non_numeric.empty:
                print(f"Found non-numeric values in {col}:")
                print(non_numeric[col].head())
            
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print("\nLast 30 rows after processing:")
    print(df[['date', 'pm_duakomalima']].tail(30))
    
except Exception as e:
    print(f"Error: {e}")
