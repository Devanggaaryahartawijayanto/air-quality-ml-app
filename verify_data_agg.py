
import sys
import os
import pandas as pd

# Add current dir to path
sys.path.append(os.getcwd())

try:
    from app import preprocess_and_load_data
    print("✅ app.py imported successfully")
except ImportError as e:
    print(f"❌ Failed to import app.py: {e}")
    sys.exit(1)

try:
    print("Running preprocess_and_load_data()...")
    df = preprocess_and_load_data()
    
    print(f"DataFrame Shape: {df.shape}")
    print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    
    # Check for duplicates
    dup_count = df.duplicated(subset=['date']).sum()
    if dup_count == 0:
        print("✅ No duplicate dates found.")
    else:
        print(f"❌ Found {dup_count} duplicate dates!")
        
    print("\nLast 5 rows:")
    print(df[['date', 'pm_duakomalima']].tail())
    
    # Check if values are reasonable (e.g. not 0-25 range if 71 is expected)
    last_val = df['pm_duakomalima'].iloc[-1]
    print(f"\nLast Value: {last_val}")
    if last_val > 30:
         print("✅ Value seems reasonable (matches expected ~70 range from raw data).")
    else:
         print("⚠️ Value seems low. Check if aggregation caused issues.")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
