
import os
import pandas as pd
import requests

def download_data():
    url = "https://raw.githubusercontent.com/naufalrahmanu/Dataset_PMLD/refs/heads/main/data-indeks-standar-pencemar-udara-(ispu)-di-provinsi-dki-jakarta-(1759802929504).csv"
    output_dir = "data"
    output_file = os.path.join(output_dir, "historical_data.csv")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
        
    print(f"Downloading data from {url}...")
    try:
        df = pd.read_csv(url)
        df.to_csv(output_file, index=False)
        print(f"✅ Data saved to {output_file}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"❌ Error downloading data: {e}")

if __name__ == "__main__":
    download_data()
