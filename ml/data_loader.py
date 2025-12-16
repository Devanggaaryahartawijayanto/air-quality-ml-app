"""
Data loading module for air quality forecasting.
Only handles raw data retrieval.
"""

import pandas as pd


def load_air_quality_data(url: str = None) -> pd.DataFrame:
    """
    Load air quality dataset from URL or default GitHub source.
    
    Args:
        url: CSV file URL. If None, uses default GitHub URL.
    
    Returns:
        Raw DataFrame with original column names and data types.
    """
    if url is None:
        url = "https://raw.githubusercontent.com/naufalrahmanu/Dataset_PMLD/refs/heads/main/data-indeks-standar-pencemar-udara-(ispu)-di-provinsi-dki-jakarta-(1759802929504).csv"
    
    df = pd.read_csv(url)
    return df


if __name__ == "__main__":
    # Test loading
    df = load_air_quality_data()
    print(f"✅ Data loaded: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")