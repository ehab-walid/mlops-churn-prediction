import os
import pandas as pd

def fetch_data():
    """Downloads the raw dataset from a reliable public URL."""
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
    # Create the folders if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    
    print("Downloading dataset...")
    df = pd.read_csv(url)
    
    # Save it locally
    raw_path = "data/raw/churn_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"Saved raw data to {raw_path}")
    return df

def clean_data(df):
    """Cleans the dataset and prepares it for ML."""
    print("Cleaning data...")
    
    # Drop the ID column (it has no predictive power)
    df = df.drop(columns=['customerID'])
    
    # The 'TotalCharges' column has blank spaces that pandas reads as strings. 
    # We force it to be numbers, and turn blanks into NaN (Not a Number)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop any rows that have missing values now
    df = df.dropna()
    
    return df

def save_processed_data(df):
    """Saves the cleaned dataset."""
    os.makedirs("data/processed", exist_ok=True)
    processed_path = "data/processed/churn_processed.csv"
    df.to_csv(processed_path, index=False)
    print(f"Saved cleaned data to {processed_path}")

if __name__ == "__main__":
    # This is the "engine" of the script. When you run this file, 
    # it executes these three functions in order.
    raw_df = fetch_data()
    clean_df = clean_data(raw_df)
    save_processed_data(clean_df)
    print("Phase 2 complete!")