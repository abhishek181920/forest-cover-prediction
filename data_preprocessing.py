import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(filepath):
    """Load the dataset from the given filepath"""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocess the dataset"""
    # Create a copy to avoid SettingWithCopyWarning
    data = df.copy()
    
    # Drop the 'Id' column if it exists as it's not a feature
    if 'Id' in data.columns:
        data = data.drop('Id', axis=1)
    
    # Feature engineering (if needed)
    # For example, you could create new features like:
    data['Distance_To_Hydrology'] = np.sqrt(
        data['Horizontal_Distance_To_Hydrology']**2 + 
        data['Vertical_Distance_To_Hydrology']**2
    )
    
    # Get the list of binary feature columns
    binary_cols = [col for col in data.columns if 'Wilderness_Area' in col or 'Soil_Type' in col]
    
    # Separate features and target
    X = data.drop('Cover_Type', axis=1)
    y = data['Cover_Type']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for later use in predictions
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()

def save_processed_data(X_train, X_test, y_train, y_test, output_dir='data/processed'):
    """Save the processed data to disk"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(f"{output_dir}/X_train.npy", X_train)
    np.save(f"{output_dir}/X_test.npy", X_test)
    np.save(f"{output_dir}/y_train.npy", y_train)
    np.save(f"{output_dir}/y_test.npy", y_test)
    
    print(f"Processed data saved to {output_dir}")

if __name__ == "__main__":
    # This will be used when running the script directly
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python data_preprocessing.py <path_to_raw_data.csv>")
        sys.exit(1)
        
    filepath = sys.argv[1]
    print(f"Loading data from {filepath}...")
    df = load_data(filepath)
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    
    print("Saving processed data...")
    save_processed_data(X_train, X_test, y_train, y_test)
    
    print("Data preprocessing completed successfully!")
