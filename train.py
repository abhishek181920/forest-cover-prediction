import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_data, preprocess_data
from model import ForestCoverClassifier
import joblib

def explore_data(df):
    """Perform exploratory data analysis on the dataset"""
    print("\n=== Dataset Information ===")
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {df.shape[1] - 1} (excluding target)")
    
    # Target distribution
    print("\n=== Target Distribution ===")
    target_counts = df['Cover_Type'].value_counts().sort_index()
    print(target_counts)
    
    # Plot target distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Cover_Type', data=df)
    plt.title('Distribution of Forest Cover Types')
    plt.xlabel('Cover Type')
    plt.ylabel('Count')
    plt.savefig('reports/figures/target_distribution.png')
    plt.close()
    
    # Basic statistics for numerical features
    numerical_cols = ['Elevation', 'Aspect', 'Slope', 
                     'Horizontal_Distance_To_Hydrology',
                     'Vertical_Distance_To_Hydrology',
                     'Horizontal_Distance_To_Roadways',
                     'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                     'Horizontal_Distance_To_Fire_Points']
    
    print("\n=== Basic Statistics ===")
    print(df[numerical_cols].describe())

def main():
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_path = os.path.join('data', 'raw', 'forest_cover.csv')
    df = load_data(data_path)
    
    # Explore the data
    explore_data(df)
    
    # Preprocess the data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    
    # Initialize and train the model
    print("Initializing the model...")
    model = ForestCoverClassifier(n_estimators=200, random_state=42)
    model.train(X_train, y_train)
    
    # Evaluate the model
    print("\nEvaluating the model...")
    evaluation = model.evaluate(X_test, y_test)
    
    # Plot feature importances
    print("\nPlotting feature importances...")
    model.plot_feature_importances(feature_names)
    
    # Save the model
    model.save_model()
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
