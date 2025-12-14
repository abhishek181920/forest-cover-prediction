import numpy as np
import pandas as pd
import joblib
from model import ForestCoverClassifier

# Forest cover type mapping
COVER_TYPES = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

def load_model_and_scaler(model_path='models/forest_cover_model.pkl', scaler_path='models/scaler.pkl'):
    """Load the trained model and scaler"""
    model = ForestCoverClassifier.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def preprocess_input(input_data, scaler, feature_names):
    """Preprocess input data using the same scaler used during training"""
    # Convert input to DataFrame with the same columns as training data
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Scale the features
    scaled_features = scaler.transform(input_df)
    
    return scaled_features

def predict_cover_type(input_data):
    """Make a prediction for a single input"""
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    # Get feature names (in a real app, these should be saved during training)
    # For now, we'll use the expected feature names
    feature_names = [
        'Elevation', 'Aspect', 'Slope', 
        'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points'
    ]
    
    # Add binary features (simplified - in reality, these should match your training data)
    # You should adjust this based on your actual feature names
    for i in range(1, 5):
        feature_names.append(f'Wilderness_Area_{i}')
    
    for i in range(1, 41):
        feature_names.append(f'Soil_Type_{i}')
    
    # Ensure input has all required features
    for feature in feature_names:
        if feature not in input_data:
            input_data[feature] = 0  # Default value for missing binary features
    
    # Reorder input data to match training feature order
    ordered_input = {feature: input_data[feature] for feature in feature_names}
    
    # Preprocess input
    processed_input = preprocess_input(ordered_input, scaler, feature_names)
    
    # Make prediction
    prediction = model.predict(processed_input)[0]
    
    return {
        'cover_type': int(prediction),
        'cover_type_name': COVER_TYPES.get(prediction, "Unknown")
    }

if __name__ == "__main__":
    # Example usage
    example_input = {
        'Elevation': 3000,
        'Aspect': 180,
        'Slope': 10,
        'Horizontal_Distance_To_Hydrology': 300,
        'Vertical_Distance_To_Hydrology': 50,
        'Horizontal_Distance_To_Roadways': 2000,
        'Hillshade_9am': 220,
        'Hillshade_Noon': 230,
        'Hillshade_3pm': 200,
        'Horizontal_Distance_To_Fire_Points': 1000,
        'Wilderness_Area_1': 1,
        'Wilderness_Area_2': 0,
        'Wilderness_Area_3': 0,
        'Wilderness_Area_4': 0,
        'Soil_Type_1': 1,
        # ... (set other soil types to 0)
    }
    
    # Set remaining soil types to 0
    for i in range(2, 41):
        example_input[f'Soil_Type_{i}'] = 0
    
    # Make prediction
    result = predict_cover_type(example_input)
    print(f"Predicted Forest Cover Type: {result['cover_type']} - {result['cover_type_name']}")
