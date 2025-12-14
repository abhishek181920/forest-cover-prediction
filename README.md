# Forest Cover Type Prediction

This project implements a machine learning model to predict forest cover types in the Roosevelt National Forest of northern Colorado based on various geographic and environmental features.

## Dataset

The dataset contains cartographic variables from the US Forest Service (USFS) Resource Information System (RIS) data. It includes information about 30m x 30m patches of forest, with the goal of predicting the forest cover type based on the given features.

### Features

- **Elevation**: Elevation in meters
- **Aspect**: Aspect in degrees azimuth
- **Slope**: Slope in degrees
- **Horizontal_Distance_To_Hydrology**: Horizontal distance to nearest surface water features
- **Vertical_Distance_To_Hydrology**: Vertical distance to nearest surface water features
- **Horizontal_Distance_To_Roadways**: Horizontal distance to nearest roadway
- **Hillshade_9am/Noon/3pm**: Hillshade index at different times
- **Horizontal_Distance_To_Fire_Points**: Horizontal distance to nearest wildfire ignition points
- **Wilderness_Area**: 4 binary columns indicating wilderness area designation
- **Soil_Type**: 40 binary columns indicating soil type designation

### Target Variable (Cover_Type)

1. Spruce/Fir
2. Lodgepole Pine
3. Ponderosa Pine
4. Cottonwood/Willow
5. Aspen
6. Douglas-fir
7. Krummholz

## Project Structure

```
forest_cover_prediction/
├── data/
│   ├── raw/               # Raw dataset
│   └── processed/         # Processed data files
├── models/                # Saved models and scalers
├── reports/
│   └── figures/           # Visualization outputs
├── data_preprocessing.py  # Data loading and preprocessing
├── model.py              # Model definition
├── train.py              # Training script
├── predict.py            # Prediction script
└── requirements.txt      # Python dependencies
```

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd forest_cover_prediction
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare the Data

Place your dataset in the `data/raw/` directory with the filename `forest_cover.csv`.

### 2. Train the Model

```bash
python train.py
```

This will:
- Preprocess the data
- Train a Random Forest classifier
- Evaluate the model
- Save the trained model and scaler
- Generate visualizations

### 3. Make Predictions

You can use the `predict.py` script to make predictions with the trained model. The script includes an example of how to format the input data.

```bash
python predict.py
```

## Model Performance

After training, the model will output performance metrics including:
- Accuracy score
- Classification report (precision, recall, f1-score)
- Confusion matrix
- Feature importance plot

## Customization

- **Model Parameters**: Adjust the model parameters in `train.py` (e.g., number of trees, max depth)
- **Feature Engineering**: Modify the `preprocess_data` function in `data_preprocessing.py` to add or modify features
- **Visualizations**: Customize the plotting functions in `model.py`

## Dependencies

- Python 3.7+
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib

## License

This project is licensed under the MIT License - see the LICENSE file for details.
