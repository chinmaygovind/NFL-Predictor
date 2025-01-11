import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the trained model
model_save_path = 'nfl_predictor.keras'
model = tf.keras.models.load_model(model_save_path)

# Define categorical and numerical columns (must match training setup)
categorical_columns = ['schedule_season', 'schedule_week', 'team_home', 'team_away', 'team_favorite_id', 'stadium']
numerical_columns = ['spread_favorite', 'weather_temperature', 'weather_wind_mph']

# Inline preprocessing: Define preprocessor
numerical_scaler = StandardScaler()
categorical_encoder = OneHotEncoder(handle_unknown='ignore')

# Function to preprocess input data
def preprocess_input(input_data, data_reference):
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Scale numerical features
    numerical_data = input_df[numerical_columns]
    numerical_reference = data_reference[numerical_columns]
    numerical_scaler.fit(numerical_reference)
    scaled_numerical = numerical_scaler.transform(numerical_data)
    
    # Encode categorical features
    categorical_data = input_df[categorical_columns]
    categorical_reference = data_reference[categorical_columns]
    categorical_encoder.fit(categorical_reference)
    encoded_categorical = categorical_encoder.transform(categorical_data)
    
    # Concatenate scaled numerical and encoded categorical features
    preprocessed_data = np.hstack([scaled_numerical, encoded_categorical])
    return preprocessed_data

# Example reference data for preprocessing
data_reference_path = 'cleaned_scores.csv'  # Replace with your CSV file path
data_reference = pd.read_csv(data_reference_path)

# Input data example
user_input = {
    'schedule_season': 1966,
    'schedule_week': 2,
    'team_home': 'Miami Dolphins',
    'team_away': 'New York Jets',
    'team_favorite_id': 'Miami Dolphins',
    'spread_favorite': -3.5,
    'stadium': 'Orange Bowl',
    'weather_temperature': 80,
    'weather_wind_mph': 10
}

# Predict scores
preprocessed_data = preprocess_input(user_input, data_reference)
predicted_scores = model.predict(preprocessed_data)

print(f"Predicted Home Team Score: {predicted_scores[0][0]:.2f}")
print(f"Predicted Away Team Score: {predicted_scores[0][1]:.2f}")
