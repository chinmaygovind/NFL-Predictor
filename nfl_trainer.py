import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the CSV into a DataFrame
file_path = 'cleaned_scores.csv'  # Replace with your CSV file path
df = pd.read_csv(file_path)

# Step 1: Preprocessing
# Convert `schedule_date` to datetime format
df['schedule_date'] = pd.to_datetime(df['schedule_date'])

# Fill missing values with 0 or 'Unknown' (can be adjusted as needed)
df.fillna({
    'spread_favorite': 0,
    'over_under_line': 0,
    'weather_temperature': 0,
    'weather_wind_mph': 0,
    'weather_humidity': 0,
    'weather_detail': 'Unknown',
}, inplace=True)

# Drop columns not used for modeling
features = df.drop(columns=['schedule_date', 'score_home', 'score_away', 'weather_humidity', 'weather_detail'])
df.dropna(subset=['weather_temperature', 'weather_wind_mph'], inplace=True)

# Define target variables (scores for both teams)
targets = df[['score_home', 'score_away']]

# Define categorical and numerical columns
categorical_columns = ['schedule_season', 'schedule_week', 'team_home', 'team_away', 'stadium'] #
numerical_columns = ['weather_temperature', 'weather_wind_mph'] #, 'over_under_line'
# Preprocessor for categorical and numerical features
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_columns),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
])



# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Fit and transform data using the preprocessor
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Step 2: Build a TensorFlow Model for Multi-Output Regression
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2)  # 2 outputs: score_home and score_away
])

# Compile the model
model.compile(optimizer='adam',
              loss='mse',  # Mean squared error for regression
              metrics=['mae'])  # Mean absolute error for evaluation

# Step 3: Train the Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32
)

# Step 4: Evaluate the Model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.2f}")

# Step 5: Make Predictions
predictions = model.predict(X_test)
for i in range(5):  # Show the first 5 predictions
    print(f"Predicted: {predictions[i]}, Actual: {y_test.iloc[i].values}")

model_save_path = 'nfl_predictor.keras'
model.save(model_save_path)  # Save in TensorFlow SavedModel format
print(f"Model saved to {model_save_path}")

wildcard_games = [
{
    'schedule_season': '2025',  # season value
    'schedule_week': "Wildcard",         # week value
    'team_home': 'Eagles',      # home team
    'team_away': 'Packers',      # away team
    'stadium': 'Lincoln Financial Field',
    'weather_temperature': 36,  # game temperature
    'weather_wind_mph': 6     # game wind speed
},
{
    'schedule_season': '2025',  # season value
    'schedule_week': "Wildcard",         # week value
    'team_home': 'Buccaneers',      # home team
    'team_away': 'Commanders',      # away team
    'stadium': 'Raymond James Stadium',
    'weather_temperature': 62,  # game temperature
    'weather_wind_mph': 6     # game wind speed
},
{
    'schedule_season': '2025',  # season value
    'schedule_week': "Wildcard",         # week value
    'team_home': 'Rams',      # home team
    'team_away': 'Vikings',      # away team
    'weather_temperature': 62,  # game temperature
    'stadium': 'State Farm Stadium',
    'weather_wind_mph': 3     # game wind speed
},
{
    'schedule_season': '2025',  # season value
    'schedule_week': "Wildcard",         # week value
    'team_home': 'Bills',      # home team
    'team_away': 'Broncos',      # away team
    'weather_temperature': 33,  # game temperature
    'weather_wind_mph': 11,     # game wind speed
    'stadium': 'Highmark Stadium'

},
{
    'schedule_season': '2025',  # season value
    'schedule_week': "Wildcard",         # week value
    'team_home': 'Baltimore',      # home team
    'team_away': 'Steelers',      # away team
    'stadium': 'M&T Bank Stadium',
    'weather_temperature': 31,  # game temperature
    'weather_wind_mph': 10     # game wind speed
},
{
    'schedule_season': '2025',  # season value
    'schedule_week': "Wildcard",         # week value
    'team_home': 'Texans',      # home team
    'team_away': 'Chargers',      # away team
    'weather_temperature': 54,  # game temperature
    'weather_wind_mph': 5,     # game wind speed
    'stadium': 'NRG Stadium'
}
]

divisional_games = [
  {
    'schedule_season': '2025',  # season value
    'schedule_week': "Division",         # week value
    'team_home': 'Rams',      # home team
    'team_away': 'Lions',      # away team
    'stadium': 'Ford Field',
    'weather_temperature': 34,  # game temperature
    'weather_wind_mph': 10     # game wind speed
  },  
  {
    'schedule_season': '2025',  # season value
    'schedule_week': "Division",         # week value
    'team_home': 'Buccaneers',      # home team
    'team_away': 'Eagles',      # away team
    'stadium': 'Lincoln Financial Field',
    'weather_temperature': 43,  # game temperature
    'weather_wind_mph': 7     # game wind speed
  },   
  {
    'schedule_season': '2025',  # season value
    'schedule_week': "Division",         # week value
    'team_home': 'Chiefs',      # home team
    'team_away': 'Texans',      # away team
    'stadium': 'Arrowhead Stadium',
    'weather_temperature': 28,  # game temperature
    'weather_wind_mph': 11     # game wind speed
  }, 
  {
    'schedule_season': '2025',  # season value
    'schedule_week': "Division",         # week value
    'team_home': 'Bills',      # home team
    'team_away': 'Ravens',      # away team
    'stadium': 'Highmark Stadium',
    'weather_temperature': 34,  # game temperature
    'weather_wind_mph': 14     # game wind speed
  }, 
]

conference_games = [
    {
    'schedule_season': '2025',  # season value
    'schedule_week': "Conference",         # week value
    'team_home': 'Lions',      # home team
    'team_away': 'Eagles',      # away team
    'stadium': 'Ford Field',
    'weather_temperature': 33,  # game temperature
    'weather_wind_mph': 10     # game wind speed
  },
  {
    'schedule_season': '2025',  # season value
    'schedule_week': "Conference",         # week value
    'team_home': 'Chiefs',      # home team
    'team_away': 'Bills',      # away team
    'stadium': 'Arrowhead Stadium',
    'weather_temperature': 30,  # game temperature
    'weather_wind_mph': 10     # game wind speed
  },
]

superbowl_games = [
    {
    'schedule_season': '2025',  # season value
    'schedule_week': "Superbowl",         # week value
    'team_home': 'Lions',      # home team
    'team_away': 'Chiefs',      # away team
    'stadium': 'Caesars Superdome',
    'weather_temperature': 40,  # game temperature
    'weather_wind_mph': 5     # game wind speed
  },
]
for game in superbowl_games:
    # Convert the dictionary to a DataFrame
    new_game_df = pd.DataFrame([game])

    # Step 2: Preprocess the Input Data (similar to training)
    new_game_preprocessed = preprocessor.transform(new_game_df)

    # Step 3: Make the Prediction
    predicted_scores = model.predict(new_game_preprocessed)
    print(f'{game['schedule_week']}: {game['team_home']} (Home) Vs. {game['team_away']} (Away): {round(predicted_scores[0][0])}-{round(predicted_scores[0][1])}')
