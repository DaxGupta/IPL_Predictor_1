# Import necessary libraries
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS for cross-origin requests
import numpy as np # Import numpy for array handling
import pandas as pd # Import pandas for DataFrame creation

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Define the path to your pickle file
PICKLE_FILE_PATH = 'pipe.pkl'

# Load the machine learning model
model = None
try:
    with open(PICKLE_FILE_PATH, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded successfully from {PICKLE_FILE_PATH}")
except FileNotFoundError:
    print(f"Error: '{PICKLE_FILE_PATH}' not found. Please ensure the model file is in the same directory.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")

@app.route('/')
def home():
    """
    Root endpoint for the Flask application.
    In a production environment, this might serve the index.html file.
    For this setup, the index.html file will be opened directly in the browser,
    and it will make API calls to this Flask backend.
    """
    return "IPL Win Predictor Backend is running. Open index.html in your browser to use the predictor."

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives cricket match details via a POST request,
    uses the loaded ML model to predict winning percentages,
    and returns the probabilities as JSON.

    Expected JSON input:
    {
        "batting_team": "Team A",
        "bowling_team": "Team B",
        "city": "Venue City",
        "target": 180,
        "score": 120,
        "wickets": 3,
        "overs": 12.5
    }
    """
    # Check if the model was loaded successfully
    if model is None:
        return jsonify({"error": "Prediction model not loaded. Please check backend logs."}), 500

    try:
        # Get JSON data from the request body
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided in the request body."}), 400

        # Extract data points from the JSON payload
        batting_team = data.get('batting_team')
        bowling_team = data.get('bowling_team')
        city = data.get('city')
        target_score = data.get('target')
        current_score = data.get('score')
        wickets = data.get('wickets')
        overs_completed = data.get('overs')

        # Basic validation for required fields and types
        if not all([batting_team, bowling_team, city,
                    target_score is not None, current_score is not None,
                    wickets is not None, overs_completed is not None]):
            return jsonify({"error": "Missing one or more required input fields."}), 400

        try:
            target_score = int(target_score)
            current_score = int(current_score)
            wickets = int(wickets)
            overs_completed = float(overs_completed)
        except ValueError:
            return jsonify({"error": "Invalid data type for numerical fields. Please provide valid numbers."}), 400

        # --- Feature Engineering based on the Streamlit app's logic ---
        # These calculations create the features that your model likely expects.
        runs_left = target_score - current_score
        balls_left = 120 - (overs_completed * 6)
        
        # Calculate Current Run Rate (CRR)
        crr = current_score / overs_completed if overs_completed > 0 else 0
        
        # Calculate Required Run Rate (RRR)
        # Handle cases where balls_left is 0 or negative to avoid division by zero
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else (runs_left * 6) 

        # --- Construct the DataFrame for the model ---
        # This is the crucial part to resolve the "Specifying columns using strings" error.
        # The column names MUST EXACTLY match the feature names and their order
        # that your 'pipe.pkl' model was trained on.
        # Based on typical IPL prediction models and the Streamlit app's logic,
        # a common set of features might be:
        # ['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left',
        #  'wickets', 'total_runs_x', 'crr', 'rrr']
        # 'total_runs_x' typically refers to the target score.
        # 'wickets' here is assumed to be wickets fallen. If your model expects
        # wickets remaining, you would use (10 - wickets).

        input_df = pd.DataFrame([[
            batting_team,
            bowling_team,
            city,
            runs_left,
            balls_left,
            wickets, # Wickets fallen
            target_score, # This is 'total_runs_x'
            crr,
            rrr
        ]], columns=[
            'batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left',
            'wickets', 'total_runs_x', 'crr', 'rrr'
        ])

        # Predict winning probabilities
        # The model.predict_proba() method typically returns probabilities for each class.
        # Assuming result[0][0] is bowling team win probability and result[0][1] is batting team win probability.
        # This mapping depends on how your model's classes are ordered during training.
        probabilities = model.predict_proba(input_df)[0]
        
        # Ensure the correct probability is assigned to batting and bowling teams.
        # If your model was trained with batting team as class 1 and bowling team as class 0,
        # then probabilities[1] is batting_team_prob and probabilities[0] is bowling_team_prob.
        # Adjust these indices if your model's class order is different.
        batting_team_prob = round(probabilities[1] * 100, 2)
        bowling_team_prob = round(probabilities[0] * 100, 2)

        # Return the results as JSON
        return jsonify({
            "batting_team_prob": batting_team_prob,
            "bowling_team_prob": bowling_team_prob
        })

    except Exception as e:
        # Catch any unexpected errors during prediction
        print(f"An error occurred during prediction: {e}")
        return jsonify({"error": f"An unexpected error occurred during prediction: {e}"}), 500

if __name__ == '__main__':
    # Run the Flask app on port 5000
    # debug=True allows for automatic reloading on code changes and provides a debugger
    app.run(debug=True, port=5000)
