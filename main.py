from flask import Flask, render_template, request
import numpy as np
import pickle

# 1. Initialize Flask App
# Flask looks for templates in the 'Template' folder automatically
app = Flask(__name__)

# --- Load the Trained Model Globally ---
MODEL_FILENAME = 'crop_model.pkl'

try:
    with open(MODEL_FILENAME, 'rb') as file:
        model = pickle.load(file)
    print(f"✅ Model '{MODEL_FILENAME}' loaded successfully.")
except FileNotFoundError:
    model = None
    print(f"❌ WARNING: Model file '{MODEL_FILENAME}' not found. Prediction will fail.")


# ---------------------------------------------

# 2. Home Route: Serves the HTML form
@app.route('/')
def home():
    # Renders index.html from the 'Template' folder
    return render_template('index.html')


# 3. Prediction Route: The connection point
@app.route('/predict', methods=['POST'])
def predict():
    """Receives form data, predicts the crop, and sends the result back to the HTML."""

    try:
        # 3.1. Extract the 7 input features from the form
        # We use request.form.get() to safely retrieve values
        form_data = {
            'N': request.form.get('Nitrogen'),
            'P': request.form.get('Phosphorus'),
            'K': request.form.get('Potassium'),
            'temp': request.form.get('Temperature'),
            'humidity': request.form.get('Humidity'),
            'ph': request.form.get('pH'),
            'rainfall': request.form.get('Rainfall')
        }
        
        # --- Input Validation and Conversion ---
        
        # 3.2. Check for missing or empty values (This is what prevents your ValueError)
        for key, value in form_data.items():
            if not value:
                # Return an error message to the user on the same page
                return render_template('index.html', prediction_text=f"❌ Error: The '{key}' field is required. Please fill in all inputs.")
            
        # Convert valid strings to floats
        N = float(form_data['N'])
        P = float(form_data['P'])
        K = float(form_data['K'])
        temp = float(form_data['temp'])
        humidity = float(form_data['humidity'])
        ph = float(form_data['ph'])
        rainfall = float(form_data['rainfall'])

        # 3.3. Prepare the data for the model (must be a 2D array)
        features = np.array([[N, P, K, temp, humidity, ph, rainfall]])

        prediction_result = "Prediction Error: Model not loaded."

        if model:
            # 3.4. Run the prediction
            predicted_crop = model.predict(features)[0]
            prediction_result = f'✅ The optimal crop is: {predicted_crop}'
            
        # 3.5. Return the result to index.html using the Jinja variable 'prediction_text'
        return render_template('index.html', prediction_text=prediction_result)

    except ValueError:
        # This catches errors if the input is a valid string but not a number (e.g., 'abc')
        return render_template('index.html', prediction_text="❌ Error: All inputs must be valid numeric values.")

    except Exception as e:
        # Catches any other unexpected errors (e.g., problem with model loading)
        error_message = f"An unexpected error occurred: {str(e)}"
        return render_template('index.html', prediction_text=f"❌ {error_message}")


if __name__ == '__main__':
    # Running the app with debug mode on for development
    app.run(debug=True)