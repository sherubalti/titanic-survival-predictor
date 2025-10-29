# app.py
from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None
model_loaded = False

def load_model():
    """Load the model and scaler from file"""
    global model, scaler, model_loaded
    try:
        if not os.path.exists('simple_titanic_model.pkl'):
            print("❌ Model file not found")
            return False
        
        with open('simple_titanic_model.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        
        model = artifacts['model']
        scaler = artifacts['scaler']
        model_loaded = True
        print("✅ Model loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def preprocess_input(pclass, sex, age, fare, embarked):
    """Preprocess input data for prediction"""
    # Convert sex to numeric
    sex_numeric = 1 if sex == 'female' else 0
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        'pclass': [pclass],
        'sex': [sex_numeric],
        'age': [age],
        'fare': [fare],
        'embarked_Q': [1 if embarked == 'Q' else 0],
        'embarked_S': [1 if embarked == 'S' else 0]
    })
    
    return input_data

# Load model at startup
load_model()

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if not model_loaded:
        return render_template('index.html', 
                             error="Model not loaded. Please try again later.", 
                             model_loaded=model_loaded)
    
    try:
        # Get form data
        pclass = int(request.form['pclass'])
        sex = request.form['sex']
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        embarked = request.form['embarked']
        
        # Preprocess input
        input_data = preprocess_input(pclass, sex, age, fare, embarked)
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        survival_prob = probability * 100
        
        # Prepare result
        if prediction == 1:
            result = "SURVIVED"
            result_class = "survived"
            emoji = "😊"
            message = f"Congratulations! You had a {survival_prob:.1f}% chance of survival."
        else:
            result = "DID NOT SURVIVE"
            result_class = "not-survived"
            emoji = "😔"
            message = f"Unfortunately, you had only a {survival_prob:.1f}% chance of survival."
        
        # Passenger info
        passenger_info = {
            'pclass': pclass,
            'sex': 'Female' if sex == 'female' else 'Male',
            'age': age,
            'fare': fare,
            'embarked': embarked,
            'result': result,
            'result_class': result_class,
            'emoji': emoji,
            'message': message,
            'probability': f"{survival_prob:.1f}%"
        }
        
        return render_template('result.html', **passenger_info)
    
    except Exception as e:
        return render_template('index.html', 
                             error=f"Error processing your request: {str(e)}", 
                             model_loaded=model_loaded)

# Health check endpoint for Render
@app.route('/health')
def health():
    return {"status": "healthy", "model_loaded": model_loaded}

if __name__ == '__main__':
    # Use Render's port or default to 5000 for local development
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)