from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('credit_card_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from the form
        income = float(request.form['income'])
        employment_years = float(request.form['employment_years'])
        family_members = int(request.form['family_members'])
        credit_history = float(request.form['credit_history'])  # proxy 0–1

        # Create feature array 
        features = np.array([[income, employment_years, family_members, credit_history]])

        # Predict (0 = not approved, 1 = approved)
        prediction = model.predict(features)[0]
        result = "Approved" if prediction == 1 else "Not Approved"

        # Disclaimer
        disclaimer = (
            "Important Note: This is a predictive model based on historical data patterns. "
            "It is not a guarantee of approval and should not replace official credit applications. "
            "Credit decisions involve many factors not included here."
        )

        return render_template('index.html', result=result, disclaimer=disclaimer)
    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}. Please check your inputs.")

if __name__ == '__main__':
    app.run(debug=True)