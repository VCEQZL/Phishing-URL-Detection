import pandas as pd
from flask import Flask, request, render_template
import numpy as np
import pickle
from feature import FeatureExtraction
from urllib.parse import quote

# Load the model
with open("pickle/model.pkl", "rb") as file:
    gbc = pickle.load(file)

# Print model type and scikit-learn version for debugging
print(f"Loaded model type: {type(gbc)}")
import sklearn
print(f"scikit-learn version: {sklearn.__version__}")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = quote(request.form["url"])  # Using quote instead of url_quote
        obj = FeatureExtraction(url)
        feature_list = obj.getFeaturesList()  # Extract features

        # Ensure that we have 30 features
        print(f"Feature list length: {len(feature_list)}")
        
        # Assuming these are the feature names (replace this with the actual 30 feature names if needed)
        feature_columns = [f'feature_{i+1}' for i in range(30)]  # Generating 30 feature names
        
        # Check if the length of the feature list is correct
        if len(feature_list) != 30:
            return render_template('index.html', xx=-1, url=url, error="Feature list has an incorrect number of features.")
        
        # Create a DataFrame with the correct column names
        x_df = pd.DataFrame([feature_list], columns=feature_columns)

        try:
            y_pred = gbc.predict(x_df)[0]  # Prediction
            y_pro_phishing = gbc.predict_proba(x_df)[0, 0]  # Probability for phishing
            y_pro_non_phishing = gbc.predict_proba(x_df)[0, 1]  # Probability for non-phishing
            pred = "It is {0:.2f}% safe to go".format(y_pro_phishing * 100)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('index.html', xx=-1, url=url)
        
        return render_template('index.html', xx=round(y_pro_non_phishing, 2), url=url)
    
    return render_template("index.html", xx=-1)

if __name__ == "__main__":
    app.run(debug=True)
