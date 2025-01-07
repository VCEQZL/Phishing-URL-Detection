# importing required libraries
from flask import Flask, request, render_template
import numpy as np
import pickle
from feature import FeatureExtraction
from urllib.parse import quote  # Add this import for url quoting

# Load the model from pickle
with open("pickle/model.pkl", "rb") as file:
    gbc = pickle.load(file)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = quote(request.form["url"])  # Using quote instead of url_quote
        obj = FeatureExtraction(url)
        # Extract features and reshape to ensure it's 2D with 30 features
        x = np.array(obj.getFeaturesList()).reshape(1, -1)  # Automatically get correct shape

        # Debugging: Check the shape of the input data
        print(f"Input data shape: {x.shape}")

        # Ensure the model was trained with the same number of features (30)
        if x.shape[1] != 30:
            return render_template('index.html', xx=-1, error="Feature mismatch error.")

        try:
            # Get prediction
            y_pred = gbc.predict(x)[0]
            y_pro_phishing = gbc.predict_proba(x)[0, 0]
            y_pro_non_phishing = gbc.predict_proba(x)[0, 1]
            pred = "It is {0:.2f}% safe to go".format(y_pro_phishing * 100)
            return render_template('index.html', xx=round(y_pro_non_phishing, 2), url=url, pred=pred)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('index.html', xx=-1, error="An error occurred during prediction.")
    
    return render_template("index.html", xx=-1)

if __name__ == "__main__":
    app.run(debug=True)
