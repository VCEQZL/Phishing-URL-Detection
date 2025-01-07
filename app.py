import pandas as pd
from flask import Flask, request, render_template
import pickle
from feature import FeatureExtraction
from urllib.parse import quote

# Load the model
file = open("pickle/model.pkl", "rb")
gbc = pickle.load(file)
file.close()

# Print model type and scikit-learn version for debugging
print(f"Loaded model type: {type(gbc)}")
import sklearn
print(f"scikit-learn version: {sklearn.__version__}")

# Define the actual feature names that the model expects
expected_features = [
    "AbnormalURL", "AgeofDomain", "AnchorURL", "DNSRecording", "DisableRightClick",
    "DomainAge", "HasHTTPs", "HasQuery", "IPAddressInURL", "NumDots", "NumSlashes", 
    "NumSubdomains", "PathLength", "ShorteningService", "SuspiciousDomain", 
    "TopLevelDomain", "URLLength", "URLDepth", "URLHostname", "URLPath", 
    "URLPort", "URLProtocol", "URLQueryString", "URLSubdomains", "URLTitle", 
    "IsExternal", "HasFavicon", "HasMetaTags", "HasSSL", "HasWhois", "IsPhishing"
]

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = quote(request.form["url"])  # Using quote instead of url_quote
        obj = FeatureExtraction(url)
        feature_list = obj.getFeaturesList()  # Extract features
        
        # Ensure that we have the same number of features
        print(f"Feature list length: {len(feature_list)}")

        if len(feature_list) != len(expected_features):
            # Debugging print of missing or extra features
            print(f"Extracted features: {feature_list}")
            error_message = f"Expected {len(expected_features)} features, but got {len(feature_list)}."
            print(error_message)
            return render_template('index.html', xx=-1, url=url, error=error_message)

        # Optionally: Add missing features to match the expected number
        if len(feature_list) < len(expected_features):
            missing_features_count = len(expected_features) - len(feature_list)
            print(f"Missing {missing_features_count} features. Adding default values.")
            # You could add placeholders such as None or 0 to fill in missing features
            feature_list.extend([0] * missing_features_count)  # Replace `0` with appropriate default values
            
        # Create a DataFrame with the correct column names
        x_df = pd.DataFrame([feature_list], columns=expected_features)

        try:
            y_pred = gbc.predict(x_df)[0]  # Prediction
            y_pro_phishing = gbc.predict_proba(x_df)[0, 0]  # Probability for phishing
            y_pro_non_phishing = gbc.predict_proba(x_df)[0, 1]  # Probability for non-phishing
            pred = "It is {0:.2f}% safe to go".format(y_pro_phishing * 100)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('index.html', xx=-1, url=url, error=str(e))
        
        return render_template('index.html', xx=round(y_pro_non_phishing, 2), url=url)
    
    return render_template("index.html", xx=-1)

if __name__ == "__main__":
    app.run(debug=True)
