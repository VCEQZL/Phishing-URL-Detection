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

# Default values for missing features
default_feature_values = {feature: 0 for feature in expected_features}

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = quote(request.form["url"])  # Using quote instead of url_quote
        obj = FeatureExtraction(url)
        feature_list = obj.getFeaturesList()  # Extract features
        
        # Ensure that we have the same number of features
        print(f"Feature list length: {len(feature_list)}")
        print(f"Extracted features: {feature_list}")

        if len(feature_list) != len(expected_features):
            # Fill in missing features with default values
            filled_features = {name: default_feature_values[name] for name in expected_features}
            
            # Update with extracted feature values
            for i, feature_name in enumerate(expected_features[:len(feature_list)]):
                filled_features[feature_name] = feature_list[i]

            # Ensure feature_list matches the order of expected_features
            feature_list = [filled_features[name] for name in expected_features]
            print(f"Adjusted feature list: {feature_list}")

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
