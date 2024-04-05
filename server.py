from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from huggingface_hub import hf_hub_download
import joblib
import json
import numpy as np

app = Flask(__name__)

REPO_ID = "sardal/obesity_risk"
FILENAME = "lgbm_classifier.pkl"

model = joblib.load(
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    features = [x for x in request.form.values()]
    data = {
        'Age': int(features[4]),
        'Height': float(features[2]),
        'Weight': int(features[1]),
        'FCVC': 3,
        'NCP': 2,
        'CH2O': 2,
        'FAF': int(features[3]),
        'TUE': 2,
        'Gender_Male': 1 if features[0] == "Male" else 0,  # One-hot encoded feature, assuming Male
        'family_history_with_overweight_yes': 1,  # One-hot encoded feature, assuming Yes
        'FAVC_yes': 1,  # One-hot encoded feature, assuming Yes
        'CAEC_Frequently': 0,  # One-hot encoded feature, assuming Not Frequently
        'CAEC_Sometimes': 1,  # One-hot encoded feature, assuming Sometimes
        'CAEC_no': 0,  # One-hot encoded feature, assuming No
        'SMOKE_yes': 0,  # One-hot encoded feature, assuming No
        'SCC_yes': 0,  # One-hot encoded feature, assuming No
        'CALC_Frequently': 0,  # One-hot encoded feature, assuming Not Frequently
        'CALC_Sometimes': 1,  # One-hot encoded feature, assuming Sometimes
        'CALC_no': 0,  # One-hot encoded feature, assuming No
        'MTRANS_Bike': 0,  # One-hot encoded feature, assuming not Bike
        'MTRANS_Motorbike': 0,  # One-hot encoded feature, assuming not Motorbike
        'MTRANS_Public_Transportation': 1,  # One-hot encoded feature, assuming Public Transportation
        'MTRANS_Walking': 0  # One-hot encoded feature, assuming not Walking
    }

    sample_df = pd.DataFrame(data, index=[0])

    # Load the trained model from the pickle file
    with open('lgbm_classifier.pkl', 'rb') as f:
        model = pickle.load(f)

    # Use the loaded model to make predictions
    predictions = model.predict(sample_df)

    return render_template('index.html', prediction_text='Obesity Risk is: {}'.format(predictions[0]))

if __name__ == '__main__':
    app.run(debug=True)
