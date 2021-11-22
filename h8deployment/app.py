from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
#from sklearn.preprocessing import StandardScaler 
from model.model import scaler

app = Flask(__name__, template_folder="templates")

model = pickle.load(open("model/model_classifier.pkl", "rb"))

@app.route("/")
def main():
    return render_template("main.html")

@app.route("/predict", methods =["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = np.array(float_features).reshape(1,-1)
    prediction = model.predict(scaler.transform(features)) 

    output = {0: 'Dont Deposit', 1: 'Will Deposit'}

    return render_template("main.html", prediction_text = "The people {}".format(output[prediction[0]]))


if __name__ == "__main__":
    app.run(debug=True)    