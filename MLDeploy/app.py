import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

flask_app = Flask(__name__)
CLFmodel = pickle.load(open('model.pkl', 'rb'))

@flask_app.route("/")
def index():
    return render_template("index.html")


@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [int(x) for x in request.form.values()]
    features = [np.array(float_features)]
    result = CLFmodel.predict(features)
    return render_template("index.html", predicted_text = result)

if __name__ == "__main__":
    flask_app.run(debug = True)