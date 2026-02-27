from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    features = [
        float(request.form["u_q"]),
        float(request.form["coolant"]),
        float(request.form["u_d"]),
        float(request.form["motor_speed"]),
        float(request.form["torque"]),
        float(request.form["i_d"]),
        float(request.form["i_q"]),
        float(request.form["ambient"])
    ]

    scaled = scaler.transform([features])
    prediction = model.predict(scaled)

    return render_template("index.html",
                           prediction_text=f"Predicted Temperature: {round(prediction[0],2)} °C")

if __name__ == "__main__":
    app.run(debug=True)