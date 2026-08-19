from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("model_export/model.joblib")


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "IRIS API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = [[
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"],
    ]]
    prediction = model.predict(features)
    return jsonify({"prediction": prediction[0]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)