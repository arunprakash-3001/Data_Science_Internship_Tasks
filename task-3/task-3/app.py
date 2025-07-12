from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open("iris_model.pkl", "rb"))

@app.route('/')
def home():
    return "🌸 Iris Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]
    prediction = model.predict([features])
    return jsonify({'species': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
