from flask import Flask, request, render_template
import pickle
import numpy as np
import sklearn # type: ignore

app = Flask(__name__)

# Load the model
model = pickle.load(open('iris_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from form
    input_features = [float(x) for x in request.form.values()]
    features = [np.array(input_features)]
    prediction = model.predict(features)
    output = prediction[0]

    return render_template('index.html', prediction_text='The predicted Iris species is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
