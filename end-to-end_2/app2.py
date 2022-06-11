import numpy
import pickle

import numpy as np
from flask import Flask, render_template, request, jsonify
import sklearn

app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('car.html')


@app.route('/predict',methods=['POST','GET'])
def results():
    Year = float(request.form['Year'])
    Mileage = float(request.form['Mileage'])

    X = np.array([[Year, Mileage]])
    model2 = pickle.load(open('model2.pkl','rb'))
    Y_prediction = model2.predict(X)
    return jsonify({'model2 Prediction': float(Y_prediction)})

if __name__ =='__main__':
    app.run(debug = True, port= 2020)


