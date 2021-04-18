from flask import Flask,render_template,url_for,request
import pandas as pd 
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

def getParameters():
    parameters = []
    #parameters.append(request.form('name'))
    parameters.append(request.form['age'])
    parameters.append(request.form['sex'])
    parameters.append(request.form['cp'])
    parameters.append(request.form['trestbps'])
    parameters.append(request.form['chol'])
    parameters.append(request.form['fbs'])
    parameters.append(request.form['restecg'])
    parameters.append(request.form['thalach'])
    parameters.append(request.form['exang'])
    parameters.append(request.form['oldpeak'])
    parameters.append(request.form['slope'])
    parameters.append(request.form['ca'])
    parameters.append(request.form['thal'])
    return parameters

@app.route('/predict',methods=['POST'])
def predict():
    model = open("data/Heart_model.pkl","rb")
    clfr = joblib.load(model)

    if request.method == 'POST':
        parameters = getParameters()
        inputFeature = np.asarray(parameters).reshape(1,-1)
        my_prediction = clfr.predict(inputFeature)
    return render_template('result.html',prediction = int(my_prediction[0]))

if __name__ == '__main__':
	app.run(debug=True)