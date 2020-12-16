# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl','rb'))

#contoh data
data = [{
    "marriage" : "1",
    "sex" : "1",
    "edu" : "1",
    "limit" : "1",
    "pay_amt_1" : "1",
    "pay_amt_2" : "1",
    "pay_amt_3" : "1",
    "pay_amt_4" : "1",
    "pay_amt_5" : "1",
    "pay_amt_6" : "1",
    "bil_amt_1" : "1",
    "bil_amt_2" : "1",
    "bil_amt_3" : "1",
    "bil_amt_4" : "1",
    "bil_amt_5" : "1",
    "bil_amt_6" : "1",
    "pay_status_1" : "1",
    "pay_status_2" : "1", 
    "pay_status_3" : "1", 
    "pay_status_4" : "1", 
    "pay_status_5" : "1", 
    "pay_status_6" : "1", 
  }]

@app.route('/', methods=['GET'])
def home():
    return '''
<div style="text-align: center">
    <h1> Back-end untuk Prediksi Default Kartu Kredit</h1>
    <h2>Mata kuliah AI untuk bisnis.</h2><br/>
    <P> Route: /api </p>
</div>'''

@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict([[np.array(data['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'])]])

    # Take the first value of prediction
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)