# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl','rb'))

#contoh data
contoh = {
    "marriage" : 1,
    "sex" : 1,
    "edu" : 1,
    "limit" : 1,
    "age" : 37,
    "pay_amt_1" : 2000.0,
    "pay_amt_2" : 2019.0,
    "pay_amt_3" : 1200.0,
    "pay_amt_4" : 1100.0,
    "pay_amt_5" : 1069.0,
    "pay_amt_6" : 1000.0,
    "bil_amt_1" : 46990.0,
    "bil_amt_2" : 48233.0,
    "bil_amt_3" : 49291.0,
    "bil_amt_4" : 28314.0,
    "bil_amt_5" : 28959.0,
    "bil_amt_6" : 29547.0,
    "pay_status_1" : 0,
    "pay_status_2" : 0, 
    "pay_status_3" : 0, 
    "pay_status_4" : 0, 
    "pay_status_5" : 0, 
    "pay_status_6" : 0, 
    "sex_marriage" : 2
  }

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

    # # Make prediction using model loaded from disk as per the data.
    df = pd.DataFrame(data, index=[0])
    test = df[["limit","sex","edu","marriage","age",
        "pay_status_1","pay_status_2","pay_status_3","pay_status_4","pay_status_5","pay_status_6",
        "bil_amt_1","bil_amt_2","bil_amt_3","bil_amt_4","bil_amt_5","bil_amt_6",
        "pay_amt_1","pay_amt_2","pay_amt_3","pay_amt_4","pay_amt_5","pay_amt_6",
        "sex_marriage"]]
    prediction = model.predict(test)

    # Take the first value of prediction
    output = prediction[0]

    # terminal output debug
    # app.logger.info( output )

    return jsonify(str(output))

if __name__ == '__main__':
    app.run(port=5000, debug=True)