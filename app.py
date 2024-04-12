import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
cb = pickle.load(open('breast_c.pkl', 'rb'))
mms = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/')
def home():
  return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
  
  features = [float(x) for x in request.form.values()]
  final_features = [np.array(features)]
  final_features = mms.transform(final_features) 
  df = pd.DataFrame(final_features)   
  prediction = cb.predict(df)
  

  if prediction == 0:
      res_val = "Breast cancer"
  else:
      res_val = "No Breast cancer"


  return render_template('home.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
  app.run(debug=True)