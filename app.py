from flask import Flask, request, render_template
import sys
import os
sys.path.append(os.path.join(os.getcwd(),'src'))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData


app = Flask(__name__)

## Route for home page

@app.route('/')
def home():
    return render_template('index.html')


## Route for prediction
@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        pipeline = PredictPipeline()
        preds = pipeline.predict(data.get_data_as_data_frame())
        return render_template('home.html', preds = preds[0])
    

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
