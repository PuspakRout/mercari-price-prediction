from code import model_training, test_frame
from flask import Flask, render_template, url_for, request
import joblib
import pandas as pd

app = Flask(__name__)

# model = joblib.load('objects/model.obj')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def result():
    if request.method == 'POST':
        df = pd.DataFrame(request.form, index=[0])
        joblib.dump(df,'objects/df')
        feature_matrix = test_frame(df)
        result = model.predict(feature_matrix)[0]
        result = f'Predicted Price {result:0.2f} USD'
        return render_template('result.html', result=result)
    
    

if __name__ == '__main__':
    #model_training()
    model = joblib.load('objects/model.obj')
    app.run()
    
    
