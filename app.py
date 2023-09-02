from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)   
model=pickle.load(open('AmazonReview.pkl','rb'))

from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features = 1500)
corpus=pd.read_csv('AmazonCorpus.csv')
corpus1=corpus['corpus'].tolist()
corpus1 = [str(text) if pd.notna(text) else '' for text in corpus1]
X = cv.fit_transform(corpus1).toarray()


@app.route('/')
def home():

    return render_template("index.html")


@app.route('/predict',methods=['GET'])

def predict():

    text = (request.args.get('text'))
    text=[text]
    input_data = cv.transform(text).toarray()


    prediction = model.predict(input_data)
    if prediction == 1:
      return render_template('index.html', prediction_text='Positive')

    else:
      return render_template('index.html', prediction_text='Negative')
#------------------------------About us-------------------------------------------
@app.route('/aboutusnew')
def aboutusnew():
    return render_template('aboutusnew.html')

@app.route('/moreprojects')
def moreprojects():
    return render_template('moreprojects.html')

@app.route('/gallery')
def gallery():
    return render_template('gallery.html')

if __name__ == "__main__":
    app.run(debug=True)
