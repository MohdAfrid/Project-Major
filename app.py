from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)   
model=pickle.load(open('ARSA.pkl','rb'))

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer  = TfidfVectorizer(max_features=5000,ngram_range=(2,2))
corpus=pd.read_csv('ARSAcorpus.csv')
corpus1=corpus['corpus'].tolist()
X = tfidf_vectorizer.fit_transform(corpus1).toarray()


@app.route('/')
def home():
  
    return render_template("index.html")

  
@app.route('/predict',methods=['GET'])

def predict():
    
    text = (request.args.get('text'))
    text=[text]
    input_data = tfidf_vectorizer.transform(text).toarray()
    
    
    prediction = model.predict(input_data)
    
    if prediction == 2:
      return render_template('index.html', prediction_text='Negative')
    elif prediction == 1:
      return render_template('index.html', prediction_text='Neutral')     
    else:
      return render_template('index.html', prediction_text='Positive')
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
