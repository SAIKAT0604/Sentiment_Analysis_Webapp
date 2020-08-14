from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import os
import keras
import tensorflow as tf
from numpy import array
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model

app = Flask(__name__)



@app.route('/')
def home():

    return render_template("home.html")

@app.route('/predict', methods = ['POST', "GET"])
def predict():
    if request.method=='POST':
        try:
            text = request.form['text']
            sentiment = ''
            MAXLEN = 500
            word_index = imdb.get_word_index()

            def encode_text(text):
              tokens = keras.preprocessing.text.text_to_word_sequence(text)
              tokens = [word_index[word] if word in word_index else 0 for word in tokens]
              return sequence.pad_sequences([tokens], MAXLEN)[0]

            model = load_model('sentiment_analysis_model_new.h5')
            encoded_text = encode_text(text)
            pred = np.zeros((1,MAXLEN))
            pred[0] = encoded_text
            result = model.predict(pred)

            if result[0] <0.5 :
                sentiment = 'Negative'
                img_filename = ('static/img_pool/Sad_Emoji.png')
            else:
                sentiment = 'Positive'
                img_filename = ('static/img_pool/Smiling_Emoji.png')
        except ValueError:
            return "Please check if the values are entered correctly"
    return render_template('predict.html', text=text, sentiment=sentiment, probability=result[0], image=img_filename)


if __name__ == "__main__":
    app.run()
