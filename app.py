from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import librosa.display
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
from sklearn import metrics
import IPython.display as ipd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features
app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/rec',methods=["POST","GET"])
def rec():
    df=pd.read_csv(r'clean2.csv')
    X = np.array(df['features'].tolist())
    y = np.array(df['class'].tolist())
    labelencoder = LabelEncoder()
    y = to_categorical(labelencoder.fit_transform(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    num_labels = y.shape[1]
    if request.method=="POST":
        audio=request.files['file']
        print(type(audio))
        vectors=features_extractor(audio)
        print(vectors)
        vectors=pd.DataFrame([vectors])
        new_model=load_model("saved_models/audio_classification2.hdf5")
        result = new_model.predict([vectors])
        result=np.argmax([result], axis=-1)
        print(result)
        
        if result==0:
            msg = 'The Speaker identified is Arjun'
        elif result==1:
            msg='The Speaker identified is Benjamin Netanyau'
        elif result==2:
            msg = 'The Speaker identified is Jens Stoltenberg'
        elif result==3:
            msg = 'The Speaker identified is Jovin'
        elif result==4:
            msg=  'The Speaker identified is Julia Gillard'
        elif result==5:
            msg=  'The Speaker identified is Magaret Tarcher'
        elif result==6:
            msg = 'The Speaker identified is Nelson Mandela'
        else:
            msg = 'The Speaker identified is Soorya'
        return render_template('rec.html',msg=msg)
    return render_template('rec.html')


if __name__=="__main__":
    app.run(debug=True)