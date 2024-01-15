import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import librosa.display
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
#list the files

filelist = os.listdir(r'\16000_pcm_speeches')
#read them into pandas
train_df = pd.DataFrame(filelist)
train_df = train_df.rename(columns={0:'file'})
path = r"D:\16000_pcm_speeches"
p = os.listdir(path)
l = []
s = []
for x in p:
  p1 = os.path.join(path, x)
  files = os.listdir(p1)
  for i in files:
    l.append(i)
    s.append(x)
data = {'file':l,'speaker':s}
df = pd.DataFrame(data)
# filename='C:/Users/ymts0359/Downloads/archive (8)/16000_pcm_speeches/Benjamin_Netanyau/1.wav'
# plt.figure(figsize=(14,5))
# data,sample_rate=librosa.load(filename)
# librosa.display.waveplot(data,sr=sample_rate)
# ipd.Audio(filename)
audio_dataset_path=r'dataset\16000_pcm_speeches\Benjamin_Netanyau'
def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features

# import numpy as np
# from tqdm import tqdm
# ### Now we iterate through every audio file and extract features
# ### using Mel-Frequency Cepstral Coefficients
# extracted_features=[]
# for index_num,row in tqdm(metadata.iterrows()):
#     file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
#     final_class_labels=row["class"]
#     data=features_extractor(file_name)
#     extracted_features.append([data,final_class_labels])

import numpy as np
from tqdm import tqdm
### Now we iterate through every audio file and extract features
### using Mel-Frequency Cepstral Coefficients
extracted_features=[]
# for index_num,row in tqdm(df.iterrows()):
#     file_name = audio_dataset_path
#     print(file_name)
#     final_class_labels=row["speaker"]
#     print(file_name)
dir_name = '/Benjamin_Netanyau'
for audio in glob.glob(dir_name+'/*.wav'):
  # print(audio)
  data=features_extractor(audio)
  print(data)
  extracted_features.append([data,'Benjamin_Netanyau'])

print(extracted_features)
bn=pd.DataFrame(extracted_features)
print(bn)

extracted_features2=[]
dir_name = '/Jens_Stoltenberg'
for audio in glob.glob(dir_name+'/*.wav'):
  # print(audio)
  data=features_extractor(audio)
  print(data)
  extracted_features2.append([data,'Jens_Stoltenberg'])

print(extracted_features)
js=pd.DataFrame(extracted_features2)
print(js)

extracted_features3=[]
dir_name = '/16000_pcm_speeches/Julia_Gillard'
for audio in glob.glob(dir_name+'/*.wav'):
  # print(audio)
  data=features_extractor(audio)
  print(data)
  extracted_features3.append([data,'Julia_Gillard'])

print(extracted_features3)
jg=pd.DataFrame(extracted_features3)
print(jg)

extracted_features4=[]
dir_name = '/Magaret_Tarcher'
for audio in glob.glob(dir_name+'/*.wav'):
  # print(audio)
  data=features_extractor(audio)
  print(data)
  extracted_features4.append([data,'Magaret_Tarcher'])

print(extracted_features4)
mt=pd.DataFrame(extracted_features4)
print(mt)

extracted_features5=[]
dir_name = '/16000_pcm_speeches/Nelson_Mandela'
for audio in glob.glob(dir_name+'/*.wav'):
  # print(audio)
  data=features_extractor(audio)
  print(data)
  extracted_features5.append([data,'Nelson_Mandela'])

print(extracted_features5)
nm=pd.DataFrame(extracted_features5)
print(nm)

extracted_features6=[]
dir_name = '/16000_pcm_speeches/Arjun'
for audio in glob.glob(dir_name+'/*.wav'):
  # print(audio)
  data=features_extractor(audio)
  print(data)
  extracted_features6.append([data,'Arjun'])

print(extracted_features6)
pq=pd.DataFrame(extracted_features6)
print(pq)

extracted_features7=[]
dir_name = r'dataset\16000_pcm_speeches\Jovin'
for audio in glob.glob(dir_name+'/*.wav'):
  # print(audio)
  data=features_extractor(audio)
  print(data)
  extracted_features7.append([data,'Jovin'])

print(extracted_features7)
rs=pd.DataFrame(extracted_features7)
print(rs)

extracted_features8=[]
dir_name = '/16000_pcm_speeches/Soorya'
for audio in glob.glob(dir_name+'/*.wav'):
  # print(audio)
  data=features_extractor(audio)
  print(data)
  extracted_features8.append([data,'Soorya'])

print(extracted_features8)
tu=pd.DataFrame(extracted_features8)
print(tu)


full = pd.concat([bn, js, jg, mt, nm, pq, rs, tu], axis=0)
full.columns=['features','class']
full.to_csv('clean2.csv')
print(full)
X=np.array(full['features'].tolist())
y=np.array(full['class'].tolist())
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
num_labels=y.shape[1]
model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification2.hdf5',
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)


test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])

# new=X_test[0]
# abc=X_test
# result=model.predict([new])
# np.argmax([result])