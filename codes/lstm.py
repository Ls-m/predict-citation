import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential, model_from_json
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from datetime import datetime
import random

name1 = "train1000-row-72390.csv"   #train file
n = 2000
years = -47

in_len = 7
out_len = 5

batch = 16
epoch = 100

name = 'lstm-'+str(in_len)+'-'+str(out_len)+'-'+str(epoch)+'-batch'+str(batch)+'-'+str(n)+'-row-'
r = random.randint(1,100000)
now = str(r)
name = name+now+'.h5'            #model name




def normal_data(scaler,values):
  feature_reshape = values.reshape(-1, 1)
  scaled_df = scaler.fit_transform(feature_reshape)
  # print(scaled_df)
  return scaled_df

def split_into_sequences(data, seq_len):
    n_seq = len(data) - seq_len + 1
    return np.array([data[i:(i+seq_len)] for i in range(n_seq)])


def my_split(data,n1,n2):
    n = n1+n2
    array1 = split_into_sequences(data, n)
    # print("array1 is ",array1)
    # print("array1[0:n1-1] is ",array1[0:n1-1])
    # print("array1[n1:n-1] is ",array1[n1:n-1])
    s1arr = []
    s2arr = []
    for i in range(len(array1)):
        split1 = array1[i]
        # print("split1 is ",split1)
        # print(split1[0,0])
        s1 = split1[0:n1]
        s2 = split1[n1:n]
        # print("s1 is ", s1)
        # print("s2 is ", s2)
        s1arr.append(s1)
        s2arr.append(s2)

    return s1arr, s2arr

    # return array1[0:n1-1],array1[n1:n-1]



train = pd.read_csv(name1)

train = train.iloc[: , years:]

print("*****************Train************************")
print(train)




# exit()




trainX = []
trainY = []

for i in range(len(train)):
  paper = train.iloc[i]
  paper2 = paper.to_numpy()
  x,y = my_split(paper2,in_len,out_len)
  for i in range(len(x)):
    trainX.append(x[i])

  for i in range(len(y)):
    trainY.append(y[i])

trainX = np.asarray(trainX)
trainY = np.asarray(trainY)

print("*** trainX ****")
print(trainX)

print("**** trainY ***")
print(trainY)

print("*******")


print("trainX shape is ",trainX.shape)
print("trainY shape is ",trainY.shape)

# exit()

trainX = trainX.reshape(len(trainX),in_len,1)
trainY = trainY.reshape(len(trainY),out_len,1)



from keras.layers import RepeatVector
from keras.layers import TimeDistributed

model = Sequential()

# encoder layer
model.add(LSTM(100, activation='relu', input_shape=(in_len, 1)))

# repeat vector
model.add(RepeatVector(out_len)) #output shape

# decoder layer
model.add(LSTM(100, activation='relu', return_sequences=True))

model.add(TimeDistributed(Dense(1, activation='relu')))
model.compile(optimizer='adam', loss='mse')

print(model.summary())


history = model.fit(trainX, trainY, epochs=epoch, validation_split=0.2, verbose=1, batch_size=batch)

model.save(name)
