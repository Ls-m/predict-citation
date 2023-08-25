import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from datetime import datetime
import random


name1='train100-row-70114.csv'  #train file
name2='test100-row-70114.csv'  #test file
n = 100
years = -47


in_len = 7
out_len = 5





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

test = pd.read_csv(name2)

test = test.iloc[: , years:]

print("*****************Test************************")
print(test)





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




limit = test.to_numpy().max()


testX = []
testY = []


for i in range(len(test)):
  paper = test.iloc[i]
  paper2 = paper.to_numpy()

  x,y = my_split(paper2,in_len,out_len)
  for i in range(len(x)):
    testX.append(x[i])

  for i in range(len(y)):
    testY.append(y[i])

testX = np.asarray(testX)
testY = np.asarray(testY)

print('*************** testX **********************')
print(testX)
print('****************** testY *******************')
print(testY)



print('****************8980*********************')


tshape = testY.shape
print(testY.shape)

avgY = np.zeros(shape=tshape)
print(avgY)
print(avgY.shape)

lastY = np.zeros(shape=tshape)
randY = np.zeros(shape=tshape)

for i in range(len(testX)):
    # print(testX[i])
    # print(testX[i][-3:])
    avg = sum(testX[i][-3:])/3
    # print(avg)

    kk = np.zeros(shape=[1,tshape[1]]) 
    for j in range(tshape[1]):
        kk[0,j] = avg
    # print (kk)
    avgY[i] = kk
    # print(avgY)

    mm = np.zeros(shape=[1,tshape[1]]) 
    for j in range(tshape[1]):
        mm[0,j] = testX[i][-1:]
    # print(mm)
    lastY[i] = mm
    # print(lastY)

    ff = np.zeros(shape=[1,tshape[1]])
    for j in range(tshape[1]):
        ff[0,j] = random.randint(0,limit)
    # print("ff is ",ff)
    randY[i] = ff
    

   

# exit()
# load model 

model = linear_model.LinearRegression().fit(trainX, trainY)
test_output = model.predict(testX)
print(test_output.shape)
print(testY.shape)
test_output = test_output.reshape(720,out_len)
print(test_output.shape)
print(test_output)

print('*************9009990990909009990************************')
r0 = r2_score(testY, test_output)
print("r2 for lr: ",r0)
m0 = mean_squared_error(testY, test_output, squared=False)
print("rmse for lr: ",m0)


print('*************9009990990909009990************************')
r2 = r2_score(testY, avgY)
print("r2 for avg",r2)
m2 = mean_squared_error(testY, avgY, squared=False)
print("rmse for avg",m2)



print('*************9009990990909009990************************')
r3 = r2_score(testY, lastY)
print("r2 for last",r3)
m3 = mean_squared_error(testY, lastY, squared=False)
print("rmse for last",m3)


print('*************9009990990909009990************************')
a1 = r2_score(testY, randY)
print("r2 for rand",a1)
a2 = mean_squared_error(testY, randY, squared=False)
print("rmse for rand",a2)


 