import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
#Loading the Dataset in Pandas DataFrame
sonar=pd.read_csv('Sonar data.csv',header=None)
#print(sonar.head())
#print(sonar.shape)
#print(sonar.describe())
#print(sonar.value_counts(60))
sonar.groupby(60).mean()
x=sonar.drop(columns=60,axis=1)
y=sonar[60]
#separating the training and test data
trainx, testx, trainy, testy=train_test_split(x,y,test_size=0.1,stratify=y, random_state=1)
#print(trainx.shape)
#print(testx.shape)
model=LogisticRegression()
#training the model
model.fit(trainx,trainy)
#Finding Accuracy on training data
predictiontrain=model.predict(trainx)
trainingaccuracy=accuracy_score(predictiontrain,trainy)
#print(trainingaccuracy)
#Finding Accuracy on test data
predictiontest=model.predict(testx)
#print(predictiontest)
testaccuracy=accuracy_score(predictiontest,testy)
#print(testaccuracy)
input_data=(0.0317,0.0956,0.1321,0.1408,0.1674,0.1710,0.0731,0.1401,0.2083,0.3513,0.1786,0.0658,0.0513,0.3752,0.5419,0.5440,0.5150,0.4262,0.2024,0.4233,0.7723,0.9735,0.9390,0.5559,0.5268,0.6826,0.5713,0.5429,0.2177,0.2149,0.5811,0.6323,0.2965,0.1873,0.2969,0.5163,0.6153,0.4283,0.5479,0.6133,0.5017,0.2377,0.1957,0.1749,0.1304,0.0597,0.1124,0.1047,0.0507,0.0159,0.0195,0.0201,0.0248,0.0131,0.0070,0.0138,0.0092,0.0143,0.0036,0.0103)

#changing the input_data to numpy array
arr=np.asarray(input_data)

#reshaping the np array
reshapearr=arr.reshape(1,-1)

prediction=model.predict(reshapearr)

if(prediction[0]=='R'):
  print("It is a Rock")
elif(prediction[0]=='M'):
  print("It is a Mine")