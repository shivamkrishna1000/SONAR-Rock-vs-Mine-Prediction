import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
#Loading the Dataset in Pandas DataFrame
sonar=pd.read_csv('Sonar data.csv',header=None)
sonar.head()
sonar.shape
sonar.describe()
sonar.value_counts(60)
sonar.groupby(60).mean()
x=sonar.drop(columns=60,axis=1)
y=sonar[60]
#separating the training and test data
trainx, testx, trainy, testy=train_test_split(x,y,test_size=0.1,stratify=y, random_state=1)
trainx.shape
testx.shape
model=LogisticRegression()
#training the model
model.fit(trainx,trainy)
#Finding Accuracy on training data
predictiontrain=model.predict(trainx)
trainingaccuracy=accuracy_score(predictiontrain,trainy)
