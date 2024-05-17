import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
#Loading the Dataset in Pandas DataFrame
sonar=pd.read_csv('Sonar data.csv',header=None)
sonar.head()