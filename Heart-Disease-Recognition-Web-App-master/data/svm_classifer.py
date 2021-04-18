

import pandas as pd
from sklearn import model_selection
from sklearn import svm
import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
from sklearn.externals import joblib

df = pd.read_csv('heart_tidy.csv', names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target'])
features = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']


X = df.loc[1:,features].values
Y = df.loc[1:,['target']].values

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size = 0.3, random_state = 0)

clfr = svm.SVC(kernel = 'linear',C = 0.1).fit(X_train, Y_train)
joblib.dump(clfr, 'Heart_model.pkl')

acc = clfr.score(X_test, Y_test)
print("Accuracy: ",acc*100," %.")