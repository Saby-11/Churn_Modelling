# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data -- independent variable as it also contains the strings
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#dummy vari create krta h
onehotencoder = ColumnTransformer( 
	[('onehotencoder',OneHotEncoder(categories ='auto'), [1])],remainder = 'passthrough')
X = onehotencoder.fit_transform(X)
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Lets make ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialising ANN
classifier = Sequential()

#adding input and hidden layer
classifier.add(Dense(6, kernel_initializer ='uniform', activation='relu', input_shape=(11,)))
#adding 2 hidden layer
classifier.add(Dense(6, kernel_initializer ='uniform', activation='relu'))
#adding output layer
classifier.add(Dense(1, kernel_initializer ='uniform', activation='sigmoid'))
#compiling / applying schotastic gradient descent
classifier.compile(optimizer= "adam", loss= "binary_crossentropy", metrics= ['accuracy'])

#fiiting the ANN to training set
classifier.fit(X_train, y_train, batch_size= 10,epochs= 100 )

#prediting on test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)