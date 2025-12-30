import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

diabetes_dataset = pd.read_csv(r'E:\TMSL\coding\pythonpro\diabetes-1.csv')

 
diabetes_dataset.head()

diabetes_dataset.shape

diabetes_dataset.describe()

plot = sns.catplot( x = 'Pregnancies', y = 'Age', data = diabetes_dataset, kind ='box',)
plt.show()

diabetes_dataset['Outcome'].value_counts()

diabetes_dataset.groupby('Outcome').mean()

X = diabetes_dataset.drop(columns='Outcome', axis = 1)
Y = diabetes_dataset['Outcome']


print(X)

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, stratify=Y, random_state=2)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# print(Standardized_data)

print(X)
print(Y)

print(X.shape, X_test.shape, X_train.shape)

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, Y_train)


X_train_prediction = classifier.predict(X_train)

training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data', test_data_accuracy)
input_data= ( 1,189,60,23,846,30.1,0.398,59)

input_as_numpy = np.asarray(input_data)

input_data_reshaped = input_as_numpy.reshape(1,-1)

standard_data2 = scaler.transform(input_data_reshaped)

print(standard_data2)

prediction = classifier.predict(standard_data2)
print(prediction)

if(prediction[0] == 0):
  print('Person is non Diabetic')
else:
  print('Person is Diabetic')

pickle.dump(classifier, open('diabetes_model.sav', 'wb'))
pickle.dump(scaler, open('scaler.sav', 'wb'))
print("Model and scaler saved successfully.")
