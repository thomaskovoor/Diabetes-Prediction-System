import pandas as pd
import numpy as np

file = pd.read_csv('D:\My Work\College\\3rd Year\S6\Miniproject\machine-learning-projects\Predicting Diabetes\diabetes.csv')

file[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = file[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

file['Glucose'] = file['Glucose'].fillna(file['Glucose'].mean())
file['BloodPressure'] = file['BloodPressure'].fillna(file['BloodPressure'].mean())
file['SkinThickness'] = file['SkinThickness'].fillna(file['SkinThickness'].mean())
file['Insulin'] = file['Insulin'].fillna(file['Insulin'].mean())
file['BMI'] = file['BMI'].fillna(file['BMI'].mean())

x = file.iloc[:,0:8].values
y = file.iloc[:,-1].values
#print(x[0])
#print(y[0])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 0)
# print( x_train[0])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#print(x_train[0])
# print( type(x_train))

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
print(accuracy)
print(y_pred[0])
print(y_test[0])

array = np.array([[0,101,65,28,0,24.6,0.237,22]])
print(array)
array = sc.transform(array)
print(array)

test_pred = clf.predict(array)
print(test_pred)


#
#Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
#test_data = ['Pregnancies' : 6,'Glucose' : 148, ''72,35,0,33.6,0.627,50,1]
#test_data['Glucose'] = test_data['Glucose'].fillna(test_data['Glucose'].mean())
#test_data['BloodPressure'] = test_data['BloodPressure'].fillna(test_data['BloodPressure'].mean())
#test_data['SkinThickness'] = test_data['SkinThickness'].fillna(test_data['SkinThickness'].mean())
#test_data['Insulin'] = test_data['Insulin'].fillna(test_data['Insulin'].mean())
#test_data['BMI'] = test_data['BMI'].fillna(test_data['BMI'].mean())
#