import pandas as pd
import numpy as np


file = pd.read_csv('diabetes.csv')

file[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = file[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

file['Glucose'] = file['Glucose'].fillna(file['Glucose'].mean())
file['BloodPressure'] = file['BloodPressure'].fillna(file['BloodPressure'].mean())
file['SkinThickness'] = file['SkinThickness'].fillna(file['SkinThickness'].mean())
file['Insulin'] = file['Insulin'].fillna(file['Insulin'].mean())
file['BMI'] = file['BMI'].fillna(file['BMI'].mean())
file['DiabetesPedigreeFunction'] = file['DiabetesPedigreeFunction'].fillna(file['DiabetesPedigreeFunction'].mean())


mean_array = [file['Glucose'].mean(),file['BloodPressure'].mean(),file['SkinThickness'].mean(),file['Insulin'].mean(),file['BMI'].mean(),file['DiabetesPedigreeFunction'].mean()]
mean_array = np.array(mean_array)
print(mean_array)
np.savetxt("mean_values.txt", mean_array, delimiter =", ")


# with open('mean_values.txt','w') as writer:
#     writer.write(str(mean_array))


x = file.iloc[:,0:8].values
y = file.iloc[:,-1].values
#print(x[0])
#print(y[0])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 0)
# print( x_train[0])

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

#print(x_train[0])
# print( type(x_train))

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

import pickle

with open("main-model","wb") as file:
    pickle.dump(clf, file)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
# print(accuracy)
# print(y_pred[0])
# print(y_test[0])

array = np.array([[15,136,70,32,110,37.1,0.153,43]])
print(array)
# array = sc.transform(array)
print(array)

test_pred = clf.predict(array)
print(test_pred)






# [[  0.    101.     65.     28.      0.     24.6     0.237  22.   ]]
# [[   0.    -101.     -65.     -28.       0.     -24.6     -0.237  -22.   ]]
# [1]