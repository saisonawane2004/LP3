#Experiment 3.Given a bank customer, build a neural network-based classifier that can determine whether they 
#will leave or not in the next 6 months.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
df = pd.read_csv(r'D:\Ml_Code\ML\ML\Practical B3\Churn_Modelling.csv')
df.info()

df.head()

df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

df.isna().sum()

df.describe()

X=df.iloc[:, :df.shape[1]-1].values
y=df.iloc[:, -1].values
X.shape, y.shape

print(X[:8,1], '...will now become: ')

label_X_country_encoder = LabelEncoder()
X[:,1] = label_X_country_encoder.fit_transform(X[:,1])
print(X[:8,1])

print(X[:6,2], '...will now become: ')

label_X_gender_encoder = LabelEncoder()
X[:,2] = label_X_gender_encoder.fit_transform(X[:,2])
print(X[:6,2])

transform = ColumnTransformer([("countries", OneHotEncoder(), [1])], remainder="passthrough")
X = transform.fit_transform(X)
X

X = X[:,1:]
X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train[:,np.array([2,4,5,6,7,10])] = sc.fit_transform(X_train[:,np.array([2,4,5,6,7,10])])
X_test[:,np.array([2,4,5,6,7,10])] = sc.transform(X_test[:, np.array([2,4,5,6,7,10])])

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train



#initializing the ANN
classifier = Sequential()

from tensorflow.keras.layers import Dense

classifier.add(Dense(activation = 'relu', input_dim = 11, units=256, kernel_initializer='uniform'))

#adding hidden layer

classifier.add(Dense(activation = 'relu', units=512, kernel_initializer= 'uniform'))
classifier.add(Dense(activation = 'relu', units=256, kernel_initializer= 'uniform'))
classifier.add(Dense(activation = 'relu', units=128, kernel_initializer= 'uniform'))

classifier.add(Dense(activation = 'sigmoid', units=1, kernel_initializer = 'uniform'))

#create optimizer with default learning rate


classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.summary()

classifier.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    epochs=20,
    batch_size=32
)

y_pred = classifier.predict(X_test)
y_pred

y_pred = (y_pred > 0.5)
y_pred

from sklearn.metrics import confusion_matrix, classification_report

cm1 = confusion_matrix(y_test, y_pred)
cm1

print(classification_report(y_test, y_pred))

accuracy_model1 = ((cm1[0][0] + cm1[1][1]) *100)/(cm1[0][0]+cm1[1][1] + cm1[0][1] + cm1[1][0])
print(accuracy_model1, '% of testing data was classified correctly')

