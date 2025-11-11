#Experiment 2. Classify the email using the binary classification method. 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import preprocessing

df = pd.read_csv(r'D:\Ml_Code\ML\ML\Practical B2\emails.csv')

df.info()

df.head()

df.dtypes

df.drop(columns=['Email No.'], inplace=True)

df.isna().sum()

df.describe()

df = df.fillna(0)

X=df.iloc[:, :df.shape[1]-1]
#Independent Variables
y=df.iloc[:, -1]
#Dependent Variable
X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=8)

models = {
"K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=2),
"Linear SVM":LinearSVC(random_state=8, max_iter=900000),
"Polynomical SVM":SVC(kernel="poly", degree=2, random_state=8),
"RBF SVM":SVC(kernel="rbf", random_state=8),
"Sigmoid SVM":SVC(kernel="sigmoid", random_state=8)
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy for {model_name} model : {acc:.4f}")