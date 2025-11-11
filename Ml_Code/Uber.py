#Experiment 1.  Predict the price of the Uber ride from a given pickup point to the agreed drop-off location.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pylab
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import preprocessing

"""Loading Dataset"""

df = pd.read_csv(r'D:\Ml_Code\ML\ML\Practical B1\uber.csv')
df.info()

df.head()

df.describe()

"""Cleaning"""

df = df.drop(['Unnamed: 0','key'],axis=1)
df.isna().sum()

"""Remove All Null Values"""

df.dropna(axis=0,inplace=True)
df.dtypes

df.pickup_datetime = pd.to_datetime(df.pickup_datetime,errors='coerce')

df = df.assign(
    second = df.pickup_datetime.dt.second,
    minute = df.pickup_datetime.dt.minute,
    hour = df.pickup_datetime.dt.hour,
    day = df.pickup_datetime.dt.day,
    month = df.pickup_datetime.dt.month,
    year = df.pickup_datetime.dt.year,
    dayofweek = df.pickup_datetime.dt.dayofweek
)
df = df.drop('pickup_datetime',axis=1)
df.info()

df.head()

incorrect_coordinates = df.loc[
    (df.pickup_latitude > 90) | (df.pickup_latitude < -90) |
    (df.dropoff_latitude > 90) | (df.dropoff_latitude < -90) |
    (df.pickup_longitude > 180) | (df.pickup_longitude < -180) |
    (df.dropoff_longitude > 90) | (df.dropoff_longitude < -90)
]

df.drop(incorrect_coordinates,inplace=True,errors='ignore')

def distance_transform(longitude1,latitude1,longitude2,latitude2):
  long1,lati1,long2,lati2 = map(np.radians,[longitude1,latitude1,longitude2,latitude2])
  dist_long = long2 - long1
  dist_lati = lati2 - lati1
  a = np.sin(dist_lati/2)**2 + np.cos(lati1) * np.cos(lati2) * np.sin(dist_long/2)**2
  c = 2 * np.arcsin(np.sqrt(a)) * 6371
  # long1 ,lati1,long2,lati2 = longitude1[pos], latitude1[pos],longitude2[pos],latitude2[pos]
  # c = sqrt((long2 - long1) ** 2 + (lati2 - lati1) ** 2)asin

  return c

df['Distance'] = distance_transform(
    df['pickup_longitude'],
    df['pickup_latitude'],
    df['dropoff_longitude'],
    df['dropoff_latitude']
)

df.head()

plt.scatter(df['Distance'], df['fare_amount'])
plt.xlabel("Distance")
plt.ylabel('fare_amount')

plt.figure(figsize=(20,12))
sns.boxplot(data=df)

df.drop(df[df['Distance'] >= 60].index , inplace=True)
df.drop(df[df['fare_amount'] <= 0].index , inplace=True)

df.drop(df[(df['fare_amount']>100) & (df['Distance']<1)].index, inplace=True)
df.drop(df[(df['fare_amount']<100) & (df['Distance']>100)].index, inplace=True)

plt.scatter(df['Distance'], df['fare_amount'])
plt.xlabel("Distance")
plt.ylabel('fare_amount')

corr = df.corr()

corr.style.background_gradient(cmap='BuGn')

x = df['Distance'].values.reshape(-1,1)  #Independent Variable
y = df['fare_amount'].values.reshape(-1,1)  #Dependent Variable

from sklearn.preprocessing import  StandardScaler
std = StandardScaler()
y_std = std.fit_transform(y)
print(y_std)

x_std = std.fit_transform(x)
print(x_std)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_std,y_std,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
l_reg = LinearRegression()
l_reg.fit(x_train,y_train)

print("Training set score: {:.2f}".format(l_reg.score(x_train,y_train)))
print("Test set score: {:.7f}".format(l_reg.score(x_test,y_test)))

y_pred = l_reg.predict(x_test)

result = pd.DataFrame()
result[['Actual']] = y_test
result[['predicted']] = y_pred

result.sample(10)

print('Mean Absolute Error : ', metrics.mean_absolute_error(y_test,y_pred))
print('Mean Absolute % Error : ', metrics.mean_absolute_percentage_error(y_test,y_pred))
print('Mean Squared Error : ', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error : ', np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))
print('Mean Absolute Error : ', np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))

plt.subplot(2,2,1)
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,l_reg.predict(x_train),color='blue')
plt.title("Fare vs Distance (Training Set)")

plt.subplot(2,2,2)
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,l_reg.predict(x_train),color='blue')
plt.ylabel("fare_amount")
plt.xlabel("Distance")
plt.title("Fare vs Distance (Test Set)")

plt.tight_layout()
plt.show()

cols = ['Model', 'RMSE', 'R-Squared']

#create a empty dataframe of the columns
#columns: specifies the columns to be selected
result_tabulation = pd.DataFrame(columns=cols)

#compile the required information
linreg_metrics = pd.DataFrame([[
    'Linear Regression Model',
    np.sqrt(metrics.mean_squared_error(y_test,y_pred)),
    metrics.r2_score(y_test,y_pred)
]], columns = cols)

result_tabulation = pd.concat([result_tabulation,linreg_metrics], ignore_index = True)

result_tabulation

"""**RandomForestRegressor**"""

rf_reg = RandomForestRegressor(n_estimators=100, random_state=10)

#fit the regressor with training dataset
rf_reg.fit(x_train,y_train)

#predictr the value on test dataset using predict()
y_pred_RF = rf_reg.predict(x_test)

result = pd.DataFrame()
result[['Actual']] = y_test
result[['predicted']] = y_pred_RF.reshape(-1,1)

result.sample(10)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_RF))
print('Mean Absolute % Error:', metrics.mean_absolute_percentage_error(y_test,y_pred_RF))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_RF))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_RF)))
print('R Squared (R^2):', np.sqrt(metrics.r2_score(y_test, y_pred_RF)))

"""**Visualization**"""

#Build Scaterplot
plt.scatter(x_test,y_test,c = 'b', alpha = 0.5, marker = '.', label = 'Real')
plt.scatter(x_test,y_pred_RF, c='r', alpha = 0.5, marker = '.', label = 'Predicted')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.grid(color = '#D3D3D3', linestyle = 'solid')
plt.legend(loc = 'lower right')

plt.tight_layout()
plt.show()

#compile the required information
random_forest_metrics = pd.DataFrame([[
    "Random Forest Regressor Model",
    np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF)),
    np.sqrt(metrics.r2_score(y_test,y_pred_RF))
]], columns = cols)

result_tabulation = pd.concat([result_tabulation, random_forest_metrics], ignore_index = True)

result_tabulation

!jupyter nbconvert --to webpdf "/content/LP3_ML(B1).ipynb" --WebPDFExporter.allow_chromium_download=True