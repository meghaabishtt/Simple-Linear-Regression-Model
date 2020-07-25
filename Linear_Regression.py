#import libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline

#load data
advert = pd.read_csv('Advertising.csv')
advert.head()
advert.info()

advert.columns
advert.drop(['Unnamed: 0'], axis = 1, inplace = True)
advert.head()

#Exploratory analysis
import seaborn as sns
sns.distplot(advert.sales);
sns.distplot(advert.newspaper);
sns.distplot(advert.radio);
sns.distplot(advert.TV);

sns.pairplot(advert, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', height=7, aspect=0.7, kind='reg');
advert.TV.corr(advert.sales)
advert.corr()
sns.heatmap( advert.corr(), annot=True );

#Creating a simple linear regression model
X = advert[['TV']]
X.head()
y = advert.sales
y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)


#Interpreting model coefficients
print(linreg.intercept_)
print(linreg.coef_)
zip(advert.TV, linreg.coef_)

#make predictions
y_pred = linreg.predict(X_test)

#calculate Mean Absolute Error 
from sklearn import metrics
print(metrics.mean_absolute_error(true, pred))

#Mean Squared Error
print(metrics.mean_squared_error(true, pred))

# calculate RMSE by hand
print(np.sqrt(((10**2 + 0**2 + 20**2 + 10**2) / 4)))

# calculate RMSE using scikit-learn
print(np.sqrt(metrics.mean_squared_error(true, pred)))

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))