# Setup
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates

import seaborn as sns
import matplotlib.pyplot as plt 
%matplotlib inline 

from sklearn.datasets import load_boston
load_boston = load_boston()
X = load_boston.data
y = load_boston.target

data = pd.DataFrame(X,columns = load_boston.feature_names)
data['SalePrice'] = y

data.head()


print(load_boston.DESCR)


sns.pairplot(data,height = 2.5)
plt.tight_layout()


sns.distplot(data['SalePrice']);


print('Skewness: %f' % data['SalePrice'].skew())
print('Kurtosis: %f' % data['SalePrice'].kurt())

fig ,ax = plt.subplots()
ax.scatter(x = data['CRIM'],y = data['SalePrice'])
plt.ylabel('SalePrice',fontsize = 13)
plt.xlabel('CRIM',fontsize = 13)
plt.show()


fig,ax = plt.subplots()
ax.scatter(x = data['AGE'],y = data['SalePrice'])
plt.ylabel('SalePrice',fontsize = 13)
plt.xlabel('CRIM',fontsize = 13)
plt.show()


from scipy import stats
from scipy.stats import norm,skew

sns.distplot(data['SalePrice'],fit = norm)

(mu,sigma) = norm.fit(data['SalePrice'])
print('\n mu ={:.2f} and sigma {:.2f} \n'.format(mu,sigma))

plt.legend(['Normal dist. ($\mu=${:.2f} and $\sigma=${:.2f} )'.format(mu,sigma)],
           loc = 'best'
           )
plt.ylabel("Frequency")
plt.xlabel('SalePrice distribution')


fig  = plt.figure()
res = stats.probplot(data['SalePrice'],plot = plt)
plt.show()



from sklearn.linear_model import LinearRegression
lineR = LinearRegression()


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 42,test_size = 0.2)


# linear regression fit
lineR.fit(X_train,y_train)
y_pred = lineR.predict(X_test)



print('Actual value of house : ',y_test[0])
print('Actual value of house : ',y_pred[0])


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
print(mse)