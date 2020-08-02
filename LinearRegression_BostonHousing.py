# see the dataset  https://www.kaggle.com/prasadperera/the-boston-housing-dataset

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import numpy as np
np.random.seed(2)

dataset = load_boston()
#print(dataset['data'])

X = dataset['data']
y = dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X,y)

model = LinearRegression()
model.fit(X_train, y_train) #training

p_train = model.predict(X_train) #prediction
p_test  = model.predict(X_test)  #prediction

#validation
mae_train = mean_absolute_error(y_train, p_train)
mae_test  = mean_absolute_error(y_test,  p_test)

print('MAE train', mae_train)
print('mean y train', np.mean(y_train))

print('MAE test', mae_test)
print('mean y test', np.mean(y_test))
