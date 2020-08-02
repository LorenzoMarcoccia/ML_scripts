# see iris dataset documentation https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dataset = load_iris()
X = dataset['data']
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5)

model = DecisionTreeClassifier()
model.fit(X_train, y_train) #training

#prediction
p_train = model.predict(X_train) 
p_test  = model.predict(X_test) 

#validation
acc_train = accuracy_score(y_train, p_train) 
acc_test = accuracy_score(y_test, p_test)  

print( f'Train {acc_train}, test {acc_test}')
