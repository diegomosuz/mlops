import numpy as np
import pandas as pd
import sklearn
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

data = pd.read_csv('storepurchasedata.csv')

data.describe()

X = data.iloc[:, :-1].values
y = data.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

y_pred_prob = classifier.predict_proba(X_test)[:,1]

cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(y_test, y_pred)

new_pred = classifier.predict(sc.transform(np.array([[40, 20000]])))
new_pred_proba = classifier.predict_proba(sc.transform(np.array([[40, 20000]])))[:,1]

new_pred = classifier.predict(sc.transform(np.array([[42, 50000]])))
new_pred_proba = classifier.predict_proba(sc.transform(np.array([[42, 50000]])))[:,1]

model_file = 'classifier.pickle'
pickle.dump(classifier, open(model_file, 'wb'))

scaler_file = 'scaler.pickle'
pickle.dump(sc, open(scaler_file, 'wb'))




