import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

data = pd.read_csv('model/breast-cancer.data')

modl = preprocessing.LabelEncoder()

class_f = modl.fit_transform(list(data['class']))
age = modl.fit_transform(list(data['age']))
menopause = modl.fit_transform(list(data['menopause']))
tumor_size = modl.fit_transform(list(data['tumor-size']))
inv_nodes = modl.fit_transform(list(data['inv-nodes']))
node_caps = modl.fit_transform(list(data['node-caps']))
deg_f = modl.fit_transform(list(data['dag-malig']))
breast = modl.fit_transform(list(data['breast']))
breast_quad = modl.fit_transform(list(data['breast-quad']))
irradiat = modl.fit_transform(list(data['irradiat']))

X_Axis = list(zip(age,menopause, tumor_size, inv_nodes, node_caps, deg_f, breast, breast_quad, irradiat))
Y_Axis = list(class_f)

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X_Axis, Y_Axis, test_size=0.1)

KN = KNeighborsClassifier(n_neighbors=11)
KN.fit(x_train, y_train)
acc = KN.score(x_test, y_test)
print(acc)

SVM = svm.SVC(kernel='linear')
SVM.fit(x_train, y_train)
predictions = SVM.predict(x_test)
acc = sklearn.metrics.accuracy_score(predictions, y_test)
hunter = ['no-recurrence-events', 'recurrence-events']
for i in range(len(predictions)):
    print(hunter[predictions[i]], ' Actual: ', hunter[y_test[i]])

plt.scatter(data['age'], data['class'])
plt.show()