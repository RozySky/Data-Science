import tensorflow
from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sklearn

data = pd.read_csv('model/student-mat.csv', sep=';')
data = data[['G1', 'G2', 'G3', 'absences', 'failures']]
np_data = np.array(data.drop(['G3'], 1))
np_predict = np.array(data['G3'])



x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(np_data, np_predict, test_size=0.1)



load_model = open('model/acc.pickle', 'rb')
mod = pickle.load(load_model)

machine_predictions = mod.predict(x_test)
for i in range(len(machine_predictions)):
    print(machine_predictions[i], ' True Value:' + str(y_test[i]))

from matplotlib import style
style.use('ggplot')

plt.scatter(data['G2'],data['G3'])
plt.show()

