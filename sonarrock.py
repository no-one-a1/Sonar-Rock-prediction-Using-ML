import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
sonar_data = pd.read_csv(
    "C:/Users/DELL/OneDrive/Desktop/pyprojects/mlprojects/Copy of sonar data.csv", header=None)
sonar_data.head()
# no of rows and columns
sonar_data.shape
# statistical values of data
sonar_data.describe()
# no of rocks and mines in dataset
sonar_data[60].value_counts()
# mean value of both rocks and mines
sonar_data.groupby(60).mean()
# separating data and labels
x = sonar_data.drop(columns=60, axis=1)
y = sonar_data[60]
print(x)
print(y)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, stratify=y, random_state=1)
print(x.shape, x_train.shape, x_test.shape)
print(x_train)
print(y_train)
# model Training using logistic regression model
model = LogisticRegression()
# training the logistic regression model with training data
model.fit(x_train, y_train)
# -->model evaluation
# accuracy on training data
x_train_prediction = model.predict(x_train)
train_data_accuracy = accuracy_score(x_train_prediction, y_train)
# accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print("Accuracy on test data: ", test_data_accuracy)
# ----->making a predictive data
input_data = (0.1313, 0.2339, 0.3059, 0.4264, 0.4010, 0.1791, 0.1853, 0.0055, 0.1929, 0.2231, 0.2907, 0.2259, 0.3136, 0.3302, 0.3660, 0.3956, 0.4386, 0.4670, 0.5255, 0.3735, 0.2243, 0.1973, 0.4337, 0.6532, 0.5070, 0.2796, 0.4163, 0.5950, 0.5242,
              0.4178, 0.3714, 0.2375, 0.0863, 0.1437, 0.2896, 0.4577, 0.3725, 0.3372, 0.3803, 0.4181, 0.3603, 0.2711, 0.1653, 0.1951, 0.2811, 0.2246, 0.1921, 0.1500, 0.0665, 0.0193, 0.0156, 0.0362, 0.0210, 0.0154, 0.0180, 0.0013, 0.0106, 0.0127, 0.0178, 0.0231)
# changing innput data to numpy array
input_data_as_npy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one instance
input_data_reshape = input_data_as_npy_array.reshape(1, -1)
prediction = model.predict(input_data_reshape)
if prediction[0] == "R":
    print("The object is Rock")
else:
    print("The object is Mine")
