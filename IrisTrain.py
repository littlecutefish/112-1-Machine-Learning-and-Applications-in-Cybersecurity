# 參考: https://medium.com/@vickygamingoff2004/machine-learning-iris-dataset-3-algorithms-2c55207f54f9

import pandas as pd
from sklearn import neighbors
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# %% Loading Iris Data into a DataFrame
iris_data = pd.read_csv("Iris.csv")

# %% Filtering the Data
iris_data = iris_data.drop('Id', axis=1)

# %% Splitting the Train and Test Data
x = iris_data.drop('Species', axis=1)
y = iris_data['Species']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)

# %% Binarize the output labels for AUC calculation
y_train_bin = label_binarize(y_train, classes=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
y_test_bin = label_binarize(y_test, classes=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

# %% Logistic Regression
logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)
logistic_predictions = logistic_regression.predict(x_test)
logistic_predictions_proba = logistic_regression.predict_proba(x_test)

print('======== Logistic Regression ========')
print('Accuracy: %.2f' % (accuracy_score(y_test, logistic_predictions) * 100))
print('Predictions: %.2f' % (precision_score(y_test, logistic_predictions, average='macro') * 100))
print('Recall: %.2f' % (recall_score(y_test, logistic_predictions, average='macro') * 100))
print('AUC: %.2f' % (roc_auc_score(y_test_bin, logistic_predictions_proba, multi_class='ovr') * 100))

# %% KNeighborsClassifier
k_classifier = neighbors.KNeighborsClassifier()
k_classifier.fit(x_train, y_train)
k_neighbors_predictions = k_classifier.predict(x_test)
k_neighbors_predictions_proba = k_classifier.predict_proba(x_test)

print('\n======== KNeighborsClassifier ========')
print('Accuracy: %.2f' % (accuracy_score(y_test, k_neighbors_predictions) * 100))
print('Predictions: %.2f' % (precision_score(y_test, k_neighbors_predictions, average='macro') * 100))
print('Recall: %.2f' % (recall_score(y_test, k_neighbors_predictions, average='macro') * 100))
print('AUC: %.2f' % (roc_auc_score(y_test_bin, k_neighbors_predictions_proba, multi_class='ovr') * 100))

# %% DecisionTreeClassifier
decision_classifier = tree.DecisionTreeClassifier()
decision_classifier.fit(x_train, y_train)
decision_classifier_predictions = decision_classifier.predict(x_test)
decision_classifier_predictions_proba = decision_classifier.predict_proba(x_test)

print('\n======== DecisionTreeClassifier ========')
print('Accuracy: %.2f' % (accuracy_score(y_test, decision_classifier_predictions) * 100))
print('Predictions: %.2f' % (precision_score(y_test, decision_classifier_predictions, average='macro') * 100))
print('Recall: %.2f' % (recall_score(y_test, decision_classifier_predictions, average='macro') * 100))
print('AUC: %.2f' % (roc_auc_score(y_test_bin, decision_classifier_predictions_proba, multi_class='ovr') * 100))

# # %% Creating a Comparison DataFrame with The predictions with all 3 Algorithms and its accuracy
# result_data = [logistic_predictions, k_neighbors_predictions, decision_classifier_predictions]
# result_predictions = [(accuracy_score(y_test, logistic_predictions) * 100),
#  (accuracy_score(y_test, k_neighbors_predictions) * 100),
#  (accuracy_score(y_test, decision_classifier_predictions) * 100)]
# data_frame = pd.DataFrame([result_data, result_predictions],
#  columns=['Logistic Regression', 'KNeighborsClassification', 'DecisionTreeClassification'])
# data_frame = data_frame.T
# data_frame.columns = ['Predictions', 'Accuracy']
