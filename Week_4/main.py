# import pandas as pd
#
# # Read the CSV file
# df = pd.read_csv('data/loan_data.csv')
#
# # Display the first few rows of the DataFrame
# print("\n\nDataFrame:\n\n", df.head())
#
# # Display the summary statistics of the DataFrame
# print("\n\nSummary Statistics of the DataFrame:\n\n", df.describe())
#
#
# # Check for the missing values
# print("\n\nMissing values:\n\n", df.isnull().sum())
#
# # feature and target variable separation
# x = df[['age', 'income', 'loan_amount', 'credit_score']]
# y = df['default']
#
# # split the data
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#
# # import warning
# import warnings
# warnings.filterwarnings('ignore')
#
#
# from sklearn.linear_model import LogisticRegression
# # initialize the model
# model = LogisticRegression()
#
# # train the model
# model.fit(x_train, y_train)
# LogisticRegression()
#
# # make predictions
# y_pred = model.predict(x_test)
#
# from sklearn.metrics import accuracy_score
# accuracy_score = accuracy_score(y_test, y_pred)
#
# from sklearn.metrics import precision_score
# prediction = precision_score(y_test, y_pred)
#
# from sklearn.metrics import recall_score
# recall = recall_score(y_test, y_pred)
#
# from sklearn.metrics import confusion_matrix
# confusion = confusion_matrix(y_test, y_pred)
#
#
# print("\n\nAccuracy Score: ", accuracy_score)
# print("Precision Score: ", prediction)
# print("Recall Score: ", recall)
# print("Confusion Matrix: ", confusion)
#
# # example : new customer data
# new = [[30, 50000, 20000, 700]]
# prediction = model.predict(new)
# print("\n\nPrediction for new customer: ", prediction)

# -----------------------------------------------------------------------------------------------------------------------
# SVM Binary Classification

# Problem Statement

# The Iris dataset contains 150 samples from three species of
# Iris flowers setosa versicolor and virginica.
# Each sample includes four features: sepal length,
# sepal width, petal length, and petal width. For this example, we will
#
# 1. Consider only ten zanessa and 'venasto
# 2. Use only two features seallergy and sepal net


import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:100, :2]
y = iris.target[:100]

# Split the data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM classifier
from sklearn.svm import SVC

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

data = [[4.9, 3.0]]
prediction = clf.predict(data)
print(f"\nPrediction for sample data {data}: {iris.target_names[prediction][0]}")
import matplotlib.pyplot as plt


# Define a function to visualize the decision boundary
def plot_decision_boundary(clf, X, y):
    # Create a mesh to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('SVM Decision Boundary')
    plt.show()


# Plot the decision boundary
plot_decision_boundary(clf, X, y)
