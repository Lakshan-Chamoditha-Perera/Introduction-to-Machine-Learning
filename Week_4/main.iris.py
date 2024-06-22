from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Prediction
data = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(data)

predicted_species = iris.target_names[prediction][0]
print(f"Prediction for data {data}: {predicted_species}")
