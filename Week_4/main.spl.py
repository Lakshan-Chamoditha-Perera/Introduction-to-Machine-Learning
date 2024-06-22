from sklearn.tree import DecisionTreeClassifier
import pandas as pd

df = pd.read_csv('data/student_performance_large.csv')
X = df[['Class Attendance', 'Previous Grades', 'Pass/Fail']]
y = df['Pass/Fail']

model = DecisionTreeClassifier()
model.fit(X, y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.svm import SVC

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

