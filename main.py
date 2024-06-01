import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming you have already loaded the dataset into 'df'
df = pd.read_csv('data/music.csv')

# Display the shape of the dataframe
print(df.shape)

# Prepare the feature variables (X) and the target variable (y)
x = df.drop(columns='genre')
y = df['genre']

# Initialize the Decision Tree model
model = DecisionTreeClassifier()

# Train the model with the data
model.fit(x, y)

# Make predictions for a new data point (age 21, male)
predictions = model.predict([[21, 1]])
print("Predicted genre for age 21, male:", predictions[0])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Train the model on the training data
model.fit(x_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(x_test)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------


model = DecisionTreeClassifier()








