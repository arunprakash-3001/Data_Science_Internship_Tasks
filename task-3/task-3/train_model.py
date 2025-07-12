import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
df = pd.read_csv("iris.csv")

# Rename columns to make them easier to work with
df.columns = ['Id', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Select features (exclude Id)
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model as a .pkl file
pickle.dump(model, open("iris_model.pkl", "wb"))
print("✅ Model saved as iris_model.pkl")
