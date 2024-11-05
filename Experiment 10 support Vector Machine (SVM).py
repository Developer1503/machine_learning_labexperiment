# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/mnt/data/Task-2_Diabetes_Classification.csv')

# Display basic info and first few rows of the dataset
print("Dataset Information:")
print(data.info())
print("\nFirst few rows of the dataset:")
print(data.head())

# Define features (e.g., Glucose, BloodPressure, Age, etc.) and target variable (e.g., Outcome)
X = data[['Glucose', 'BloodPressure', 'BMI', 'Age']]  # Select relevant columns
y = data['Outcome']  # Target variable for classification

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the Support Vector Classifier
model = SVC(kernel='linear', random_state=0)  # Using a linear kernel
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Diabetic'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
