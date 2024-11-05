# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('/mnt/data/Task-4_Customer_Segmentation.csv')

# Display basic info and first few rows of the dataset
print("Dataset Information:")
print(data.info())
print("\nFirst few rows of the dataset:")
print(data.head())

# Histogram for Age distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['Age'], kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Scatter plot for Age vs Annual Income
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='Age', y='Annual Income (k$)', hue='Gender', palette='viridis')
plt.title('Age vs Annual Income')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.legend(title='Gender')
plt.show()

# Box plot for Spending Score by Gender
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x='Gender', y='Spending Score (1-100)', palette='Set2')
plt.title('Spending Score by Gender')
plt.xlabel('Gender')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Heatmap for Correlation Matrix
plt.figure(figsize=(10, 8))
correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
