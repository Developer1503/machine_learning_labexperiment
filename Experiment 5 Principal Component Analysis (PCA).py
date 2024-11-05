# Import necessary libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('/mnt/data/Task-2_Diabetes_Classification.csv')

# Display basic info and first few rows of the dataset
print("Dataset Information:")
print(data.info())
print("\nFirst few rows of the dataset:")
print(data.head())

# Define features (e.g., Glucose, BloodPressure, Age, etc.)
X = data[['Glucose', 'BloodPressure', 'Age', 'Insulin', 'BMI']]  # Select relevant columns

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce to 2 principal components for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by each component: {explained_variance}")
print(f"Total explained variance by 2 components: {explained_variance.sum()}")

# Convert the PCA result to a DataFrame
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Outcome'] = data['Outcome']  # Add target variable for color coding in visualization

# Visualize the results
plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Outcome', palette='Set1', alpha=0.7)
plt.title('PCA of Diabetes Dataset (2 Components)')
plt.xlabel(f'PC1 ({explained_variance[0]:.2%} Variance)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2%} Variance)')
plt.legend(title='Outcome', labels=['Non-Diabetic', 'Diabetic'])
plt.show()
