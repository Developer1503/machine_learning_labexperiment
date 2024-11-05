# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('/mnt/data/Task-4_Customer_Segmentation.csv')

# Display basic info and first few rows of the dataset
print("Dataset Information:")
print(data.info())
print("\nFirst few rows of the dataset:")
print(data.head())

# Select features for clustering (e.g., Age, Annual Income, Spending Score)
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Choose the optimal number of clusters (e.g., k=4 based on the Elbow curve)
k_optimal = 4
kmeans = KMeans(n_clusters=k_optimal, random_state=0)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters in a 2D scatter plot (using two features for simplicity)
plt.figure(figsize=(10, 8))
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set1', s=100, alpha=0.7)
plt.title(f'Customer Segmentation with K-means (k={k_optimal})')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
