# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
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

# Perform hierarchical clustering using the 'ward' linkage method
Z = linkage(X_scaled, method='ward')

# Plot the dendrogram
plt.figure(figsize=(12, 8))
dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=45, leaf_font_size=12, show_contracted=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index or Cluster size')
plt.ylabel('Distance')
plt.show()

# Optional: Create clusters based on the dendrogram cut-off
from scipy.cluster.hierarchy import fcluster
max_distance = 6  # Adjust this threshold based on the dendrogram for optimal cluster separation
data['Cluster'] = fcluster(Z, max_distance, criterion='distance')

# Visualize clusters in a 2D scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set1', s=100, alpha=0.7)
plt.title('Customer Segmentation with Hierarchical Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
