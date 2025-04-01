import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import cdist

# Simulate delivery locations (latitude, longitude)
np.random.seed(42)
locations = np.array([
    [20, 35], [22, 37], [24, 32], [30, 50], [40, 65], 
    [45, 70], [60, 60], [70, 80], [75, 90], [80, 95]
])

# Perform AGNES (Hierarchical Clustering) using Ward's method (default)
Z = linkage(locations, method='ward')  # 'ward' minimizes variance within clusters

# Create clusters - decide how many clusters you want (e.g., 3 clusters here)
num_clusters = 3
clusters = fcluster(Z, t=num_clusters, criterion='maxclust')

# Nearest Neighbor algorithm to find the optimal route in each cluster
def nearest_neighbor_route(locations):
    route = [locations[0]]
    remaining_locations = locations[1:].tolist()

    while remaining_locations:
        last_location = route[-1]
        distances = cdist([last_location], remaining_locations)
        nearest_idx = np.argmin(distances)
        route.append(remaining_locations.pop(nearest_idx))

    return np.array(route)

# Plot the dendrogram to visualize the clustering process
plt.figure(figsize=(8, 6))
plt.title('Dendrogram of Delivery Locations (Ward’s Method Clustering)')
dendrogram(Z)
plt.xlabel('Delivery Locations')
plt.ylabel('Distance')
plt.show()

# Plot the locations, the clusters, and the optimal routes
plt.figure(figsize=(8, 6))

# Plot each cluster and its centroid
for i in range(1, num_clusters + 1):
    cluster_locations = locations[clusters == i]
    plt.scatter(cluster_locations[:, 0], cluster_locations[:, 1], label=f'Cluster {i}')
    
    # Calculate and plot the route for the cluster
    route = nearest_neighbor_route(cluster_locations)
    plt.plot(route[:, 0], route[:, 1], marker='o', label=f'Route for Cluster {i}')

plt.title('Optimal Delivery Routes with Ward’s Method Clustering')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.show()
