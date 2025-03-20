from sklearn.metrics import silhouette_score

X_train_num = X_train[numerical_features]
X_test_num = X_test[numerical_features]

# 1. Perform PCA on training data
pca = PCA().fit(X_train_num)

# Plot explained variance ratio to determine optimal components
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Components')
plt.axhline(y=0.80, color='r', linestyle='--', label='80% Variance')
plt.grid(True)
plt.legend()
plt.show()

# Choose number of components that explain 80% of variance
n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.80) + 1
print(f"Number of components needed to explain 80% of variance: {n_components}")
print("-"*65)

# 2. Apply PCA with the determined number of components
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_num)
X_test_pca = pca.transform(X_test_num)  # Apply same transformation to test data

# 3. First K-means on original features
# Calaulate inertia for each K on original features
inertia_original = []
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train_num)
    inertia_original.append(kmeans.inertia_)

# Calculate silhouette scores for different values of K on original features
silhouette_scores_original = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_train_num)
    
    silhouette_avg = silhouette_score(X_train_num, cluster_labels)
    silhouette_scores_original.append(silhouette_avg)
    print(f"Original features - For n_clusters = {k}, silhouette score: {silhouette_avg:.3f}")

# Find optimal K for original features
optimal_k_original = K_range[np.argmax(silhouette_scores_original)]
print(f"Optimal number of clusters on original features: {optimal_k_original}")
print("-"*65)

# Apply K-means with optimal K on original features
kmeans_original = KMeans(n_clusters=optimal_k_original, random_state=42, n_init=10)
train_clusters_original = kmeans_original.fit_predict(X_train_num)
test_clusters_original = kmeans_original.predict(X_test_num)

# 4. Second K-means on PCA-transformed features
# Calaulate inertia for each K on PCA features
inertia_pca = []
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train_pca)
    inertia_pca.append(kmeans.inertia_)

# Calculate silhouette scores for different values of K on PCA features
silhouette_scores_pca = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_train_pca)
    
    silhouette_avg = silhouette_score(X_train_pca, cluster_labels)
    silhouette_scores_pca.append(silhouette_avg)
    print(f"PCA features - For n_clusters = {k}, silhouette score: {silhouette_avg:.3f}")

# Find optimal K for PCA features
optimal_k_pca = K_range[np.argmax(silhouette_scores_pca)]
print(f"Optimal number of clusters on PCA features: {optimal_k_pca}")
print("-"*65)

# Apply K-means with optimal K on PCA features
kmeans_pca = KMeans(n_clusters=optimal_k_pca, random_state=42, n_init=10)
train_clusters_pca = kmeans_pca.fit_predict(X_train_pca)
test_clusters_pca = kmeans_pca.predict(X_test_pca)

# 5. Compare clustering results
# Plot silhouette scores comparison
plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores_original, marker='o', label='Original Features')
plt.plot(K_range, silhouette_scores_pca, marker='s', label='PCA Features')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Comparison: Original vs. PCA Features')
plt.legend()
plt.grid(True)
plt.show()

# Plot inertia comparison
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia_original, marker='o', label='Original Features')
plt.plot(K_range, inertia_pca, marker='s', label='PCA Features')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Inertia Comparison: Original vs. PCA Features')
plt.legend()
plt.grid(True)
plt.show()

# 6. Visualize clusters vs. actual classes
if n_components >= 2:
    # Create a figure with 2x2 subplots
    plt.figure(figsize=(14, 10))
    
    # Plot 1: K-means clusters on original features
    # We'll use the first 2 PCA components just for visualization
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_clusters_original, cmap='viridis', s=50)
    plt.title(f'K-means on Original Features (k={optimal_k_original})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter)
    
    # Plot 2: K-means clusters on PCA features
    plt.subplot(2, 2, 2)
    scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_clusters_pca, cmap='viridis', s=50)
    plt.title(f'K-means on PCA Features (k={optimal_k_pca})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter)
    
    # Plot 3: Actual classes
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', s=50)
    plt.title('Actual Deactivation Classes')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter)
    
    # Plot 4: Distribution of actual classes within clusters
    plt.subplot(2, 2, 4)
    # For this plot, we'll choose the clustering with higher silhouette score
    if np.max(silhouette_scores_original) > np.max(silhouette_scores_pca):
        best_clusters = train_clusters_original
        cluster_type = "Original Features"
        n_clusters = optimal_k_original
    else:
        best_clusters = train_clusters_pca
        cluster_type = "PCA Features"
        n_clusters = optimal_k_pca
    
    # Count deactivated catalysts in each cluster
    cluster_counts = pd.DataFrame({
        'Cluster': best_clusters,
        'Deactivated': y_train.reset_index(drop=True)
    }).groupby('Cluster')['Deactivated'].value_counts(normalize=True).unstack().fillna(0)
    
    # Plot as stacked bar chart
    cluster_counts.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title(f'Class Distribution in Each Cluster ({cluster_type})')
    plt.xlabel('Cluster')
    plt.ylabel('Proportion')
    plt.legend(title='Deactivated')
    
    plt.tight_layout()
    plt.show()

# 7. Summary of findings
better_method = "original features" if np.max(silhouette_scores_original) > np.max(silhouette_scores_pca) else "PCA features"
print(f"\nSummary of Findings:")
print(f"- PCA required {n_components} components to explain 80% of variance")
print(f"- Optimal number of clusters: {optimal_k_original} for original features, {optimal_k_pca} for PCA features")
print(f"- Best silhouette score: {np.max(silhouette_scores_original):.3f} for original features, {np.max(silhouette_scores_pca):.3f} for PCA features")
print(f"- Clustering with {better_method} yielded better separation based on silhouette scores")