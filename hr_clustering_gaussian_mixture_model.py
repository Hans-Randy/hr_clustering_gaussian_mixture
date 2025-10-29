import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
from skimage.transform import rotate
from skimage.util import random_noise
from skimage import exposure
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

RANDOM_STATE = 48

def plot_samples(data, title, n_samples=10, n_cols=5, img_h=64, img_w=64):
    """Helper function to plot sample images."""
    print(f"Generating plot: {title}")
    plt.figure(figsize=(1.5 * n_cols, 2.2 * (n_samples // n_cols + 1)))
    plt.suptitle(title, fontsize=16, y=1.03)
    n_rows = int(np.ceil(n_samples / n_cols))
    
    for i in range(n_samples):
        if i >= len(data):
            break
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(data[i].reshape(img_h, img_w), cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
    
    plt.tight_layout()
    filename = title.lower().replace(' ', '_') + '.png'
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()

def plot_stratified_split(y_train, y_val, y_test, y_total):
    """Visualizes the class distribution across splits."""
    title = "Stratified Split Class Distribution"
    print(f"Generating plot: {title}")
    
    n_classes = len(np.unique(y_total))
    bins = np.arange(n_classes + 1) - 0.5
    
    plt.figure(figsize=(15, 5))
    
    # Training set
    plt.subplot(1, 3, 1)
    plt.hist(y_train, bins=bins, alpha=0.7, label='Train')
    plt.title(f'Training Set (n={len(y_train)})')
    plt.xlabel('Person ID')
    plt.ylabel('Count')
    plt.ylim(0, 10) # Olivetti has 10 images per person
    
    # Validation set
    plt.subplot(1, 3, 2)
    plt.hist(y_val, bins=bins, alpha=0.7, label='Validation', color='orange')
    plt.title(f'Validation Set (n={len(y_val)})')
    plt.xlabel('Person ID')
    plt.ylim(0, 10)
    
    # Test set
    plt.subplot(1, 3, 3)
    plt.hist(y_test, bins=bins, alpha=0.7, label='Test', color='green')
    plt.title(f'Test Set (n={len(y_test)})')
    plt.xlabel('Person ID')
    plt.ylim(0, 10)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = title.lower().replace(' ', '_') + '.png'
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()

def plot_dendrogram(X_train_pca):
    """Computes linkage and plots the dendrogram."""
    title = "Hierarchical Clustering Dendrogram"
    print(f"Generating plot: {title}")
    
    chosen_linkage = 'ward'

    # Compute the linkage matrix
    Z = linkage(X_train_pca, method=chosen_linkage)
    
    plt.figure(figsize=(25, 10))
    plt.title(title, fontsize=20)
    plt.xlabel('Sample Index (or Cluster Size)', fontsize=14)
    plt.ylabel('Distance (Ward)', fontsize=14)
    
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x-axis labels
        leaf_font_size=8.,  # font size for the x-axis labels
        truncate_mode='lastp',  # show only the last p merged clusters
        p=40,  # show the last 40 merges
        show_contracted=True,  # to get a more compact view
    )
    
    plt.axhline(y=17, color='r', linestyle='--', label='Cluster Cut-off (k=40)')
    plt.legend()
    plt.tight_layout()
    
    filename = title.lower().replace(' ', '_') + '.png'
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()
    
    return Z

def plot_hierarchical_clusters(X_train, X_train_pca, n_clusters=40):
    """Fits AgglomerativeClustering and visualizes sample clusters."""
    print("\nFitting AgglomerativeClustering and visualizing clusters.")
    
    # Fit the model
    # We use n_clusters=40 to match the number of individuals
    agg_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    y_pred_train = agg_cluster.fit_predict(X_train_pca)
    
    # Visualize samples from a few selected clusters
    # We'll pick 5 clusters to show
    n_clusters_to_show = 5
    n_samples_per_cluster = 5
    
    fig, axes = plt.subplots(n_clusters_to_show, n_samples_per_cluster, figsize=(10, 10))
    fig.suptitle(f'Sample Images from {n_clusters_to_show} Hierarchical Clusters (k=40)', fontsize=16, y=1.03)
    
    for i in range(n_clusters_to_show):
        cluster_indices = np.where(y_pred_train == i)[0]
        
        # Pick random samples from this cluster
        if len(cluster_indices) > 0:
            selected_indices = np.random.choice(cluster_indices, n_samples_per_cluster, replace=False)
            
            for j, idx in enumerate(selected_indices):
                ax = axes[i, j]
                ax.imshow(X_train[idx].reshape(64, 64), cmap=plt.cm.gray)
                ax.set_xticks(())
                ax.set_yticks(())
                if j == 0:
                    ax.set_ylabel(f'Cluster {i}', fontsize=12)
        else:
            # Handle empty clusters if any
            for j in range(n_samples_per_cluster):
                axes[i, j].set_visible(False)

    plt.tight_layout()
    filename = 'hierarchical_cluster_samples.png'
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()

def find_best_gmm(X_train_pca):
    """Finds the best GMM covariance type and number of clusters using AIC/BIC."""
    print("\nFinding best GMM using AIC/BIC...")
    
    n_components_range = range(30, 51)  # Test around 40 clusters
    covariance_types = ['spherical', 'tied', 'diag', 'full']
    
    aic_scores = {cov_type: [] for cov_type in covariance_types}
    bic_scores = {cov_type: [] for cov_type in covariance_types}
    
    best_bic = np.inf
    best_gmm = None
    
    for cov_type in covariance_types:
        print(f"  Testing covariance type: {cov_type}")
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type, random_state=RANDOM_STATE)
            gmm.fit(X_train_pca)
            
            aic_scores[cov_type].append(gmm.aic(X_train_pca))
            bic_scores[cov_type].append(gmm.bic(X_train_pca))
            
            if bic_scores[cov_type][-1] < best_bic:
                best_bic = bic_scores[cov_type][-1]
                best_gmm = gmm
    
    print(f"\nBest GMM found: {best_gmm.n_components} components, covariance type '{best_gmm.covariance_type}'")
    
    # Plot AIC/BIC scores
    title = "GMM AIC and BIC Scores by Covariance Type"
    print(f"Generating plot: {title}")
    
    plt.figure(figsize=(12, 6))
    
    # Plot BIC
    plt.subplot(1, 2, 1)
    for cov_type in covariance_types:
        plt.plot(n_components_range, bic_scores[cov_type], label=f'BIC {cov_type}')
    plt.title('BIC Scores')
    plt.xlabel('Number of Components')
    plt.ylabel('BIC')
    plt.legend()
    
    # Plot AIC
    plt.subplot(1, 2, 2)
    for cov_type in covariance_types:
        plt.plot(n_components_range, aic_scores[cov_type], label=f'AIC {cov_type}')
    plt.title('AIC Scores')
    plt.xlabel('Number of Components')
    plt.ylabel('AIC')
    plt.legend()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = 'gmm_aic_bic_scores.png'
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()
    
    return best_gmm

def plot_gmm_clusters(X_train, X_train_pca, best_gmm):
    """Visualizes sample images from the best GMM's clusters."""
    print("Visualizing sample clusters from best GMM...")
    
    y_pred_gmm = best_gmm.predict(X_train_pca)
    n_clusters_to_show = 5
    n_samples_per_cluster = 5
    
    fig, axes = plt.subplots(n_clusters_to_show, n_samples_per_cluster, figsize=(10, 10))
    fig.suptitle(f'Sample Images from Best GMM Clusters (k={best_gmm.n_components}, cov={best_gmm.covariance_type})', fontsize=16, y=1.03)
    
    for i in range(n_clusters_to_show):
        # Find clusters that are not empty
        cluster_id = np.unique(y_pred_gmm)[i]
        cluster_indices = np.where(y_pred_gmm == cluster_id)[0]
        
        if len(cluster_indices) > 0:
            selected_indices = np.random.choice(cluster_indices, n_samples_per_cluster, replace=True)
            
            for j, idx in enumerate(selected_indices):
                ax = axes[i, j]
                ax.imshow(X_train[idx].reshape(64, 64), cmap=plt.cm.gray)
                ax.set_xticks(())
                ax.set_yticks(())
                if j == 0:
                    ax.set_ylabel(f'Cluster {cluster_id}', fontsize=12)
        else:
             for j in range(n_samples_per_cluster):
                axes[i, j].set_visible(False)

    plt.tight_layout()
    filename = 'gmm_cluster_samples.png'
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()


def create_anomalies(images, n_anomalies=5):
    """Applies transformations to create anomalous images."""
    print("\nCreating anomalous images...")
    anomalies = []
    original_images = images[:n_anomalies]
    
    # 1. Rotate
    anomalies.append(rotate(original_images[0], angle=45, resize=False, cval=0, preserve_range=True))
    # 2. Flip
    anomalies.append(np.fliplr(original_images[1]))
    # 3. Darken
    anomalies.append(exposure.adjust_gamma(original_images[2], gamma=2.0))
    # 4. Noise
    anomalies.append(random_noise(original_images[3], mode='s&p', amount=0.3))
    # 5. Rotate 90
    anomalies.append(rotate(original_images[4], angle=90, resize=False, cval=0, preserve_range=True))
    
    anomalies = np.array(anomalies)
    
    # Plot anomalies
    title = "Transformed Anomalous Images"
    print(f"Generating plot: {title}")
    
    fig, axes = plt.subplots(2, n_anomalies, figsize=(12, 5))
    fig.suptitle(title, fontsize=16)
    
    for i in range(n_anomalies):
        # Plot original
        axes[0, i].imshow(original_images[i].reshape(64, 64), cmap=plt.cm.gray)
        axes[0, i].set_xticks(())
        axes[0, i].set_yticks(())
        if i == 0:
            axes[0, i].set_ylabel('Original')
            
        # Plot anomaly
        axes[1, i].imshow(anomalies[i].reshape(64, 64), cmap=plt.cm.gray)
        axes[1, i].set_xticks(())
        axes[1, i].set_yticks(())
        if i == 0:
            axes[1, i].set_ylabel('Anomaly')

    axes[1, 0].set_title('Rotate 45')
    axes[1, 1].set_title('Flip LR')
    axes[1, 2].set_title('Darken')
    axes[1, 3].set_title('S&P Noise')
    axes[1, 4].set_title('Rotate 90')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = 'anomalous_images.png'
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()
    
    return anomalies


def main():
    print("Loading Olivetti Faces dataset...")
    # Using 'data_home' to cache the dataset in a local directory
    if not os.path.exists('olivetti_faces_data'):
        os.makedirs('olivetti_faces_data')
        
    olivetti = fetch_olivetti_faces(data_home='olivetti_faces_data', shuffle=True, random_state=RANDOM_STATE)
    X_hans_randy = olivetti.data  # Image features
    y_hans_randy = olivetti.target  # Labels (person ID)
    
    print(f"Dataset loaded. Shape of X: {X_hans_randy.shape}, Shape of y: {y_hans_randy.shape}")
    
    # Display a few sample images
    plot_samples(X_hans_randy, "Sample Olivetti Faces", n_samples=10)

    print("\nSplitting dataset with stratified sampling...")
    
    # First split: 70% train, 30% temp (for val/test)
    X_train, X_temp, y_train, y_temp = train_test_split(X_hans_randy, y_hans_randy, test_size=0.3, random_state=RANDOM_STATE, stratify=y_hans_randy)
    
    # Second split: 50% of temp (15% of total) for val, 50% (15% of total) for test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp)
    
    print(f"Split complete:")
    print(f"  Training set:   X={X_train.shape}, y={y_train.shape}")
    print(f"  Validation set: X={X_val.shape}, y={y_val.shape}")
    print(f"  Test set:       X={X_test.shape}, y={y_test.shape}")

    # Plot the stratified distribution
    plot_stratified_split(y_train, y_val, y_test, y_hans_randy)

    print("\nApplying PCA...")
    # Retain 99% of variance
    pca = PCA(n_components=0.99, whiten=True, random_state=RANDOM_STATE)
    
    # Fit PCA only on the training data
    pca.fit(X_train)
    
    # Transform all sets
    X_train_pca = pca.transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    
    print(f"PCA complete. Number of principal components selected: {pca.n_components_}")
    print(f"  Original dimensions: {X_train.shape[1]}")
    print(f"  Reduced dimensions:  {X_train_pca.shape[1]}")

    # --- Hierarchical Clustering and Visualization ---
    # Plot dendrogram
    plot_dendrogram(X_train_pca)
    
    # Plot sample clusters
    plot_hierarchical_clusters(X_train, X_train_pca, n_clusters=40)
    print("Hierarchical clustering and visualization complete.")

    # --- Gaussian Mixture Model (GMM) Clustering ---
    best_gmm = find_best_gmm(X_train_pca)
    
    # Visualize sample images from the best GMM
    plot_gmm_clusters(X_train, X_train_pca, best_gmm)

    # --- Output Hard & Soft Clustering Assignments ---
    print("\n[GMM Hard and Soft Clustering Assignments (showing first 5 test samples):")
    
    # Use the test set for this output
    test_samples_pca = X_test_pca[:5]
    
    # Hard assignments
    hard_assignments = best_gmm.predict(test_samples_pca)
    print(f"  Hard Assignments (Cluster IDs): {hard_assignments}")
    
    # Soft clustering probabilities
    soft_probabilities = best_gmm.predict_proba(test_samples_pca)
    print("  Soft Probabilities (Likelihood of belonging to each cluster):")
    for i in range(5):
        # Print probabilities rounded to 2 decimal places
        probs_str = [f"{p:.2f}" for p in soft_probabilities[i, :10]] # Show first 10 clusters
        print(f"    Sample {i} (Cluster {hard_assignments[i]}): [{', '.join(probs_str)} ...]")

    # --- Generate New Faces Using the Model ---
    print("\nGenerating new synthetic faces using GMM...")
    
    # Generate new samples in the PCA space
    n_new_faces = 10
    new_faces_pca, _ = best_gmm.sample(n_samples=n_new_faces)
    
    # Convert back to original image space
    new_faces = pca.inverse_transform(new_faces_pca)
    
    # Visualize the generated faces
    plot_samples(new_faces, "Generated Synthetic Faces", n_samples=n_new_faces)

    # --- Modify Some Images (Create Anomalies) ---
    # Use the original test set images (X_test)
    anomalous_images = create_anomalies(X_test, n_anomalies=5)

    # --- Detect Anomalies Using the Model ---
    print("\nDetecting anomalies using GMM score_samples()...")
    
    # We must transform the anomalies into the PCA space
    # Flatten anomalies first (shape (5, 64, 64) -> (5, 4096))
    anomalies_flat = anomalous_images.reshape(anomalous_images.shape[0], -1)
    anomalies_pca = pca.transform(anomalies_flat)
    
    # Get log-likelihood scores for normal test images
    normal_scores = best_gmm.score_samples(X_test_pca[:10]) # First 10 normal test images
    
    # Get log-likelihood scores for anomalous images
    anomaly_scores = best_gmm.score_samples(anomalies_pca)
    
    print("  Log-Likelihood Scores (Higher is better/more normal):")
    print(f"    Average score for NORMAL images:  {np.mean(normal_scores):.2f}")
    print(f"    Average score for ANOMALOUS images: {np.mean(anomaly_scores):.2f}")
    
    print("\n  Individual Scores:")
    print("    Normal Samples:  ", [f"{s:.2f}" for s in normal_scores])
    print("    Anomalous Samples:", [f"{s:.2f}" for s in anomaly_scores])
    
    print("\n" + "=" * 50)
    print("Script Execution Complete.")
    print("All plots have been saved as .png files in the current directory.")

if __name__ == "__main__":
    main()