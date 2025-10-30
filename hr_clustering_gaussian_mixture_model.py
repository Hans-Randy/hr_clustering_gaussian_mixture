import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from skimage.transform import rotate
from skimage.util import random_noise
from skimage import exposure
import warnings
import os
import csv


warnings.filterwarnings("ignore")  # cleaner output
RANDOM_STATE = 48
np.random.seed(RANDOM_STATE)

IMG_H = 64
IMG_W = 64

OUTPUT_DIR = "output_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_samples(data, title, n_samples=10, n_cols=5, img_h=IMG_H, img_w=IMG_W):
    """Helper function to plot sample images and save to PNG."""
    print(f"Generating plot: {title}")
    plt.figure(figsize=(1.6 * n_cols, 2.3 * (n_samples // n_cols + 1)))
    plt.suptitle(title, fontsize=14, y=1.02)
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
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches="tight")
    print(f"Saved plot to {filename}")
    plt.close()


def plot_stratified_split(y_train, y_val, y_test, y_total):
    """Visualizes the class distribution across splits."""
    title = "Stratified Split Class Distribution"   
    print(f"Generating plot: {title}")
    n_classes = len(np.unique(y_total))
    bins = np.arange(n_classes + 1) - 0.5
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(y_train, bins=bins, alpha=0.8)
    plt.title(f'Training Set (n={len(y_train)})')
    plt.xlabel('Person ID'); plt.ylabel('Count'); plt.ylim(0, 10)

    plt.subplot(1, 3, 2)
    plt.hist(y_val, bins=bins, alpha=0.8, color='orange')
    plt.title(f'Validation Set (n={len(y_val)})')
    plt.xlabel('Person ID'); plt.ylim(0, 10)

    plt.subplot(1, 3, 3)
    plt.hist(y_test, bins=bins, alpha=0.8, color='green')
    plt.title(f'Test Set (n={len(y_test)})')
    plt.xlabel('Person ID'); plt.ylim(0, 10)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = title.lower().replace(' ', '_') + '.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches="tight")
    print(f"Saved plot to {filename}")
    plt.close()
    
def plot_pca_cumulative_variance(pca, retain=0.99):
    """
    Plot cumulative explained variance and show how many principal components
    are required to retain a given fraction.
    """

    # Cumulative explained variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    # PCs needed to reach the 'retain' threshold
    n_needed = int(np.searchsorted(cumvar, retain) + 1)

    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(cumvar) + 1), cumvar, marker='o')
    plt.axhline(y=retain, linestyle='--')  # horizontal dashed line at threshold
    plt.ylim(0, 1.02)
    plt.xlim(1, len(cumvar))
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title(f"PCA components (retain {int(retain*100)}%): n={n_needed}")

    # Optional: vertical helper line at n_needed
    plt.axvline(x=n_needed, linestyle=':', linewidth=1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_cumvar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to pca_cumvar.png")


def plot_dendrogram(X_train_pca, k=40):
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    import numpy as np
    import matplotlib.pyplot as plt

    print(f"Generating dendrogram and cut line for k={k} clusters...")
    Z = linkage(X_train_pca, method='ward')

    plt.figure(figsize=(24, 10))
    dendrogram(
        Z,
        leaf_rotation=90.,
        leaf_font_size=8.,
        truncate_mode='lastp',
        p=40,
        show_contracted=True,
    )
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample Index (or Cluster Size)")
    plt.ylabel("Distance (Ward)")

    # Add red cut line
    sorted_d = np.sort(Z[:, 2])
    cut_height = sorted_d[-k]
    plt.axhline(y=cut_height, color='red', linestyle='--', linewidth=2)
    plt.text(0, cut_height + 0.5, f'Cut line for k={k}', color='red', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "hierarchical_clustering_dendrogram_with_cut.png"), bbox_inches='tight')
    plt.close()
    print("Saved: hierarchical_clustering_dendrogram_with_cut.png")

    return Z



def visualize_clusters_from_labels(X, labels, title_prefix, n_clusters_to_show=5, n_samples_per_cluster=5):
    """Generic grid showing sample images from given clustering labels."""
    unique_clusters = np.unique(labels)
    n_show = min(n_clusters_to_show, len(unique_clusters))
    fig, axes = plt.subplots(n_show, n_samples_per_cluster, figsize=(10, 2 * n_show))
    fig.suptitle(f'{title_prefix} (showing {n_show} clusters)', fontsize=14, y=1.02)

    for i in range(n_show):
        cluster_id = unique_clusters[i]
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            for j in range(n_samples_per_cluster):
                axes[i, j].set_visible(False)
            continue
        selected_indices = np.random.choice(cluster_indices, n_samples_per_cluster, replace=True)
        for j, idx in enumerate(selected_indices):
            ax = axes[i, j]
            ax.imshow(X[idx].reshape(IMG_H, IMG_W), cmap=plt.cm.gray)
            ax.set_xticks(()); ax.set_yticks(())
            if j == 0:
                ax.set_ylabel(f'Cluster {cluster_id}', fontsize=10)

    plt.tight_layout()
    filename = title_prefix.lower().replace(' ', '_') + '.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches="tight")
    print(f"Saved plot to {filename}")
    plt.close()


def cut_dendrogram_and_visualize(Z, X_train, k=40):
    """Cut dendrogram to exactly k clusters and visualize samples."""
    print(f"Cutting dendrogram to k={k} clusters and visualizing samples...")
    labels = fcluster(Z, t=k, criterion='maxclust')
    visualize_clusters_from_labels(X_train, labels, f'Dendrogram-Cut Clusters (k={k})')
    return labels


def plot_hierarchical_clusters(X_train, X_train_pca, n_clusters=40):
    """Fits AgglomerativeClustering and visualizes sample clusters."""
    print("\nFitting AgglomerativeClustering and visualizing clusters.")
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    y_pred = agg.fit_predict(X_train_pca)
    visualize_clusters_from_labels(X_train, y_pred, f'Agglomerative Clusters (k={n_clusters})')
    return y_pred


def find_best_gmm(X_train_pca):
    """Select best GMM over cov types and components using BIC; also plot AIC/BIC."""
    print("\nFinding best GMM using AIC/BIC...")
    n_components_range = range(30, 51)  # around the expected 40 subjects
    covariance_types = ['spherical', 'tied', 'diag', 'full']

    aic_scores = {cov: [] for cov in covariance_types}
    bic_scores = {cov: [] for cov in covariance_types}

    best_bic = np.inf
    best_gmm = None

    for cov in covariance_types:
        print(f"  Testing covariance type: {cov}")
        for n in n_components_range:
            gmm = GaussianMixture(n_components=n, covariance_type=cov, random_state=RANDOM_STATE)
            gmm.fit(X_train_pca)
            aic = gmm.aic(X_train_pca); bic = gmm.bic(X_train_pca)
            aic_scores[cov].append(aic); bic_scores[cov].append(bic)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm

    print(f"Best GMM: {best_gmm.n_components} components, covariance '{best_gmm.covariance_type}'")

    title = "GMM AIC and BIC Scores by Covariance Type"
    print(f"Generating plot: {title}")
    plt.figure(figsize=(12, 5))
    # BIC
    plt.subplot(1, 2, 1)
    for cov in covariance_types:
        plt.plot(list(n_components_range), bic_scores[cov], label=f'BIC {cov}')
    plt.title('BIC vs Components'); plt.xlabel('Components'); plt.ylabel('BIC'); plt.legend()
    # AIC
    plt.subplot(1, 2, 2)
    for cov in covariance_types:
        plt.plot(list(n_components_range), aic_scores[cov], label=f'AIC {cov}')
    plt.title('AIC vs Components'); plt.xlabel('Components'); plt.ylabel('AIC'); plt.legend()
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = 'gmm_aic_bic_scores.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches="tight")
    print(f"Saved plot to {filename}")
    plt.close()

    return best_gmm


def plot_gmm_clusters(X, X_pca, gmm, n_clusters_to_show=5, n_samples_per_cluster=5):
    """Visualize sample images from best GMM's predicted clusters."""
    print("Visualizing sample clusters from best GMM...")
    labels = gmm.predict(X_pca)
    visualize_clusters_from_labels(X, labels,
                                   f'GMM Clusters (k={gmm.n_components}, cov={gmm.covariance_type})',
                                   n_clusters_to_show, n_samples_per_cluster)
    return labels


def create_anomalies(images, n_anomalies=5):
    """Applies transformations to create anomalous images and saves a PNG."""
    print("\nCreating anomalous images...")
    original_flat = images[:n_anomalies]
    original_2d = original_flat.reshape(n_anomalies, IMG_H, IMG_W)

    anomalies = []
    anomalies.append(rotate(original_2d[0], angle=45, resize=False, cval=0, preserve_range=True))
    anomalies.append(np.fliplr(original_2d[1]))
    anomalies.append(exposure.adjust_gamma(original_2d[2], gamma=2.0))
    anomalies.append(random_noise(original_2d[3], mode='s&p', amount=0.3))
    anomalies.append(rotate(original_2d[4], angle=90, resize=False, cval=0, preserve_range=True))
    anomalies = np.array(anomalies)

    title = "Transformed Anomalous Images"
    print(f"Generating plot: {title}")
    fig, axes = plt.subplots(2, n_anomalies, figsize=(12, 5))
    fig.suptitle(title, fontsize=14)
    for i in range(n_anomalies):
        axes[0, i].imshow(original_2d[i], cmap=plt.cm.gray); axes[0, i].set_xticks(()); axes[0, i].set_yticks(())
        if i == 0: axes[0, i].set_ylabel('Original', fontsize=10)
        axes[1, i].imshow(anomalies[i], cmap=plt.cm.gray); axes[1, i].set_xticks(()); axes[1, i].set_yticks(())
        if i == 0: axes[1, i].set_ylabel('Anomaly', fontsize=10)
    axes[1, 0].set_title('Rotate 45'); axes[1, 1].set_title('Flip LR')
    axes[1, 2].set_title('Darken'); axes[1, 3].set_title('S&P Noise'); axes[1, 4].set_title('Rotate 90')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = 'anomalous_images.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches="tight")
    print(f"Saved plot to {filename}")
    plt.close()

    return anomalies.reshape(anomalies.shape[0], -1)  # flattened


def save_hard_soft_assignments(base_name, labels, probs, ids=None):
    """Save hard labels and soft probabilities to CSV for ALL instances (spec requirement)."""
    n, k = probs.shape
    if ids is None:
        ids = np.arange(n)

    # Hard assignments
    hard_path = f"{base_name}_hard_assignments.csv"
    with open(hard_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "cluster_label"])
        for i in range(n):
            writer.writerow([ids[i], int(labels[i])])
    print(f"Saved hard assignments to {hard_path}")

    # Soft probabilities
    soft_path = f"{base_name}_soft_probabilities.csv"
    with open(soft_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["index"] + [f"p_cluster_{j}" for j in range(k)]
        writer.writerow(header)
        for i in range(n):
            writer.writerow([ids[i]] + list(np.asarray(probs[i], dtype=float)))
    print(f"Saved soft probabilities to {soft_path}")


def main():
    print("Loading Olivetti Faces dataset...")
    local_data_dir = 'data/olivetti_faces_data'
    os.makedirs(local_data_dir, exist_ok=True)
    olivetti = fetch_olivetti_faces(data_home=local_data_dir, shuffle=True, random_state=RANDOM_STATE)

    X_hans_randy = olivetti.data
    y_hans_randy = olivetti.target
    print(f"Dataset loaded. X shape: {X_hans_randy.shape}, y shape: {y_hans_randy.shape}")

    # Display samples
    plot_samples(X_hans_randy, "Sample Olivetti Faces", n_samples=10)

    # Stratified split: 70/15/15
    print("\nStratified train/val/test split (70/15/15)...")
    X_train, X_temp, y_train, y_temp = train_test_split(X_hans_randy, y_hans_randy, test_size=0.3, random_state=RANDOM_STATE, stratify=y_hans_randy)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp)
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape},   y={y_val.shape}")
    print(f"  Test:  X={X_test.shape},  y={y_test.shape}")
    plot_stratified_split(y_train, y_val, y_test, y_hans_randy)

    # PCA 99% variance (fit on train only)
    print("\nApplying PCA (retain 99% variance, whiten=True)...")
    retain=0.99
    pca = PCA(n_components=retain, whiten=True, random_state=RANDOM_STATE)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    print(f"PCA components: {pca.n_components_} | Original dims: {X_train.shape[1]} -> Reduced: {X_train_pca.shape[1]}")
    plot_pca_cumulative_variance(pca, retain=retain)
    
    # Dendrogram and cut to k=40 (explicitly satisfy spec)
    Z = plot_dendrogram(X_train_pca)
    _ = cut_dendrogram_and_visualize(Z, X_train, k=40)

    # Agglomerative clustering visualization (k=40)
    _ = plot_hierarchical_clusters(X_train, X_train_pca, n_clusters=40)
    print("Hierarchical clustering visualization complete.")

    # GMM selection + AIC/BIC plot
    best_gmm = find_best_gmm(X_train_pca)

    # Visualize GMM clusters on training set
    _ = plot_gmm_clusters(X_train, X_train_pca, best_gmm)

    # Hard + soft assignments for ALL images (spec asks for 'each image')
    print("\n[GMM Hard/Soft assignments for ALL images across train/val/test]")
    X_all = np.vstack([X_train, X_val, X_test])
    X_all_pca = pca.transform(X_all)
    hard_all = best_gmm.predict(X_all_pca)
    soft_all = best_gmm.predict_proba(X_all_pca)
    save_hard_soft_assignments("gmm_all_images", hard_all, soft_all)

    # Also show first 5 test samples in stdout (quick preview)
    print("\nPreview (first 5 test samples):")
    hard_preview = best_gmm.predict(X_test_pca[:5])
    soft_preview = best_gmm.predict_proba(X_test_pca[:5])
    print(f"  Hard Assignments: {hard_preview}")
    
    for i in range(5):
        probs_str = [f"{p:.2f}" for p in soft_preview[i][:min(10, soft_preview.shape[1])]]
        print(f"    Sample {i}: [{', '.join(probs_str)} ...]")

    # Generate synthetic faces
    print("\nGenerating synthetic faces via GMM.sample -> inverse PCA transform...")
    n_new_faces = 10
    new_faces_pca, _ = best_gmm.sample(n_samples=n_new_faces)
    new_faces = pca.inverse_transform(new_faces_pca)
    plot_samples(new_faces, "Generated Synthetic Faces", n_samples=n_new_faces)

    # Create anomalies
    anomalous_images = create_anomalies(X_test, n_anomalies=5)

    # Detect anomalies using score_samples
    print("\nDetecting anomalies with GMM.score_samples...")
    anomalies_pca = pca.transform(anomalous_images)  # they are already flattened
    normal_scores = best_gmm.score_samples(X_test_pca[:10])
    anomaly_scores = best_gmm.score_samples(anomalies_pca)
    print("  Log-Likelihood Scores (higher means more 'normal')")
    print(f"    Mean NORMAL (n=10):  {np.mean(normal_scores):.2f}")
    print(f"    Mean ANOMALY (n=5):  {np.mean(anomaly_scores):.2f}")
    print("    Normal:", [f"{s:.2f}" for s in normal_scores])
    print("    Anomaly:", [f"{s:.2f}" for s in anomaly_scores])

    # Optional: simple histogram comparison saved for report use
    plt.figure(figsize=(6,4))
    plt.hist(normal_scores, alpha=0.7, label='Normal (first 10 test)')
    plt.hist(anomaly_scores, alpha=0.7, label='Anomalies (5)')
    plt.title('Score Distributions: Normal vs Anomalies')
    plt.xlabel('Log-likelihood'); plt.ylabel('Count'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'anomaly_score_hist.png'), bbox_inches='tight')
    print("Saved plot to anomaly_score_hist.png")

    print("\n" + "=" * 60)
    print("Script Execution Complete.")

if __name__ == "__main__":
    main()
