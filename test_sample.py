import numpy as np
import os
import cv2
from sklearn.cluster import KMeans
from src.backbones import get_model

def select_diverse_samples(data_folder, n_samples=8, method='kmeans'):

    model = get_model('dinov2_vits14', 'cuda', smaller_edge_size=448)

    all_samples = sorted(os.listdir(data_folder))
    all_features = []

    print(f"Extracting features from {len(all_samples)} samples...")
    for sample_name in all_samples:
        img_path = os.path.join(data_folder, sample_name)
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        img_tensor, grid_size = model.prepare_image(image)
        features = model.extract_features(img_tensor)
        img_feature = features.mean(axis=0)
        all_features.append(img_feature)

    all_features = np.array(all_features)

    if method == 'kmeans':

        kmeans = KMeans(n_clusters=n_samples, random_state=42)
        kmeans.fit(all_features)

        selected_indices = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(all_features - center, axis=1)
            closest_idx = np.argmin(distances)
            selected_indices.append(closest_idx)

    elif method == 'maximin':
        selected_indices = []

        selected_indices.append(np.random.randint(len(all_features)))

        for _ in range(n_samples - 1):
            min_distances = []
            for i in range(len(all_features)):
                if i in selected_indices:
                    min_distances.append(-1)
                else:
                    distances_to_selected = [
                        np.linalg.norm(all_features[i] - all_features[j])
                        for j in selected_indices
                    ]
                    min_distances.append(min(distances_to_selected))

            next_idx = np.argmax(min_distances)
            selected_indices.append(next_idx)

    elif method == 'hybrid':

        n_clusters = n_samples // 2

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(all_features)

        selected_indices = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(all_features - center, axis=1)
            closest_idx = np.argmin(distances)
            selected_indices.append(closest_idx)

        remaining = n_samples - len(selected_indices)
        distances_to_centers = np.min([
            np.linalg.norm(all_features - center, axis=1)
            for center in kmeans.cluster_centers_
        ], axis=0)

        boundary_indices = np.argsort(distances_to_centers)[-remaining:]
        selected_indices.extend(boundary_indices.tolist())

    selected_samples = [all_samples[i] for i in selected_indices]

    print(f"\nSelected {len(selected_samples)} diverse samples:")
    for i, sample in enumerate(selected_samples):
        print(f"  {i+1}. {sample}")

    return selected_samples

def visualize_selection(data_folder, selected_samples):

    import matplotlib.pyplot as plt

    n = len(selected_samples)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
    axes = axes.flatten()

    for i, sample_name in enumerate(selected_samples):
        img_path = os.path.join(data_folder, sample_name)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        axes[i].imshow(img)
        axes[i].set_title(f"Sample {i+1}")
        axes[i].axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('selected_samples.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: selected_samples.png")

if __name__ == "__main__":
    data_folder = "data/mvtec_anomaly_detection/screw/train/good"

    selected = select_diverse_samples(data_folder, n_samples=8, method='kmeans')

    visualize_selection(data_folder, selected)
