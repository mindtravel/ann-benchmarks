#!/usr/bin/env python3
"""
使用 ivftensor 的聚类接口，输出某一个 query 到各个聚类中心的距离分布。

基于 PyIVFTensor.ClusterDataset 做 K-means 聚类，然后计算 query 到每个 centroid 的 L2 距离。
"""

import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

# 加载 PyIVFTensor（与 ivf_tensor module 一致）
ivftensor_path = "/home/diy/lzx/ivftensor"
module_paths = os.path.join(ivftensor_path, "python/build")
sys.path.insert(0, module_paths)
import PyIVFTensor

from ann_benchmarks.datasets import get_dataset


def run_clustering_and_get_centroids(
    dataset_name: str,
    n_clusters: int = 1000,
    kmeans_iters: int = 20,
    distance_mode: int = 0,  # 0=L2 (euclidean)
    use_cpu: bool = False,
):
    """加载数据并用 ivftensor（或 sklearn）做 K-means 聚类，返回 centroids。"""
    hdf5_file, _ = get_dataset(dataset_name)
    train = np.array(hdf5_file["train"], dtype=np.float32)
    test = np.array(hdf5_file["test"], dtype=np.float32)
    hdf5_file.close()

    if use_cpu:
        return _cluster_sklearn(train, test, n_clusters, kmeans_iters)
    return _cluster_ivftensor(train, test, n_clusters, kmeans_iters, distance_mode)


def _cluster_sklearn(train, test, n_clusters, kmeans_iters):
    """使用 sklearn KMeans（CPU）聚类。"""
    from sklearn.cluster import KMeans

    print(f"Running sklearn K-means (CPU): {train.shape[0]} vectors, {n_clusters} clusters, {kmeans_iters} iters...")
    kmeans = KMeans(n_clusters=n_clusters, max_iter=kmeans_iters, n_init=1, random_state=42)
    kmeans.fit(train)
    centroids = kmeans.cluster_centers_.astype(np.float32)
    cluster_counts = np.bincount(kmeans.labels_, minlength=n_clusters)
    print(f"Clustering done. Centroids shape: {centroids.shape}")
    return train, test, centroids, cluster_counts


def _cluster_ivftensor(train, test, n_clusters, kmeans_iters, distance_mode):
    """使用 ivftensor PyIVFTensor.ClusterDataset（GPU）聚类。"""
    print(f"Running ivftensor K-means (GPU): {train.shape[0]} vectors, {n_clusters} clusters, {kmeans_iters} iters...")
    dataset = PyIVFTensor.ClusterDataset()
    dataset.init_with_kmeans(
        train,
        n_clusters=n_clusters,
        kmeans_iters=kmeans_iters,
        use_minibatch=False,
        distance_mode=distance_mode,
        use_interleaved=True,
    )

    (_, _, centroids, _, cluster_counts, _) = dataset.get_data()
    centroids = np.array(centroids)
    print(f"Clustering done. Centroids shape: {centroids.shape}")
    return train, test, centroids, cluster_counts


def compute_query_to_centroid_distances(query: np.ndarray, centroids: np.ndarray):
    """计算 query 到每个 centroid 的 L2 距离。"""
    diff = centroids - query
    return np.linalg.norm(diff, axis=1)


def plot_centroid_distance_distribution(
    distances: np.ndarray,
    query_idx: int,
    n_clusters: int,
    out_path: str,
):
    """绘制 query 到各聚类中心的距离分布。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # 1. 直方图
    ax = axes[0]
    ax.hist(distances, bins=60, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(distances.min(), color="green", linestyle="--", linewidth=1.5, label=f"min={distances.min():.1f}")
    ax.axvline(np.median(distances), color="orange", linestyle="--", linewidth=1, label=f"median={np.median(distances):.1f}")
    ax.axvline(distances.max(), color="red", linestyle="--", linewidth=1, label=f"max={distances.max():.1f}")
    ax.set_xlabel("Distance to centroid (L2)")
    ax.set_ylabel("Count (number of clusters)")
    ax.set_title(f"Query #{query_idx} vs {n_clusters} cluster centroids: distance distribution")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # 2. 按距离排序后的曲线（便于看最近/最远 centroid）
    ax = axes[1]
    sorted_dists = np.sort(distances)
    ax.plot(np.arange(len(sorted_dists)), sorted_dists, color="steelblue", linewidth=1)
    ax.set_xlabel("Cluster index (sorted by distance)")
    ax.set_ylabel("Distance (L2)")
    ax.set_title(f"Query #{query_idx}: distances to centroids (sorted ascending)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Query 到各聚类中心的距离分布（ivftensor 聚类）"
    )
    parser.add_argument(
        "--dataset",
        default="sift-128-euclidean",
        help="数据集名称",
    )
    parser.add_argument(
        "--query",
        type=int,
        default=0,
        help="query 索引 (0 ~ 9999)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=1000,
        help="聚类数量，默认 1000",
    )
    parser.add_argument(
        "--kmeans-iters",
        type=int,
        default=20,
        help="K-means 迭代次数",
    )
    parser.add_argument(
        "--out",
        default="scripts/experiment/sift1m_query_to_centroids_dist.png",
        help="输出图片路径",
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="使用 sklearn KMeans（CPU），无 GPU 或 CUDA 失败时可用",
    )
    args = parser.parse_args()

    train, test, centroids, cluster_counts = run_clustering_and_get_centroids(
        args.dataset,
        n_clusters=args.n_clusters,
        kmeans_iters=args.kmeans_iters,
        distance_mode=PyIVFTensor.DISTANCE_L2,
        use_cpu=args.use_cpu,
    )

    query = test[args.query]
    distances = compute_query_to_centroid_distances(query, centroids)

    print(f"\nQuery #{args.query} to {len(centroids)} centroids:")
    print(f"  min={distances.min():.2f}, max={distances.max():.2f}")
    print(f"  mean={distances.mean():.2f}, median={np.median(distances):.2f}, std={distances.std():.2f}")
    print(f"  Top-5 nearest centroid distances: {np.sort(distances)[:5]}")

    plot_centroid_distance_distribution(
        distances,
        args.query,
        len(centroids),
        args.out,
    )


if __name__ == "__main__":
    main()
