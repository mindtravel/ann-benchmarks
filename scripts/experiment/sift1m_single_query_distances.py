#!/usr/bin/env python3
"""
针对某一个 query，画出其与 1M 数据集中所有向量的距离分布。

用于初步了解单个 query 在全量数据上的距离 landscape。
"""

import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# scripts/experiment/ -> 上两级到 ann-benchmarks 根目录
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from ann_benchmarks.datasets import get_dataset


def compute_query_distances(dataset_name: str, query_idx: int):
    """计算指定 query 与所有 train 向量的欧氏距离。"""
    hdf5_file, _ = get_dataset(dataset_name)
    train = np.array(hdf5_file["train"])
    test = np.array(hdf5_file["test"])
    neighbors = np.array(hdf5_file["neighbors"])
    distances_gt = np.array(hdf5_file["distances"])
    hdf5_file.close()

    query = test[query_idx]
    # 向量化计算: (1M,) 距离
    diff = train - query
    distances = np.linalg.norm(diff, axis=1)

    # ground truth top-10 用于标注
    top10_indices = neighbors[query_idx, :10]
    top10_dists = distances_gt[query_idx, :10]

    return distances, top10_indices, top10_dists, query_idx


def plot_single_query_distribution(
    distances: np.ndarray,
    top10_dists: np.ndarray,
    query_idx: int,
    out_path: str,
):
    """绘制单 query 与 1M 数据的距离分布。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 9))

    # 1. 直方图（全量 1M 距离）
    ax = axes[0]
    ax.hist(distances, bins=120, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(distances.min(), color="green", linestyle="--", linewidth=1.5, label=f"min={distances.min():.1f}")
    ax.axvline(np.median(distances), color="orange", linestyle="--", linewidth=1, label=f"median={np.median(distances):.1f}")
    ax.axvline(distances.max(), color="red", linestyle="--", linewidth=1, label=f"max={distances.max():.1f}")
    for i, d in enumerate(top10_dists[:5]):  # 标出 top-5
        ax.axvline(d, color="darkgreen", linestyle=":", alpha=0.8, linewidth=1)
    ax.set_xlabel("Distance (Euclidean)")
    ax.set_ylabel("Count")
    ax.set_title(f"Query #{query_idx} vs 1M vectors distance distribution")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # 2. 放大：只看距离较小的左尾（最近邻区域）
    ax = axes[1]
    # 取到 99.9% 分位或 top10 最大值的 2 倍，保证能看到最近邻区域
    right = max(np.percentile(distances, 1), top10_dists[-1] * 1.5)
    mask = distances <= right
    subset = distances[mask]
    ax.hist(subset, bins=80, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.3)
    for i, d in enumerate(top10_dists):
        ax.axvline(d, color="darkgreen", linestyle=":", alpha=0.9, linewidth=1.2, label=f"top-{i+1}" if i < 3 else None)
    ax.set_xlabel("Distance (Euclidean)")
    ax.set_ylabel("Count")
    ax.set_title(f"Query #{query_idx} zoomed: nearest-neighbor region (x <= {right:.0f})")
    if any(l for l in [f"top-{i+1}" for i in range(3)]):
        ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="单 query 与 1M 数据的距离分布"
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
        help="query 索引 (0 ~ 9999)，默认 0",
    )
    parser.add_argument(
        "--out",
        default="scripts/experiment/sift1m_single_query_dist.png",
        help="输出图片路径",
    )
    args = parser.parse_args()

    print(f"加载数据集: {args.dataset}")
    distances, top10_idx, top10_dists, qidx = compute_query_distances(args.dataset, args.query)

    print(f"\nQuery #{qidx} 与 1M 数据的距离统计:")
    print(f"  min={distances.min():.2f}, max={distances.max():.2f}")
    print(f"  mean={distances.mean():.2f}, median={np.median(distances):.2f}, std={distances.std():.2f}")
    print(f"  top-10 距离: {top10_dists}")

    plot_single_query_distribution(distances, top10_dists, qidx, args.out)


if __name__ == "__main__":
    main()
