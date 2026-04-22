#!/usr/bin/env python3
"""
分析 SIFT1M 数据集中 query 的 top-10 最近邻距离分布。

基于 ann-benchmarks 接口，使用 HDF5 中的预计算 ground truth (neighbors, distances)。
"""

import argparse
import os
import sys

import numpy as np

# 确保从 ann-benchmarks 根目录运行
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from ann_benchmarks.datasets import get_dataset


def load_topk_distances(dataset_name: str, k: int = 10):
    """加载数据集中每个 query 的 top-k 最近邻距离。"""
    hdf5_file, _ = get_dataset(dataset_name)
    distances = np.array(hdf5_file["distances"][:, :k])  # (n_queries, k)
    hdf5_file.close()
    return distances


def print_summary_stats(distances: np.ndarray, k: int):
    """打印各 rank 的统计摘要。"""
    print("\n" + "=" * 70)
    print("各 rank 距离统计 (min / 25% / 50% / 75% / max / mean / std)")
    print("=" * 70)
    for r in range(k):
        col = distances[:, r]
        p25, p50, p75 = np.percentile(col, [25, 50, 75])
        print(
            f"  Rank {r+1:2d}: min={col.min():.2f}  "
            f"25%={p25:.2f}  median={p50:.2f}  75%={p75:.2f}  "
            f"max={col.max():.2f}  mean={col.mean():.2f}  std={col.std():.2f}"
        )
    print("=" * 70)
    pooled = distances.flatten()
    print(f"\n全部 top-{k} 距离汇总: min={pooled.min():.2f}, max={pooled.max():.2f}, "
          f"mean={pooled.mean():.2f}, median={np.median(pooled):.2f}, std={pooled.std():.2f}")
    print()


def plot_distribution(distances: np.ndarray, k: int, out_dir: str):
    """生成可视化图表。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图。可运行: pip install matplotlib")
        return

    os.makedirs(out_dir, exist_ok=True)

    # 1. 按 rank 的箱线图：展示各 rank 距离的分布
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(1, k + 1)
    bp = ax.boxplot(
        [distances[:, r] for r in range(k)],
        positions=positions,
        widths=0.6,
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)
    ax.set_xlabel("Neighbor Rank (1 = 最近)")
    ax.set_ylabel("Distance (Euclidean)")
    ax.set_title("SIFT1M: Top-10 最近邻距离分布（按 rank）")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(i) for i in range(1, k + 1)])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "top10_distance_by_rank_boxplot.png"), dpi=150)
    plt.close()
    print(f"  已保存: {out_dir}/top10_distance_by_rank_boxplot.png")

    # 2. 全部 top-k 距离的直方图
    fig, ax = plt.subplots(figsize=(8, 5))
    pooled = distances.flatten()
    ax.hist(pooled, bins=80, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(pooled.mean(), color="red", linestyle="--", label=f"mean={pooled.mean():.1f}")
    ax.axvline(np.median(pooled), color="orange", linestyle="--", label=f"median={np.median(pooled):.1f}")
    ax.set_xlabel("Distance (Euclidean)")
    ax.set_ylabel("Count")
    ax.set_title("SIFT1M: 全部 Top-10 距离的直方图")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "top10_distance_histogram.png"), dpi=150)
    plt.close()
    print(f"  已保存: {out_dir}/top10_distance_histogram.png")

    # 3. 各 rank 的 violin 图（更细的分布形状）
    fig, ax = plt.subplots(figsize=(10, 6))
    parts = ax.violinplot(
        [distances[:, r] for r in range(k)],
        positions=positions,
        widths=0.8,
        showmeans=True,
        showmedians=True,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.6)
    ax.set_xlabel("Neighbor Rank (1 = 最近)")
    ax.set_ylabel("Distance (Euclidean)")
    ax.set_title("SIFT1M: Top-10 最近邻距离分布（Violin）")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(i) for i in range(1, k + 1)])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "top10_distance_by_rank_violin.png"), dpi=150)
    plt.close()
    print(f"  已保存: {out_dir}/top10_distance_by_rank_violin.png")


def main():
    parser = argparse.ArgumentParser(
        description="分析 SIFT1M query 的 top-10 最近邻距离分布"
    )
    parser.add_argument(
        "--dataset",
        default="sift-128-euclidean",
        help="数据集名称，如 sift-128-euclidean 或 SIFT1M-128-euclidean",
    )
    parser.add_argument(
        "-k", "--topk",
        type=int,
        default=10,
        help="分析的最近邻数量，默认 10",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="不生成图表，仅打印统计",
    )
    parser.add_argument(
        "--out-dir",
        default="scripts/sift1m_distance_analysis",
        help="图表输出目录",
    )
    args = parser.parse_args()

    print(f"加载数据集: {args.dataset}")
    distances = load_topk_distances(args.dataset, k=args.topk)
    n_queries = distances.shape[0]
    print(f"Query 数量: {n_queries}, 每个 query 的 top-{args.topk} 距离")

    print_summary_stats(distances, args.topk)

    if not args.no_plot:
        print("\n生成图表...")
        plot_distribution(distances, args.topk, args.out_dir)


if __name__ == "__main__":
    main()
