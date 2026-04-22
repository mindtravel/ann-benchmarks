"""
IVF-Tensor: GPU-accelerated IVF-Flat search using CUDA

This module wraps the CUDA implementation from ivftensor for use in ann-benchmarks.
It uses pybind11 Python bindings to call the compiled CUDA library functions.
"""

import os
import sys
import time
import gc
import ctypes
import numpy as np
from typing import Optional, List
import psutil

from ..base.module import BaseANN


def drop_system_cache():
    """清空系统 page cache（需要 root 权限）"""
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
        print("[System] Page cache dropped", flush=True)
    except PermissionError:
        pass  # 没有 root 权限，忽略


def force_release_memory():
    """强制 Python 释放内存给系统"""
    gc.collect()
    if hasattr(sys, "getallocatedblocks"):
        ctypes.CDLL(None).malloc_trim(0)
    print(
        f"[Memory] Forced garbage collection, RSS: {psutil.Process().memory_info().rss / 1024**3:.2f} GB",
        flush=True,
    )


def get_memory_usage_mb():
    """获取当前进程的内存使用量（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def log_memory(label: str):
    """打印当前内存使用情况"""
    mem_mb = get_memory_usage_mb()
    print(f"[MEMORY] {label}: {mem_mb:.1f} MB", flush=True)


# 尝试加载 IVFTensor Python 扩展模块
# 拉去仓库后记得改这里
ivftensor_path = "/home/diy/lzx/ivftensor"
project_path = "/home/diy/lzx/ann-benchmarks"

module_paths = os.path.join(ivftensor_path, "python/build")
sys.path.insert(0, module_paths)
sys.path.insert(0, os.path.join(ivftensor_path, "python"))

import PyIVFTensor
from cluster_cache import ClusterCache
from .ivf_tensor_pinned import (
    numpy_to_pinned,
    register_array_as_pinned,
    ReusablePinnedBuffer,
)

# 默认缓存目录，可通过环境变量 IVF_CLUSTER_CACHE_DIR 覆盖
_CLUSTER_CACHE_DIR = os.environ.get(
    "IVF_CLUSTER_CACHE_DIR",
    os.path.join(project_path, "data", "cluster_cache"),
)


class IVFTensor(BaseANN):
    """
    IVF-Tensor: GPU-accelerated IVF-Flat search

    This implementation uses CUDA for both clustering (K-means) and search operations.
    """

    def __init__(self, metric: str, method_param: dict):
        """
        Initialize IVF-Tensor algorithm.

        Args:
            metric: Distance metric ("angular" or "euclidean")
            method_param: Dictionary containing:
                - n_lists: Number of clusters (default: sqrt(k))
                - kmeans_iters: K-means iterations (default: 20)
                - use_minibatch: Use minibatch K-means (default: False)
                - batch_size: In batch mode, query in chunks of this size (set in config.yml; None = all at once).
                - dataset_name: 数据集名称，用于聚类缓存 key（默认 "unknown"）
                - cache_cluster: 是否启用聚类缓存（默认 True）
        """
        self._metric = metric
        self._n_lists = method_param.get("n_lists", None)  # Will be set in fit()
        self._kmeans_iters = method_param.get("kmeans_iters", 20)
        self._use_minibatch = method_param.get("use_minibatch", False)
        self._batch_size = method_param.get("batch_size", None)
        print(f"[IVFTensor.__init__] method_param={method_param}", flush=True)
        print(f"[IVFTensor.__init__] batch_size={self._batch_size}", flush=True)

        self._fp16_coarse = method_param.get("fp16_coarse", True)
        self._fine_strategy = method_param.get("fine_strategy", "cpu_fp32")  # gpu_fp32 / gpu_fp16 / cpu_fp32
        if self._fine_strategy == "cpu_fp32":
            self._use_interleaved = False
            self._schedule_strategy = "resident"  # cpu_fp32 不支持其他策略
        else:
            self._use_interleaved = True
            self._schedule_strategy = "unique"  # 'unique', 'resident', 'cache'

        self._n_probes = 1  # Default, will be set via set_query_arguments

        # 聚类缓存
        self._dataset_name = method_param.get("dataset_name", "unknown")
        self._cache_cluster = method_param.get("cache_cluster", True)
        self._cluster_cache = ClusterCache(_CLUSTER_CACHE_DIR)

        # Internal state
        self._n_vectors = None
        self._vector_dim = None
        self._centroids = None
        self._cluster_info = None
        self._cluster_vectors_flat = None
        self._reordered_indices = None
        self._vector_l2_norm = None
        self._batch_results = []

        # 仅在 cache miss 路径下持有，以免 pinned 数据在 Python 侧提前释放
        self._pinned_data = None

        # IVFTensor 索引和数据集对象
        self._ivf_dataset = None
        self._ivf_searcher = PyIVFTensor.IVFSearcher()
        self._query_workspace = ReusablePinnedBuffer()

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the IVF-Tensor index to the data.

        This performs K-means clustering and reorders the data by cluster.

        Args:
            X: Training data array of shape (n_samples, n_features)
        """
        log_memory("fit start")
        data_size_mb = X.nbytes / 1024 / 1024
        print(f"[IVFTensor.fit] Input data size: {data_size_mb:.1f} MB, shape: {X.shape}", flush=True)

        if not X.flags["C_CONTIGUOUS"]:
            raise RuntimeError("Error! [索引构建] 输入数据不是C-contiguous布局")
        if X.dtype != np.float32:
            raise RuntimeError(f"Error! [索引构建] 输入数据类型错误: {X.dtype}, 期望float32")

        print("[IVFTensor.fit] 数据布局检查通过，C-contiguous float32", flush=True)
        self._n_vectors, self._vector_dim = X.shape

        if self._n_lists is None:
            if self._n_vectors > 1_000_000:
                self._n_lists = int(np.sqrt(self._n_vectors))
            else:
                self._n_lists = max(1, self._n_vectors // 10000)

        print(
            f"Building IVF-Tensor index: {self._n_vectors} vectors, {self._vector_dim} dims, {self._n_lists} clusters",
            flush=True,
        )
        self._fit_cuda(X)

    def _extract_index_payload(self) -> None:
        """从 C++ dataset 提取查询所需 payload。"""
        (
            reordered_data,
            reordered_indices,
            centroids,
            cluster_offsets,
            cluster_counts,
            n_clusters,
            vector_l2_norm,
        ) = self._ivf_dataset.get_data()

        self._reordered_indices = np.asarray(reordered_indices, dtype=np.int32)
        self._centroids = np.asarray(centroids, dtype=np.float32)

        self._cluster_info = {
            "k": int(n_clusters),
            "offsets": np.asarray(cluster_offsets, dtype=np.int32),
            "counts": np.asarray(cluster_counts, dtype=np.int32),
            "reordered_indices": self._reordered_indices,
        }

        try:
            self._cluster_vectors_flat = reordered_data.reshape(-1)
        except Exception:
            print(
                "[IVFTensor] Warning: reordered_data.reshape(-1) failed; "
                "falling back to np.ascontiguousarray(...).reshape(-1), which may copy full dataset.",
                flush=True,
            )
            self._cluster_vectors_flat = np.ascontiguousarray(reordered_data).reshape(-1)

        if vector_l2_norm is not None and getattr(vector_l2_norm, "size", 0) > 0:
            self._vector_l2_norm = np.asarray(vector_l2_norm, dtype=np.float32)
        else:
            self._vector_l2_norm = None

        print(
            f"Cluster sizes: min={cluster_counts.min()} max={cluster_counts.max()} "
            f"mean={cluster_counts.mean():.1f} empty={int((cluster_counts == 0).sum())}",
            flush=True,
        )

    def _fit_cuda(self, X: np.ndarray) -> None:
        """
        Fit using CUDA implementation.
        命中缓存时加载聚类决策（centroids + reordered_indices），
        用原始数据 X 重新完成重排和 interleaved 构建，跳过 K-means。
        """
        log_memory("_fit_cuda start")

        if self._metric == "angular":
            distance_mode = PyIVFTensor.DISTANCE_COSINE
        elif self._metric == "euclidean":
            distance_mode = PyIVFTensor.DISTANCE_L2
        else:
            raise ValueError(f"Invalid metric: {self._metric}")

        use_hierarchical = False

        algo_tag = (
            f"{'hierarchical' if use_hierarchical else 'kmeans-gpu'}"
            f"_iters{self._kmeans_iters}"
            f"_{'minibatch' if self._use_minibatch else 'full'}"
        )

        cache_enabled = self._cache_cluster and self._dataset_name != "unknown"
        if self._cache_cluster and self._dataset_name == "unknown":
            print("[ClusterCache] 警告: dataset_name 未设置，跳过缓存（传入 dataset_name 参数以启用）", flush=True)

        cached = self._cluster_cache.load(self._dataset_name, self._n_lists, algo_tag) if cache_enabled else None

        self._ivf_dataset = PyIVFTensor.ClusterDataset()
        log_memory("after creating ClusterDataset")

        if cached is not None:
            (centroids, reordered_indices, cluster_counts, cluster_offsets), _ = cached
            del cached
            log_memory("after loading cache")

            log_memory("before register_array_as_pinned")
            register_array_as_pinned(X)
            log_memory("after register_array_as_pinned")

            self._ivf_dataset.init_from_existing_inplace_array(
                X,
                reordered_indices=reordered_indices.astype(np.int32, copy=False),
                centroids=centroids.astype(np.float32, copy=False),
                cluster_offsets=cluster_offsets.astype(np.int64, copy=False),
                cluster_counts=cluster_counts.astype(np.int32, copy=False),
                use_interleaved=self._use_interleaved,
            )
            log_memory("after init_from_existing_inplace_array")

            del reordered_indices, cluster_offsets, cluster_counts
            force_release_memory()
            drop_system_cache()
        else:
            t0 = time.time()
            log_memory("before numpy_to_pinned")
            self._pinned_data = numpy_to_pinned(X)
            log_memory("after numpy_to_pinned (pinned_data created)")

            if use_hierarchical:
                print(f"Running Balanced Hierarchical Clustering ({self._kmeans_iters} iterations)...", flush=True)
                self._ivf_dataset.init_with_hierarchical(
                    X,
                    n_clusters=self._n_lists,
                    kmeans_iters=self._kmeans_iters,
                    use_minibatch=self._use_minibatch,
                    distance_mode=distance_mode,
                )
            else:
                print(f"Running GPU K-means clustering ({self._kmeans_iters} iterations)...", flush=True)
                self._ivf_dataset.init_with_kmeans_pinned(
                    self._pinned_data,
                    n_clusters=self._n_lists,
                    kmeans_iters=self._kmeans_iters,
                    use_minibatch=self._use_minibatch,
                    distance_mode=distance_mode,
                    use_interleaved=self._use_interleaved,
                )
            log_memory("after init_with_kmeans_pinned")
            elapsed = time.time() - t0
            print(f"Clustering done in {elapsed:.1f}s", flush=True)

            (_, reordered_indices, centroids, cluster_offsets, cluster_counts, _, _) = self._ivf_dataset.get_data()

            if cache_enabled:
                self._cluster_cache.save(
                    self._dataset_name,
                    self._n_lists,
                    algo_tag,
                    centroids=centroids,
                    reordered_indices=reordered_indices,
                    cluster_counts=cluster_counts,
                    cluster_offsets=cluster_offsets,
                    meta={
                        "kmeans_iters": self._kmeans_iters,
                        "use_minibatch": self._use_minibatch,
                        "metric": self._metric,
                        "elapsed_sec": round(elapsed, 2),
                        "cluster_size_min": int(cluster_counts.min()),
                        "cluster_size_max": int(cluster_counts.max()),
                        "cluster_size_mean": round(float(cluster_counts.mean()), 1),
                    },
                )

        self._extract_index_payload()

    def set_query_arguments(self, n_probes: int) -> None:
        """
        Set query arguments.

        Args:
            n_probes: Number of clusters to probe during search
        """
        self._n_probes = int(n_probes)
        if self._n_probes < 1:
            self._n_probes = 1
        elif self._n_probes > self._n_lists:
            self._n_probes = self._n_lists
            print(f"Warning: n_probes ({n_probes}) > n_lists ({self._n_lists}), setting to {self._n_lists}", flush=True)

    def query(self, v: np.ndarray, k: int) -> List[int]:
        """
        Perform a single query.

        Args:
            v: Query vector
            k: Number of nearest neighbors to return

        Returns:
            List of indices of nearest neighbors
        """
        if self._cluster_vectors_flat is None:
            raise RuntimeError("Index not fitted. Call fit() first.")

        v = np.asarray(v)
        if not v.flags["C_CONTIGUOUS"]:
            raise RuntimeError("Error! [单条查询] 查询向量不是C-contiguous布局")
        if v.dtype != np.float32:
            raise RuntimeError(f"Error! [单条查询] 查询向量数据类型错误: {v.dtype}, 期望float32")

        self.batch_query(v.reshape(1, -1), k)
        results = self.get_batch_results()
        return results[0] if results else []

    def batch_query(self, X: np.ndarray, k: int) -> None:
        """
        Perform batch queries. Batch size is passed to ivftensor (query_batch_size);
        ivftensor does internal batching in C++.

        Args:
            X: Query vectors array of shape (n_queries, n_features)
            k: Number of nearest neighbors to return per query
        """
        if self._cluster_vectors_flat is None:
            raise RuntimeError("Index not fitted. Call fit() first.")

        if not X.flags["C_CONTIGUOUS"]:
            raise RuntimeError("Error! [批量查询] 查询数据不是C-contiguous布局")
        if X.dtype != np.float32:
            raise RuntimeError(f"Error! [批量查询] 查询数据类型错误: {X.dtype}, 期望float32")

        self._batch_results = self._batch_query_cuda(X, k)

    def _batch_query_cuda(self, X: np.ndarray, k: int) -> List[List[int]]:
        """
        Batch query using CUDA implementation.
        """
        if self._ivf_searcher is None or self._ivf_dataset is None:
            raise RuntimeError("Index not initialized. Call fit() first.")
        if self._schedule_strategy in ["unique", "cache"] and not self._use_interleaved:
            raise ValueError("schedule_strategy 'unique' or 'cache' requires use_interleaved=True")

        if self._metric == "angular":
            distance_mode = PyIVFTensor.DISTANCE_COSINE
        elif self._metric == "euclidean":
            distance_mode = PyIVFTensor.DISTANCE_L2
        else:
            raise ValueError(f"Invalid metric: {self._metric}")

        cluster_vectors_flat = self._cluster_vectors_flat
        cluster_sizes = self._cluster_info["counts"]
        reordered_indices_flat = self._cluster_info["reordered_indices"]

        empty_cluster_cnt = int(np.sum(cluster_sizes == 0))
        print("=" * 103, flush=True)
        print(f"Empty cluster count: {empty_cluster_cnt}", flush=True)
        print("=" * 103, flush=True)

        pinned_queries = self._query_workspace.stage(X)

        indices, distances = self._ivf_searcher.search_pinned_queries(
            pinned_queries,
            cluster_sizes,
            cluster_vectors_flat,
            self._centroids,
            n_probes=self._n_probes,
            k=k,
            distance_mode=distance_mode,
            reordered_indices=reordered_indices_flat,
            query_batch_size=self._batch_size,
            fp16_coarse=self._fp16_coarse,
            fine_strategy=self._fine_strategy,
            use_interleaved=self._use_interleaved,
            schedule_strategy=self._schedule_strategy,
            dataset_name=self._dataset_name,
            vector_l2_norm=self._vector_l2_norm if self._vector_l2_norm is not None and self._vector_l2_norm.size > 0 else None,
        )

        duplicate_results_cnt = 0
        for i in range(len(indices)):
            if len(indices[i]) != len(set(indices[i])):
                print(f"Query {i} has duplicate results: {indices[i]}", flush=True)
                dups = [indices[i][j] for j in range(len(indices[i])) if indices[i].count(indices[i][j]) > 1]
                print(f"Duplicate results: {dups}", flush=True)
                duplicate_results_cnt += 1
                if duplicate_results_cnt >= 5:
                    break

        return indices.tolist()

    def get_batch_results(self) -> List[List[int]]:
        """Get batch query results."""
        return self._batch_results if hasattr(self, "_batch_results") else []

    def set_batch_size(self, batch_size: Optional[int]) -> None:
        """Optional override of internal batch size (primary source is config.yml)."""
        self._batch_size = batch_size

    def reset_query_workspace(self) -> None:
        """重置 pinned query workspace。"""
        self._query_workspace = ReusablePinnedBuffer()

    def __str__(self) -> str:
        s = f"IVFTensor(n_lists={self._n_lists}, n_probes={self._n_probes}, metric={self._metric}"
        if self._batch_size is not None:
            s += f", batch_size={self._batch_size}"
        if self._fp16_coarse or self._fine_strategy != "gpu_fp32":
            s += f", fp16_coarse={self._fp16_coarse}, fine_strategy={self._fine_strategy}"
        if self._schedule_strategy != "resident":
            s += f", schedule_strategy={self._schedule_strategy}"
        return s + ")"