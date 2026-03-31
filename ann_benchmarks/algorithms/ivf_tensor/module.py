"""
IVF-Tensor: GPU-accelerated IVF-Flat search using CUDA

This module wraps the CUDA implementation from ivftensor for use in ann-benchmarks.
It uses pybind11 Python bindings to call the compiled CUDA library functions.
"""

import json
import os
import sys
import time
import numpy as np
from typing import Optional, List

from ..base.module import BaseANN

# 尝试加载 IVFTensor Python 扩展模块
ivftensor_path = "/home/diy/lzx/ivftensor"
project_path = "/home/diy/lzx/ann-benchmarks"

module_paths = os.path.join(ivftensor_path, "python/build")
sys.path.insert(0, module_paths)
sys.path.insert(0, os.path.join(ivftensor_path, "python"))
import PyIVFTensor
from cluster_cache import ClusterCache

# 默认缓存目录，可通过环境变量 IVF_CLUSTER_CACHE_DIR 覆盖
_CLUSTER_CACHE_DIR = os.environ.get(
    "IVF_CLUSTER_CACHE_DIR",
    os.path.join(project_path, "data", "cluster_cache")
)

class IVFTensor(BaseANN):
    """
    IVF-Tensor: GPU-accelerated IVF-Flat search
    
    This implementation uses CUDA for both clustering (K-means) and search operations.
    """
    
    def __init__(self, metric: str, method_param: dict):
        # print("init")
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
        # self._metric = metric
        self._metric = "euclidean"
        self._n_lists = method_param.get('n_lists', None)  # Will be set in fit()
        self._kmeans_iters = method_param.get('kmeans_iters', 20)
        self._use_minibatch = method_param.get('use_minibatch', False)
        self._batch_size = method_param.get('batch_size', None)  # 在 config.yml 的 arg_groups 中配置
        self._use_blocks = method_param.get('use_blocks', False)  # BCS 平衡 block 模式
        self._std_var_ratio = method_param.get('std_var_ratio', 0.2)
        self._fp16_coarse = method_param.get('fp16_coarse', False)
        self._fine_strategy = method_param.get('fine_strategy', 'cpu_fp32')  # gpu_fp32 / gpu_fp16 / cpu_fp32
        if(self._fine_strategy == 'cpu_fp32'):
            self._use_interleaved = False
            self._lazy_upload_vectors = False
        else:
            self._use_interleaved = True
            self._lazy_upload_vectors = method_param.get('lazy_upload_vectors', True)
        self._n_probes = 1  # Default, will be set via set_query_arguments

        # 聚类缓存
        self._dataset_name = method_param.get('dataset_name', 'unknown')
        self._cache_cluster = method_param.get('cache_cluster', True)
        self._cluster_cache = ClusterCache(_CLUSTER_CACHE_DIR)
        
        # Internal state
        self._dataset = None
        self._n_vectors = None
        self._vector_dim = None
        self._centroids = None
        self._cluster_info = None
        self._reordered_data = None  # C++ 侧已按 use_interleaved 存储，1D 或 2D

        # IVFTensor 索引和数据集对象
        self._ivf_dataset = None
        self._ivf_searcher = None
    
    def fit(self, X: np.ndarray) -> None:
        # print("here")
        
        """
        Fit the IVF-Tensor index to the data.
        
        This performs K-means clustering and reorders the data by cluster.
        
        Args:
            X: Training data array of shape (n_samples, n_features)
        """
        self._dataset = np.ascontiguousarray(X, dtype=np.float32)
        self._n_vectors, self._vector_dim = X.shape
        # Determine number of clusters if not specified
        if self._n_lists is None:
            # Default: sqrt(k) or k/1000 for smaller datasets
            if self._n_vectors > 1_000_000:
                self._n_lists = int(np.sqrt(self._n_vectors))
            else:
                self._n_lists = max(1, self._n_vectors // 10000)
        
        
        print(f"Building IVF-Tensor index: {self._n_vectors} vectors, {self._vector_dim} dims, {self._n_lists} clusters")
        # Use CUDA implementation
        self._fit_cuda(self._dataset)

    def _fit_cuda(self, X: np.ndarray) -> None:
        """
        Fit using CUDA implementation.
        命中缓存时加载聚类决策（centroids + reordered_indices），
        用原始数据 X 重新完成重排和 interleaved 构建，跳过 K-means。
        """
        if self._metric == "angular":
            distance_mode = PyIVFTensor.DISTANCE_COSINE
        elif self._metric == "euclidean":
            distance_mode = PyIVFTensor.DISTANCE_L2
        else:
            raise ValueError(f"Invalid metric: {self._metric}")

        use_hierarchical = False  # 切换聚类算法时改这里

        algo_tag = (
            f"{'hierarchical' if use_hierarchical else 'kmeans-gpu'}"
            f"_iters{self._kmeans_iters}"
            f"_{'minibatch' if self._use_minibatch else 'full'}"
        )

        # dataset_name 未设置时禁用缓存，避免不同数据集互相污染
        cache_enabled = self._cache_cluster and self._dataset_name != 'unknown'
        if self._cache_cluster and self._dataset_name == 'unknown':
            print("[ClusterCache] 警告: dataset_name 未设置，跳过缓存（传入 dataset_name 参数以启用）")

        # ---- 尝试从缓存加载聚类决策 ----
        cached = self._cluster_cache.load(self._dataset_name, self._n_lists, algo_tag) \
            if cache_enabled else None

        self._ivf_dataset = PyIVFTensor.ClusterDataset()

        if cached is not None:
            (centroids, reordered_indices, cluster_counts, cluster_offsets), _ = cached
            # 用缓存的聚类决策 + 原始数据 X 重建 ClusterDataset
            # init_from_existing 内部按 reordered_indices 重排，并按需构建 interleaved
            self._ivf_dataset.init_from_existing(
                reordered_data    = X,
                reordered_indices = reordered_indices.astype(np.int32),
                centroids         = centroids,
                cluster_offsets   = cluster_offsets.astype(np.int64),
                cluster_counts    = cluster_counts.astype(np.int32),
                use_interleaved   = self._use_interleaved,
            )
        else:
            # ---- 执行聚类 ----
            t0 = time.time()
            if use_hierarchical:
                print(f"Running Balanced Hierarchical Clustering ({self._kmeans_iters} iterations)...")
                self._ivf_dataset.init_with_hierarchical(
                    X,
                    n_clusters=self._n_lists,
                    kmeans_iters=self._kmeans_iters,
                    use_minibatch=self._use_minibatch,
                    distance_mode=distance_mode,
                )
            else:
                print(f"Running GPU K-means clustering ({self._kmeans_iters} iterations)...")
                self._ivf_dataset.init_with_kmeans(
                    X,
                    n_clusters=self._n_lists,
                    kmeans_iters=self._kmeans_iters,
                    use_minibatch=self._use_minibatch,
                    distance_mode=distance_mode,
                    use_interleaved=self._use_interleaved,
                )
            elapsed = time.time() - t0
            print(f"Clustering done in {elapsed:.1f}s")

            # 取聚类决策并写入缓存（只存 centroids + indices，不存向量数据）
            (_, reordered_indices, centroids,
             cluster_offsets, cluster_counts, _) = self._ivf_dataset.get_data()

            if cache_enabled:
                self._cluster_cache.save(
                    self._dataset_name, self._n_lists, algo_tag,
                    centroids         = centroids,
                    reordered_indices = reordered_indices,
                    cluster_counts    = cluster_counts,
                    cluster_offsets   = cluster_offsets,
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

        # 统一从 dataset 拿最终数据
        (self._reordered_data, self._reordered_indices, self._centroids,
         cluster_offsets, cluster_counts, n_clusters) = self._ivf_dataset.get_data()

        self._cluster_info = {
            'k': n_clusters,
            'offsets': cluster_offsets.astype(np.int32),
            'counts': cluster_counts.astype(np.int32),
            'reordered_indices': self._reordered_indices.astype(np.int32),
        }
        self._ivf_searcher = PyIVFTensor.IVFSearcher()
        print(f"Cluster sizes: min={cluster_counts.min()} max={cluster_counts.max()} "
              f"mean={cluster_counts.mean():.1f} empty={int((cluster_counts == 0).sum())}")

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
            print(f"Warning: n_probes ({n_probes}) > n_lists ({self._n_lists}), setting to {self._n_lists}")
    
    def query(self, v: np.ndarray, k: int) -> List[int]:
        """
        Perform a single query.
        
        Args:
            v: Query vector
            k: Number of nearest neighbors to return
            
        Returns:
            List of indices of nearest neighbors
        """
        if self._reordered_data is None:
            raise RuntimeError("Index not fitted. Call fit() first.")
        
        # Use batch query with single vector
        self.batch_query(np.array([v]), k)
        results = self.get_batch_results()
        return results[0] if results else []
    
    def batch_query(self, X: np.ndarray, k: int) -> None:
        """
        Perform batch queries. Batch size is passed to ivftensor (query_batch_size);
        ivftensor does internal batching in C++ (pipeline optimization can be added there).
        
        Args:
            X: Query vectors array of shape (n_queries, n_features)
            k: Number of nearest neighbors to return per query
        """
        if self._reordered_data is None:
            raise RuntimeError("Index not fitted. Call fit() first.")
        self._batch_results = self._batch_query_cuda(X, k)
    
    def _batch_query_cuda(self, X: np.ndarray, k: int) -> List[List[int]]:
        """
        Batch query using CUDA implementation.
        """
        if self._ivf_searcher is None or self._ivf_dataset is None:
            raise RuntimeError("Index not initialized. Call fit() first.")
        if self._lazy_upload_vectors and not self._use_interleaved:
            raise ValueError("lazy_upload_vectors requires use_interleaved=True")

        # 获取聚类数据
        (reordered_data, reordered_indices, centroids,
         cluster_offsets, cluster_counts, n_clusters) = self._ivf_dataset.get_data()

        # 确定距离模式
        if(self._metric == "angular"):
            distance_mode = PyIVFTensor.DISTANCE_COSINE
        elif(self._metric == "euclidean"):
            distance_mode = PyIVFTensor.DISTANCE_L2
        else:
            raise ValueError(f"Invalid metric: {self._metric}")

        # 准备 cluster_vectors：C++ 侧已按 use_interleaved 存储，interleaved 时为 1D，否则为 2D 需展平
        # 聚类中心必须保持 [n_clusters, n_dim] 二维，PyIVFTensor.search / ivf_search 按行主序读 centroids
        cluster_vectors_flat = reordered_data if reordered_data.ndim == 1 else reordered_data.flatten()
        cluster_sizes = cluster_counts.astype(np.int32)
        reordered_indices_flat = reordered_indices.astype(np.int32)

        # 统计cluster_sizes中为空的数量
        empty_cluster_cnt = sum(cluster_sizes == 0)
        print("=======================================================================================================")
        print(f"Empty cluster count: {empty_cluster_cnt}")
        n_total_vectors = int(cluster_sizes.sum())
        n_dim = X.shape[1]
        print("=======================================================================================================")

        indices, distances = self._ivf_searcher.search(
            X,
            cluster_sizes,
            cluster_vectors_flat,
            centroids,
            n_probes=self._n_probes,
            k=k,
            distance_mode=distance_mode,
            reordered_indices=reordered_indices_flat,
            query_batch_size=0,
            fp16_coarse=self._fp16_coarse,
            fine_strategy=self._fine_strategy,
            use_blocks=self._use_blocks,
            std_var_ratio=self._std_var_ratio,
            use_interleaved=self._use_interleaved,
            lazy_upload_vectors=self._lazy_upload_vectors,
        )

        # 输出每个query中重复的结果，最多5条
        duplicate_results_cnt = 0
        for i in range(len(indices)):
            if len(indices[i]) != len(set(indices[i])):
                print(f"Query {i} has duplicate results: {indices[i]}")
                dups = [indices[i][j] for j in range(len(indices[i])) if indices[i].count(indices[i][j]) > 1]
                print(f"Duplicate results: {dups}")
                duplicate_results_cnt += 1
                if duplicate_results_cnt >= 5:
                    break
        results = indices.tolist()
        return results
    
    def get_batch_results(self) -> List[List[int]]:
        """Get batch query results."""
        return self._batch_results if hasattr(self, '_batch_results') else []

    def set_batch_size(self, batch_size: Optional[int]) -> None:
        """Optional override of internal batch size (primary source is config.yml)."""
        self._batch_size = batch_size

    def __str__(self) -> str:
        s = f"IVFTensor(n_lists={self._n_lists}, n_probes={self._n_probes}, metric={self._metric}"
        if self._batch_size is not None:
            s += f", batch_size={self._batch_size}"
        if self._fp16_coarse or self._fine_strategy != 'gpu_fp32':
            s += f", fp16_coarse={self._fp16_coarse}, fine_strategy={self._fine_strategy}"
        if self._lazy_upload_vectors:
            s += ", lazy_upload_vectors=True"
        return s + ")"

