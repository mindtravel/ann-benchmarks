"""
IVF-Tensor: GPU-accelerated IVF-Flat search using CUDA

This module wraps the CUDA implementation from ivftensor for use in ann-benchmarks.
It uses pybind11 Python bindings to call the compiled CUDA library functions.
"""

import os
import sys
import numpy as np
from typing import Optional, List

from ..base.module import BaseANN

# 尝试加载 IVFTensor Python 扩展模块
ivftensor_path = "/home/diy/lzx/ivftensor"
project_path = "/home/diy/lzx/ann-benchmarks"

module_paths = os.path.join(ivftensor_path, "python/build")
sys.path.insert(0, module_paths)
import PyIVFTensor

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
        """
        self._metric = metric
        self._n_lists = method_param.get('n_lists', None)  # Will be set in fit()
        self._kmeans_iters = method_param.get('kmeans_iters', 20)
        self._use_minibatch = method_param.get('use_minibatch', False)
        self._batch_size = method_param.get('batch_size', None)  # 在 config.yml 的 arg_groups 中配置
        self._use_blocks = method_param.get('use_blocks', False)  # BCS 平衡 block 模式
        self._std_var_ratio = method_param.get('std_var_ratio', 0.2)
        # use_fp16: 粗筛用 fp16（量化在 host）；可由 config 或环境变量 IVF_TENSOR_USE_FP16=1 启用
        self._use_fp16 = method_param.get('use_fp16', False) or (os.environ.get('IVF_TENSOR_USE_FP16', '0') == '1')
        self._n_probes = 1  # Default, will be set via set_query_arguments
        
        # Internal state
        self._dataset = None
        self._n_vectors = None
        self._vector_dim = None
        self._centroids = None
        self._cluster_info = None
        self._reordered_data = None
        
        # IVFTensor 索引和数据集对象
        self._ivf_dataset = None
        self._ivf_searcher = None
    
    def fit(self, X: np.ndarray) -> None:
        """
        Fit the IVF-Tensor index to the data.
        
        This performs K-means clustering and reorders the data by cluster.
        
        Args:
            X: Training data array of shape (n_samples, n_features)
        """
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
        self._fit_cuda(X)
    
    def _fit_cuda(self, X: np.ndarray) -> None:
        """
        Fit using CUDA implementation.
        
        This performs K-means clustering on GPU and loads the dataset.
        """
        
        # 创建 ClusterDataset 对象
        self._ivf_dataset = PyIVFTensor.ClusterDataset()
        
        # 确定距离模式
        if(self._metric == "angular"):
            distance_mode = PyIVFTensor.DISTANCE_COSINE
        elif(self._metric == "euclidean"):
            distance_mode = PyIVFTensor.DISTANCE_L2
        else:
            raise ValueError(f"Invalid metric: {self._metric}")
        
        # 初始化数据集（使用 K-means 聚类）
        print(f"Running GPU K-means clustering ({self._kmeans_iters} iterations)...")
        self._ivf_dataset.init_with_kmeans(
            X,
            n_clusters=self._n_lists,
            kmeans_iters=self._kmeans_iters,
            use_minibatch=self._use_minibatch,
            distance_mode=distance_mode,
        )
        # 获取聚类结果（用于后续搜索）
        (self._reordered_data, self._reordered_indices, self._centroids, 
         cluster_offsets, cluster_counts, n_clusters) = self._ivf_dataset.get_data()
        
        # 保存聚类信息（用于索引转换）
        self._cluster_info = {
            'k': n_clusters,
            'offsets': cluster_offsets.astype(np.int32),  # 转换为 int32（注意：原始是 long long）
            'counts': cluster_counts.astype(np.int32),
            'reordered_indices': self._reordered_indices.astype(np.int32)
        }
        
        # 创建搜索器
        self._ivf_searcher = PyIVFTensor.IVFSearcher()
        # INSERT_YOUR_CODE
        def plot_cluster_counts_distribution(cluster_counts, save_path):
            """
            绘制cluster_counts的分布图，并保存到指定路径。
            """
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print("matplotlib未安装，无法绘图(跳过)。")
                return
            plt.figure(figsize=(8, 4))
            plt.hist(cluster_counts, bins=50, color='skyblue', edgecolor='black')
            plt.title("Cluster Counts Distribution")
            plt.xlabel("Number of vectors in cluster")
            plt.ylabel("Number of clusters")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        if not os.path.exists(os.path.join(project_path, "kmeans_distribution")):
            os.makedirs(os.path.join(project_path, "kmeans_distribution"))
        plot_cluster_counts_distribution(cluster_counts, os.path.join(project_path, "kmeans_distribution/ivf_tensor_n_lists={self._n_lists}.png"))
        # print(f"GPU K-means completed. Cluster sizes: min={cluster_counts.min()}, max={cluster_counts.max()}, mean={cluster_counts.mean():.1f}")
    
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
        
        # 准备数据（展平）
        cluster_vectors_flat = reordered_data.flatten()
        cluster_centers_flat = centroids.flatten()
        cluster_sizes = cluster_counts.astype(np.int32)
        reordered_indices_flat = reordered_indices.astype(np.int32)

        # BCS 平衡 block 模式：内部 rebalance + block lookup
        indices, distances = self._ivf_searcher.search_with_blocks(
            X,
            cluster_sizes,
            cluster_vectors_flat,
            cluster_centers_flat,
            n_probes=self._n_probes,
            k=k,
            distance_mode=distance_mode,
            reordered_indices=reordered_indices_flat,
            std_var_ratio=self._std_var_ratio,
            use_fp16=self._use_fp16
        )

        
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
        if self._use_fp16:
            s += ", use_fp16=True"
        return s + ")"

