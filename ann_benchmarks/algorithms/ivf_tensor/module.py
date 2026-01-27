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

# 可能的模块路径
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
        """
        self._metric = metric
        self._n_lists = method_param.get('n_lists', None)  # Will be set in fit()
        self._kmeans_iters = method_param.get('kmeans_iters', 20)
        self._use_minibatch = method_param.get('use_minibatch', False)
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
        Perform batch queries.
        
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
        
        # 执行 GPU 搜索（带回表）
        # reordered_indices 会被传递到 GPU 进行回表操作，返回的 indices 已经是原始索引
        indices, distances = self._ivf_searcher.search(
            X,
            cluster_counts.astype(np.int32),  # cluster sizes
            cluster_vectors_flat,              # cluster vectors (flattened)
            cluster_centers_flat,              # cluster centers (flattened)
            n_probes=self._n_probes,
            k=k,
            distance_mode=distance_mode,
            reordered_indices=reordered_indices.astype(np.int32)  # 传入回表映射数组
        )
        # print("distances", np.sqrt(distances[0][:k]))
        # print("indices", indices[0][:k])  # indices 是整数数组，不需要 sqrt

        # if(indices.shape[0] != X.shape[0]):
        #     raise ValueError(f"Invalid indices shape: {indices.shape}")
        # if(indices.shape[1] != k):
        #     raise ValueError(f"Invalid indices shape: {indices.shape}")
        
        # 检查每一行 indices 是否有重复索引，如果有则输出（打印）重复信息
        # for i, row in enumerate(indices):
        #     unique_count = len(np.unique(row))
        #     if unique_count != len(row):
        #         duplicates = set([x for x in row if list(row).count(x) > 1])
        #         raise ValueError(f"Duplicate indices found in query {i}: {duplicates}. Row: {row}")
        results = indices.tolist()
        
        return results
    
    def get_batch_results(self) -> List[List[int]]:
        """Get batch query results."""
        return self._batch_results if hasattr(self, '_batch_results') else []
    
    def __str__(self) -> str:
        return f"IVFTensor(n_lists={self._n_lists}, n_probes={self._n_probes}, metric={self._metric})"

