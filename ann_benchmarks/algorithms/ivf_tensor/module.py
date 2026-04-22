"""
IVF-Tensor: GPU-accelerated IVF-Flat search using CUDA (simplified)

使用 PyIVFTensor 最新接口的简化版本：
- ClusterDataset 直接传给 IVFSearcher，无需手动提取 cluster_vectors
- 统一使用 PinnedDataset 作为主数据路径
"""

import os
import sys
import time
from typing import Optional, List

import numpy as np

from ..base.module import BaseANN

# 加载 PyIVFTensor
ivftensor_path = "/home/diy/lzx/ivftensor"
sys.path.insert(0, os.path.join(ivftensor_path, "python/build"))
sys.path.insert(0, os.path.join(ivftensor_path, "python"))
import PyIVFTensor

from cluster_cache import ClusterCache
from .ivf_tensor_pinned import (
    load_to_pinned,
    ReusablePinnedBuffer,
)

_CLUSTER_CACHE_DIR = os.environ.get(
    "IVF_CLUSTER_CACHE_DIR",
    os.path.join("/home/diy/lzx/ann-benchmarks", "data", "cluster_cache"),
)


class IVFTensor(BaseANN):
    """简化版 IVF-Tensor：使用 PyIVFTensor 最新接口。"""

    def __init__(self, metric: str, method_param: dict):
        self._metric = metric
        self._n_lists = method_param.get("n_lists", None)
        self._kmeans_iters = method_param.get("kmeans_iters", 20)
        self._use_minibatch = method_param.get("use_minibatch", False)
        self._batch_size = method_param.get("batch_size", None)

        # 策略配置
        self._fine_strategy = method_param.get("fine_strategy", "cpu_fp32")
        if self._fine_strategy == "cpu_fp32":
            self._use_interleaved = False
            self._schedule_strategy = "resident"
        else:
            self._use_interleaved = True
            self._schedule_strategy = "unique"

        self._fp16_coarse = method_param.get("fp16_coarse", True)
        self._n_probes = 1

        # 缓存
        self._dataset_name = method_param.get("dataset_name", "unknown")
        self._cache_cluster = method_param.get("cache_cluster", True)
        self._cluster_cache = ClusterCache(_CLUSTER_CACHE_DIR)

        # 核心对象（简化：只保留必要引用）
        self._ivf_dataset: Optional[PyIVFTensor.ClusterDataset] = None
        self._pinned_data: Optional[PyIVFTensor.PinnedDataset] = None
        self._ivf_searcher = PyIVFTensor.IVFSearcher()
        self._query_workspace = ReusablePinnedBuffer()

        # 轻量元数据（不从 get_data() 提取整库数据）
        self._n_vectors = None
        self._vector_dim = None

    def fit(self, X: np.ndarray) -> None:
        """兼容接口：从 numpy 数组构建（会复制到 pinned，非 1x 主路径）。"""
        from .ivf_tensor_pinned import numpy_to_pinned

        pinned = numpy_to_pinned(X)
        self.fit_pinned(pinned)

    def fit_file(self, filepath: str) -> None:
        """推荐主路径：从文件直接加载到 pinned memory 并聚类。"""
        pinned = load_to_pinned(filepath)
        self.fit_pinned(pinned)

    def fit_pinned(self, pinned_data: PyIVFTensor.PinnedDataset) -> None:
        """
        核心构建逻辑：基于 PinnedDataset 执行聚类或缓存恢复。
        这是唯一主路径，确保 1x dataset 内存峰值。
        """
        self._n_vectors, self._vector_dim = pinned_data.n(), pinned_data.dim()

        if self._n_lists is None:
            self._n_lists = int(np.sqrt(self._n_vectors)) if self._n_vectors > 1_000_000 else max(1, self._n_vectors // 10000)

        print(f"[IVFTensor] Building index: {self._n_vectors} vectors, {self._vector_dim} dims, {self._n_lists} clusters", flush=True)

        # 尝试缓存
        algo_tag = f"kmeans-gpu_iters{self._kmeans_iters}_{'minibatch' if self._use_minibatch else 'full'}"
        cached = None
        if self._cache_cluster and self._dataset_name != "unknown":
            cached = self._cluster_cache.load(self._dataset_name, self._n_lists, algo_tag)
        elif self._cache_cluster:
            print("[ClusterCache] dataset_name 未设置，跳过缓存", flush=True)

        distance_mode = PyIVFTensor.DISTANCE_COSINE if self._metric == "angular" else PyIVFTensor.DISTANCE_L2

        self._ivf_dataset = PyIVFTensor.ClusterDataset()
        self._pinned_data = pinned_data  # 保持引用，防止 GC

        if cached is not None:
            # Cache hit: 原地重排
            print("[IVFTensor] Cache hit, in-place reordering...", flush=True)
            (centroids, reordered_indices, cluster_counts, cluster_offsets), _ = cached

            self._ivf_dataset.init_from_existing_inplace(
                pinned_data,
                reordered_indices=reordered_indices.astype(np.int32, copy=False),
                centroids=centroids.astype(np.float32, copy=False),
                cluster_offsets=cluster_offsets.astype(np.int64, copy=False),
                cluster_counts=cluster_counts.astype(np.int32, copy=False),
                use_interleaved=self._use_interleaved,
            )
        else:
            # Cache miss: K-means 聚类
            print(f"[IVFTensor] Running K-means ({self._kmeans_iters} iters)...", flush=True)
            t0 = time.time()
            self._ivf_dataset.init_with_kmeans_pinned(
                pinned_data,
                n_clusters=self._n_lists,
                kmeans_iters=self._kmeans_iters,
                use_minibatch=self._use_minibatch,
                distance_mode=distance_mode,
                use_interleaved=self._use_interleaved,
            )
            print(f"[IVFTensor] Clustering done in {time.time() - t0:.1f}s", flush=True)

            # 保存缓存（只存元数据）
            if self._cache_cluster and self._dataset_name != "unknown":
                _, reordered_indices, centroids, offsets, counts, _, _ = self._ivf_dataset.get_data()
                self._cluster_cache.save(
                    self._dataset_name, self._n_lists, algo_tag,
                    centroids=centroids,
                    reordered_indices=reordered_indices,
                    cluster_counts=counts,
                    cluster_offsets=offsets,
                )

        # 打印 cluster 统计
        _, _, _, _, counts, _, _ = self._ivf_dataset.get_data()
        print(f"[IVFTensor] Cluster sizes: min={counts.min()} max={counts.max()} mean={counts.mean():.1f}", flush=True)

    def set_query_arguments(self, n_probes: int) -> None:
        self._n_probes = min(int(n_probes), self._n_lists)

    def batch_query(self, X: np.ndarray, k: int) -> None:
        """批量查询：使用新的 search_pinned 接口，直接传 dataset。"""
        if self._ivf_dataset is None:
            raise RuntimeError("Index not fitted")

        if not X.flags["C_CONTIGUOUS"] or X.dtype != np.float32:
            raise RuntimeError("Queries must be C-contiguous float32")

        # 复用或创建 query pinned buffer
        pinned_queries = self._query_workspace.stage(X)

        distance_mode = PyIVFTensor.DISTANCE_COSINE if self._metric == "angular" else PyIVFTensor.DISTANCE_L2

        # 使用新的简化接口：直接传 dataset，不传 cluster_vectors_flat
        indices, distances = self._ivf_searcher.search_pinned(
            pinned_queries=pinned_queries,
            dataset=self._ivf_dataset,  # 直接传 ClusterDataset，简化！
            n_probes=self._n_probes,
            k=k,
            distance_mode=distance_mode,
            query_batch_size=self._batch_size or 0,
            fp16_coarse=self._fp16_coarse,
            fine_strategy=self._fine_strategy,
            schedule_strategy=self._schedule_strategy,
            dataset_name=self._dataset_name,
        )

        self._batch_results = indices.tolist()

    def query(self, v: np.ndarray, k: int) -> List[int]:
        """单条查询。"""
        self.batch_query(v.reshape(1, -1), k)
        return self._batch_results[0] if self._batch_results else []

    def get_batch_results(self) -> List[List[int]]:
        return self._batch_results

    def set_batch_size(self, batch_size: Optional[int]) -> None:
        self._batch_size = batch_size

    def __str__(self) -> str:
        return f"IVFTensor(n_lists={self._n_lists}, n_probes={self._n_probes}, metric={self._metric})"
