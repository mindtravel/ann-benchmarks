"""
IVF-Tensor Pinned Memory 零拷贝封装

提供基于 PyIVFTensor.PinnedDataset 的高级 API，避免 pageable → pinned 的内存拷贝。
"""

import os
import sys
from typing import Optional, Tuple, List

import numpy as np
import psutil

# 确保能找到 PyIVFTensor
_ivftensor_path = "/home/diy/lzx/ivftensor"
_module_paths = os.path.join(_ivftensor_path, "python/build")
sys.path.insert(0, _module_paths)
sys.path.insert(0, os.path.join(_ivftensor_path, "python"))
import PyIVFTensor


def _get_memory_mb():
    """获取当前进程的内存使用量（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def numpy_to_pinned(arr: np.ndarray) -> PyIVFTensor.PinnedDataset:
    """将 numpy 数组零拷贝包装为 PinnedDataset（写入已分配的 pinned memory）。

    Args:
        arr: C-contiguous float32 数组，shape [n, dim]

    Returns:
        PinnedDataset 对象，底层为 cudaMallocHost 分配的 pinned memory
    """
    if arr.dtype != np.float32:
        raise ValueError(f"Expected float32, got {arr.dtype}")
    if not arr.flags['C_CONTIGUOUS']:
        raise ValueError("Array must be C-contiguous")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D")

    n, dim = arr.shape
    data_size_mb = arr.nbytes / 1024 / 1024
    print(f"[numpy_to_pinned] Creating PinnedDataset for {n}x{dim} array ({data_size_mb:.1f} MB)", flush=True)
    mem_before = _get_memory_mb()
    
    pinned = PyIVFTensor.PinnedDataset(n, dim)
    mem_after_create = _get_memory_mb()
    print(f"[numpy_to_pinned] After cudaMallocHost: +{mem_after_create - mem_before:.1f} MB", flush=True)
    
    pinned.numpy()[:] = arr
    mem_after_copy = _get_memory_mb()
    print(f"[numpy_to_pinned] After copy to pinned: +{mem_after_copy - mem_after_create:.1f} MB", flush=True)
    
    return pinned


def fit_kmeans_pinned(
    X: np.ndarray,
    n_clusters: int,
    metric: str = "angular",
    kmeans_iters: int = 20,
    use_minibatch: bool = False,
    use_interleaved: bool = False,
    device_id: int = 0,
    seed: int = 1234,
) -> PyIVFTensor.ClusterDataset:
    """使用 PinnedDataset 零拷贝路径执行 K-means 聚类。

    Args:
        X: 训练数据 [n_vectors, dim]，float32，C-contiguous
        n_clusters: 聚类中心数量
        metric: "angular" 或 "euclidean"
        kmeans_iters: K-means 迭代次数
        use_minibatch: 是否使用 minibatch K-means
        use_interleaved: 是否生成 interleaved 布局
        device_id: GPU 设备 ID
        seed: 随机种子

    Returns:
        初始化好的 ClusterDataset 对象
    """
    distance_mode = (
        PyIVFTensor.DISTANCE_COSINE if metric == "angular" else PyIVFTensor.DISTANCE_L2
    )

    pinned = numpy_to_pinned(X)
    dataset = PyIVFTensor.ClusterDataset()
    dataset.init_with_kmeans_pinned(
        pinned,
        n_clusters=n_clusters,
        kmeans_iters=kmeans_iters,
        use_minibatch=use_minibatch,
        distance_mode=distance_mode,
        seed=seed,
        device_id=device_id,
        use_interleaved=use_interleaved,
    )
    return dataset


def search_pinned(
    queries: np.ndarray,
    cluster_sizes: np.ndarray,
    cluster_vectors: np.ndarray,
    cluster_centers: np.ndarray,
    n_probes: int,
    k: int,
    metric: str = "angular",
    reordered_indices: Optional[np.ndarray] = None,
    query_batch_size: int = 0,
    fp16_coarse: bool = False,
    fine_strategy: str = "gpu_fp32",
    use_interleaved: bool = False,
    schedule_strategy: str = "resident",
    dataset_name: str = "default",
    vector_l2_norm: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """使用 PinnedDataset 零拷贝路径执行 IVF 搜索。

    Args:
        queries: 查询向量 [n_query, dim]，float32，C-contiguous
        cluster_sizes: 每个 cluster 的向量数 [n_clusters]
        cluster_vectors: 重排后的向量数据（flattened 或 interleaved）
        cluster_centers: 聚类中心 [n_clusters, dim]
        n_probes: 搜索时探测的 cluster 数量
        k: 返回的最近邻数量
        metric: "angular" 或 "euclidean"
        reordered_indices: 重排索引 [n_total_vectors]（可选）
        query_batch_size: 查询批次大小，0 表示全量
        fp16_coarse: 粗筛是否使用 fp16
        fine_strategy: "gpu_fp32", "gpu_fp16", "cpu_fp32"
        use_interleaved: cluster_vectors 是否为 interleaved 布局
        schedule_strategy: "resident", "unique", "cache"
        dataset_name: 数据集名称标识
        vector_l2_norm: 预计算的 L2 norm 平方 [n_total_vectors]（可选）

    Returns:
        (indices, distances): indices [n_query, k], distances [n_query, k]
    """
    distance_mode = (
        PyIVFTensor.DISTANCE_COSINE if metric == "angular" else PyIVFTensor.DISTANCE_L2
    )

    pinned_queries = numpy_to_pinned(queries)
    searcher = PyIVFTensor.IVFSearcher()

    kwargs = {
        "n_probes": n_probes,
        "k": k,
        "distance_mode": distance_mode,
        "query_batch_size": query_batch_size,
        "fp16_coarse": fp16_coarse,
        "fine_strategy": fine_strategy,
        "use_interleaved": use_interleaved,
        "schedule_strategy": schedule_strategy,
        "dataset_name": dataset_name,
    }
    if reordered_indices is not None:
        kwargs["reordered_indices"] = reordered_indices.astype(np.int32)
    if vector_l2_norm is not None:
        kwargs["vector_l2_norm"] = vector_l2_norm.astype(np.float32)

    return searcher.search_pinned_queries(
        pinned_queries,
        cluster_sizes.astype(np.int32),
        cluster_vectors.astype(np.float32),
        cluster_centers.astype(np.float32),
        **kwargs,
    )


class PinnedIVFTensor:
    """基于 PinnedDataset 的 IVF-Tensor 封装，自动管理 pinned memory 生命周期。

    用法：
        ivf = PinnedIVFTensor(metric="angular", n_lists=100)
        ivf.fit(X)                       # X 自动转入 pinned memory
        indices, distances = ivf.search(Q, k=10)  # Q 自动转入 pinned memory
    """

    def __init__(
        self,
        metric: str = "angular",
        n_lists: int = 100,
        kmeans_iters: int = 20,
        use_minibatch: bool = False,
        use_interleaved: bool = False,
        device_id: int = 0,
    ):
        self._metric = metric
        self._n_lists = n_lists
        self._kmeans_iters = kmeans_iters
        self._use_minibatch = use_minibatch
        self._use_interleaved = use_interleaved
        self._device_id = device_id

        self._dataset: Optional[PyIVFTensor.ClusterDataset] = None
        self._cluster_info: Optional[dict] = None
        self._centroids: Optional[np.ndarray] = None
        self._cluster_vectors_flat: Optional[np.ndarray] = None
        self._reordered_indices: Optional[np.ndarray] = None
        self._vector_l2_norm: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> None:
        """执行 K-means 聚类并构建索引。"""
        self._dataset = fit_kmeans_pinned(
            X,
            n_clusters=self._n_lists,
            metric=self._metric,
            kmeans_iters=self._kmeans_iters,
            use_minibatch=self._use_minibatch,
            use_interleaved=self._use_interleaved,
            device_id=self._device_id,
        )

        # 提取聚类结果供搜索使用
        (
            reordered_data,
            reordered_indices,
            centroids,
            offsets,
            counts,
            n_clusters,
            vector_l2_norm,
        ) = self._dataset.get_data()

        self._cluster_info = {
            "k": n_clusters,
            "offsets": offsets.astype(np.int32),
            "counts": counts.astype(np.int32),
        }
        self._reordered_indices = reordered_indices.astype(np.int32)
        self._centroids = centroids.astype(np.float32)
        self._cluster_vectors_flat = reordered_data.flatten()
        if vector_l2_norm.size > 0:
            self._vector_l2_norm = vector_l2_norm.astype(np.float32)

    def search(
        self,
        queries: np.ndarray,
        k: int,
        n_probes: Optional[int] = None,
        query_batch_size: int = 0,
        fp16_coarse: bool = False,
        fine_strategy: str = "gpu_fp32",
        schedule_strategy: str = "resident",
        dataset_name: str = "default",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """执行搜索。"""
        if self._dataset is None:
            raise RuntimeError("Index not fitted. Call fit() first.")

        if n_probes is None:
            n_probes = self._n_lists

        return search_pinned(
            queries=queries,
            cluster_sizes=self._cluster_info["counts"],
            cluster_vectors=self._cluster_vectors_flat,
            cluster_centers=self._centroids,
            n_probes=n_probes,
            k=k,
            metric=self._metric,
            reordered_indices=self._reordered_indices,
            query_batch_size=query_batch_size,
            fp16_coarse=fp16_coarse,
            fine_strategy=fine_strategy,
            use_interleaved=self._use_interleaved,
            schedule_strategy=schedule_strategy,
            dataset_name=dataset_name,
            vector_l2_norm=self._vector_l2_norm,
        )
