"""
IVF-Tensor Pinned Memory 零拷贝封装

提供基于 PyIVFTensor.PinnedDataset 的高级 API，避免 pageable → pinned 的内存拷贝。

这一版的重构目标：
1. 保留从文件直接加载到 pinned memory 的主路径
2. 新增 fit_pinned()，允许上游直接传入 PinnedDataset
3. 查询侧引入可复用 pinned workspace，避免每次 search 都重新申请 pinned memory
4. 避免 reordered_data.flatten() 造成潜在整库复制；优先使用 reshape(-1) view
5. 复用 IVFSearcher，避免重复构造对象

注意：
- 由于当前 PyIVFTensor.search_pinned_queries 接口仍要求 Python 侧传入
  cluster_vectors / cluster_sizes / cluster_centers / reordered_indices，
  因此这一版仍无法彻底做到“Python 侧完全不持有 full-vector payload”。
- 若要进一步逼近真正 1x dataset 峰值，需要 C++ / pybind11 侧补：
  1) ClusterDataset.init_from_existing_pinned(...)
  2) IVFSearcher.search_with_dataset(...)
"""

import os
import sys
import ctypes
from typing import Optional, Tuple

import numpy as np
import psutil

# 确保能找到 PyIVFTensor
_ivftensor_path = "/home/diy/lzx/ivftensor"
_module_paths = os.path.join(_ivftensor_path, "python/build")
sys.path.insert(0, _module_paths)
sys.path.insert(0, os.path.join(_ivftensor_path, "python"))
import PyIVFTensor


# 加载 CUDA runtime
try:
    _libcudart = ctypes.CDLL("libcudart.so")
    _cudaHostRegister = _libcudart.cudaHostRegister
    _cudaHostRegister.restype = ctypes.c_int
    _cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]

    _cudaHostUnregister = _libcudart.cudaHostUnregister
    _cudaHostUnregister.restype = ctypes.c_int
    _cudaHostUnregister.argtypes = [ctypes.c_void_p]

    _CUDA_SUCCESS = 0
    _CUDA_HOST_REGISTER_DEFAULT = 0
    _CUDA_HOST_REGISTER_PORTABLE = 1
    _CUDA_HOST_REGISTER_DEVICEMAP = 2
    _CUDA_HOST_REGISTER_IOMEMORY = 4
except OSError:
    _libcudart = None
    _cudaHostRegister = None
    _cudaHostUnregister = None


def _get_memory_mb() -> float:
    """获取当前进程的内存使用量（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def register_array_as_pinned(arr: np.ndarray, flags: int = None) -> None:
    """使用 cudaHostRegister 将 numpy 数组注册为 pinned memory。

    这是真正的零拷贝：不分配新内存，只改变现有内存属性。
    注册后数组可用于 cudaMemcpyAsync 等异步操作。

    Args:
        arr: numpy 数组，必须是 contiguous
        flags: cudaHostRegister 标志，默认 PORTABLE | DEVICEMAP
    """
    if _cudaHostRegister is None:
        raise RuntimeError("CUDA runtime not available")

    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError("Array must be C-contiguous to register as pinned")

    if flags is None:
        flags = _CUDA_HOST_REGISTER_PORTABLE | _CUDA_HOST_REGISTER_DEVICEMAP

    ptr = arr.ctypes.data
    size = arr.nbytes

    result = _cudaHostRegister(
        ctypes.c_void_p(ptr),
        ctypes.c_size_t(size),
        ctypes.c_uint(flags),
    )
    if result != _CUDA_SUCCESS:
        raise RuntimeError(f"cudaHostRegister failed with error code {result}")

    print(
        f"[register_array_as_pinned] Registered {arr.nbytes / 1024 / 1024:.1f} MB as pinned memory at {ptr}",
        flush=True,
    )


def unregister_array_as_pinned(arr: np.ndarray) -> None:
    """注销 pinned memory 注册。

    Args:
        arr: 已注册的 numpy 数组
    """
    if _cudaHostUnregister is None:
        return

    ptr = arr.ctypes.data
    result = _cudaHostUnregister(ctypes.c_void_p(ptr))
    if result != _CUDA_SUCCESS:
        print(f"[unregister_array_as_pinned] Warning: cudaHostUnregister returned {result}", flush=True)
    else:
        print(f"[unregister_array_as_pinned] Unregistered array at {ptr}", flush=True)


class PinnedArrayContext:
    """上下文管理器：自动注册/注销 numpy 数组为 pinned memory。

    用法：
        with PinnedArrayContext(my_array):
            # my_array 现在是 pinned memory
            dataset.init_from_external_pinned(my_array, ...)
        # 自动注销
    """

    def __init__(self, arr: np.ndarray, flags: int = None):
        self.arr = arr
        self.flags = flags
        self.registered = False

    def __enter__(self):
        register_array_as_pinned(self.arr, self.flags)
        self.registered = True
        return self.arr

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.registered:
            unregister_array_as_pinned(self.arr)
        return False


def _validate_float32_matrix(arr: np.ndarray, name: str) -> None:
    """验证输入为 C-contiguous float32 2D 数组。"""
    if arr.dtype != np.float32:
        raise ValueError(f"{name}: expected float32, got {arr.dtype}")
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name}: array must be C-contiguous")
    if arr.ndim != 2:
        raise ValueError(f"{name}: expected 2D array, got {arr.ndim}D")


def numpy_to_pinned(arr: np.ndarray) -> PyIVFTensor.PinnedDataset:
    """将 numpy 数组包装到新的 PinnedDataset 中。

    注意：
    - 这里不是“零分配”；会创建一份新的 pinned memory 并复制数据进去。
    - 真正的 1x 数据集主路径应优先使用：
      1) load_*_to_pinned(filepath)
      2) fit_pinned(pinned_data)

    Args:
        arr: C-contiguous float32 数组，shape [n, dim]

    Returns:
        PinnedDataset 对象，底层为 cudaMallocHost 分配的 pinned memory
    """
    _validate_float32_matrix(arr, "numpy_to_pinned(arr)")

    n, dim = arr.shape
    data_size_mb = arr.nbytes / 1024 / 1024
    print(
        f"[numpy_to_pinned] Creating PinnedDataset for {n}x{dim} array ({data_size_mb:.1f} MB)",
        flush=True,
    )
    mem_before = _get_memory_mb()

    pinned = PyIVFTensor.PinnedDataset(n, dim)
    mem_after_create = _get_memory_mb()
    print(
        f"[numpy_to_pinned] After cudaMallocHost: +{mem_after_create - mem_before:.1f} MB",
        flush=True,
    )

    pinned.numpy()[:] = arr
    mem_after_copy = _get_memory_mb()
    print(
        f"[numpy_to_pinned] After copy to pinned: +{mem_after_copy - mem_after_create:.1f} MB",
        flush=True,
    )

    return pinned


def load_npy_to_pinned(filepath: str) -> PyIVFTensor.PinnedDataset:
    """从 .npy 文件直接读取到 pinned memory，避免中间 numpy 大数组。"""
    print(f"[load_npy_to_pinned] Loading {filepath} directly to pinned memory...", flush=True)
    mem_before = _get_memory_mb()

    arr_mmap = np.load(filepath, mmap_mode="r")
    if arr_mmap.dtype != np.float32:
        raise ValueError(f"Expected float32, got {arr_mmap.dtype}")
    if arr_mmap.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr_mmap.ndim}D")

    n, dim = arr_mmap.shape
    print(
        f"[load_npy_to_pinned] Array shape: {n}x{dim} ({arr_mmap.nbytes / 1024 / 1024:.1f} MB)",
        flush=True,
    )

    pinned = PyIVFTensor.PinnedDataset(n, dim)
    mem_after_create = _get_memory_mb()
    print(
        f"[load_npy_to_pinned] After cudaMallocHost: +{mem_after_create - mem_before:.1f} MB",
        flush=True,
    )

    block_size = min(100000, n)
    pinned_arr = pinned.numpy()

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        pinned_arr[start:end] = arr_mmap[start:end]
        if start == 0:
            mem_after_first = _get_memory_mb()
            print(
                f"[load_npy_to_pinned] After first block copy: +{mem_after_first - mem_after_create:.1f} MB",
                flush=True,
            )

    mem_after = _get_memory_mb()
    print(
        f"[load_npy_to_pinned] Done. Total pinned memory: +{mem_after - mem_before:.1f} MB",
        flush=True,
    )

    return pinned


def load_fvecs_to_pinned(filepath: str) -> PyIVFTensor.PinnedDataset:
    """从 .fvecs 文件直接读取到 pinned memory。"""
    print(f"[load_fvecs_to_pinned] Loading {filepath} directly to pinned memory...", flush=True)
    mem_before = _get_memory_mb()

    with open(filepath, "rb") as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)

        vec_size = 4 + dim * 4
        n = file_size // vec_size

        print(f"[load_fvecs_to_pinned] Detected {n} vectors of dim {dim}", flush=True)

        pinned = PyIVFTensor.PinnedDataset(n, dim)
        mem_after_create = _get_memory_mb()
        print(
            f"[load_fvecs_to_pinned] After cudaMallocHost: +{mem_after_create - mem_before:.1f} MB",
            flush=True,
        )

        pinned_arr = pinned.numpy()
        block_size = min(100000, n)

        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            block_n = end - start

            f.seek(start * vec_size)
            for i in range(block_n):
                f.seek(4, 1)
                vec = np.fromfile(f, dtype=np.float32, count=dim)
                pinned_arr[start + i] = vec

            if start == 0:
                mem_after_first = _get_memory_mb()
                print(
                    f"[load_fvecs_to_pinned] After first block: +{mem_after_first - mem_after_create:.1f} MB",
                    flush=True,
                )

    mem_after = _get_memory_mb()
    print(
        f"[load_fvecs_to_pinned] Done. Total pinned memory: +{mem_after - mem_before:.1f} MB",
        flush=True,
    )

    return pinned


def load_bvecs_to_pinned(filepath: str) -> PyIVFTensor.PinnedDataset:
    """从 .bvecs 文件直接读取到 pinned memory。"""
    print(f"[load_bvecs_to_pinned] Loading {filepath} directly to pinned memory...", flush=True)
    mem_before = _get_memory_mb()

    with open(filepath, "rb") as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)

        vec_size = 4 + dim
        n = file_size // vec_size

        print(f"[load_bvecs_to_pinned] Detected {n} vectors of dim {dim}", flush=True)

        pinned = PyIVFTensor.PinnedDataset(n, dim)
        mem_after_create = _get_memory_mb()
        print(
            f"[load_bvecs_to_pinned] After cudaMallocHost: +{mem_after_create - mem_before:.1f} MB",
            flush=True,
        )

        pinned_arr = pinned.numpy()
        block_size = min(100000, n)

        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            block_n = end - start

            f.seek(start * vec_size)
            for i in range(block_n):
                f.seek(4, 1)
                vec = np.fromfile(f, dtype=np.uint8, count=dim).astype(np.float32)
                pinned_arr[start + i] = vec

            if start == 0:
                mem_after_first = _get_memory_mb()
                print(
                    f"[load_bvecs_to_pinned] After first block: +{mem_after_first - mem_after_create:.1f} MB",
                    flush=True,
                )

    mem_after = _get_memory_mb()
    print(
        f"[load_bvecs_to_pinned] Done. Total pinned memory: +{mem_after - mem_before:.1f} MB",
        flush=True,
    )

    return pinned


def load_to_pinned(filepath: str) -> PyIVFTensor.PinnedDataset:
    """根据文件扩展名自动选择加载方式，直接读取到 pinned memory。"""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".npy":
        return load_npy_to_pinned(filepath)
    if ext == ".fvecs":
        return load_fvecs_to_pinned(filepath)
    if ext == ".bvecs":
        return load_bvecs_to_pinned(filepath)

    raise ValueError(f"Unsupported file format: {ext}. Supported: .npy, .fvecs, .bvecs")


def fit_kmeans_pinned(
    X: np.ndarray,
    n_clusters: int,
    metric: str = "angular",
    kmeans_iters: int = 20,
    use_minibatch: bool = False,
    use_interleaved: bool = False,
    device_id: int = 0,
    seed: int = 1234,
) -> Tuple[PyIVFTensor.ClusterDataset, PyIVFTensor.PinnedDataset]:
    """将 numpy 转入 pinned memory 后执行 K-means 聚类。

    返回 (dataset, pinned)，避免 pinned 对象在外层丢失引用。
    """
    _validate_float32_matrix(X, "fit_kmeans_pinned(X)")

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
    return dataset, pinned


def fit_kmeans_with_existing_pinned(
    pinned: PyIVFTensor.PinnedDataset,
    n_clusters: int,
    metric: str = "angular",
    kmeans_iters: int = 20,
    use_minibatch: bool = False,
    use_interleaved: bool = False,
    device_id: int = 0,
    seed: int = 1234,
) -> PyIVFTensor.ClusterDataset:
    """直接基于已有 PinnedDataset 执行 K-means。"""
    distance_mode = (
        PyIVFTensor.DISTANCE_COSINE if metric == "angular" else PyIVFTensor.DISTANCE_L2
    )

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


def fit_kmeans_pinned_file(
    filepath: str,
    n_clusters: int,
    metric: str = "angular",
    kmeans_iters: int = 20,
    use_minibatch: bool = False,
    use_interleaved: bool = False,
    device_id: int = 0,
    seed: int = 1234,
) -> Tuple[PyIVFTensor.ClusterDataset, PyIVFTensor.PinnedDataset]:
    """从文件直接加载到 pinned memory 并执行 K-means 聚类。"""
    distance_mode = (
        PyIVFTensor.DISTANCE_COSINE if metric == "angular" else PyIVFTensor.DISTANCE_L2
    )

    pinned = load_to_pinned(filepath)
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
    return dataset, pinned


class ReusablePinnedBuffer:
    """可复用的 pinned query buffer。

    目标：
    - 避免每次 search 都重新申请一份 pinned memory
    - 容量不足时再扩容
    """

    def __init__(self):
        self._pinned: Optional[PyIVFTensor.PinnedDataset] = None
        self._capacity_rows: int = 0
        self._dim: Optional[int] = None

    @property
    def capacity_rows(self) -> int:
        return self._capacity_rows

    @property
    def dim(self) -> Optional[int]:
        return self._dim

    def ensure_capacity(self, rows: int, dim: int) -> None:
        """确保 pinned buffer 至少能容纳 rows x dim。"""
        if rows <= 0:
            raise ValueError("rows must be > 0")
        if dim <= 0:
            raise ValueError("dim must be > 0")

        needs_realloc = (
            self._pinned is None
            or self._dim != dim
            or self._capacity_rows < rows
        )

        if not needs_realloc:
            return

        new_rows = max(rows, self._capacity_rows * 2 if self._capacity_rows > 0 else rows)
        print(
            f"[ReusablePinnedBuffer] Allocating pinned query buffer: {new_rows}x{dim}",
            flush=True,
        )
        self._pinned = PyIVFTensor.PinnedDataset(new_rows, dim)
        self._capacity_rows = new_rows
        self._dim = dim

    def stage(self, queries: np.ndarray) -> PyIVFTensor.PinnedDataset:
        """将 queries 写入 pinned workspace 并返回底层 PinnedDataset。"""
        _validate_float32_matrix(queries, "ReusablePinnedBuffer.stage(queries)")
        rows, dim = queries.shape
        self.ensure_capacity(rows, dim)
        self._pinned.numpy()[:rows, :] = queries
        return self._pinned


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
    searcher: Optional[PyIVFTensor.IVFSearcher] = None,
    query_workspace: Optional[ReusablePinnedBuffer] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """使用 PinnedDataset 路径执行 IVF 搜索。

    这一版支持 query pinned workspace 复用，避免每次查询重新分配 pinned memory。
    """
    _validate_float32_matrix(queries, "search_pinned(queries)")

    distance_mode = (
        PyIVFTensor.DISTANCE_COSINE if metric == "angular" else PyIVFTensor.DISTANCE_L2
    )

    if searcher is None:
        searcher = PyIVFTensor.IVFSearcher()

    if query_workspace is None:
        pinned_queries = numpy_to_pinned(queries)
    else:
        pinned_queries = query_workspace.stage(queries)

    cluster_sizes_i32 = np.asarray(cluster_sizes, dtype=np.int32)
    cluster_centers_f32 = np.asarray(cluster_centers, dtype=np.float32)

    # 尽量避免额外复制：
    # - 如果已经是 float32，np.asarray(..., dtype=np.float32) 会直接返回 view / 原对象
    # - 不再无脑 astype()
    cluster_vectors_f32 = np.asarray(cluster_vectors, dtype=np.float32)

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
        kwargs["reordered_indices"] = np.asarray(reordered_indices, dtype=np.int32)

    if vector_l2_norm is not None:
        kwargs["vector_l2_norm"] = np.asarray(vector_l2_norm, dtype=np.float32)

    return searcher.search_pinned_queries(
        pinned_queries,
        cluster_sizes_i32,
        cluster_vectors_f32,
        cluster_centers_f32,
        **kwargs,
    )


class PinnedIVFTensor:
    """基于 PinnedDataset 的 IVF-Tensor 封装。

    用法：
        ivf = PinnedIVFTensor(metric="angular", n_lists=100)
        ivf.fit(X)                    # 会复制到 pinned，不是 1x 主路径
        ivf.fit_file(path)            # 推荐：从文件直接读入 pinned
        ivf.fit_pinned(pinned_data)   # 推荐：直接传入 pinned 数据
        indices, distances = ivf.search(Q, k=10)
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
        self._pinned_data: Optional[PyIVFTensor.PinnedDataset] = None

        self._cluster_info: Optional[dict] = None
        self._centroids: Optional[np.ndarray] = None
        self._cluster_vectors_flat: Optional[np.ndarray] = None
        self._reordered_indices: Optional[np.ndarray] = None
        self._vector_l2_norm: Optional[np.ndarray] = None

        self._searcher = PyIVFTensor.IVFSearcher()
        self._query_workspace = ReusablePinnedBuffer()

    def _extract_index_payload(self) -> None:
        """从 ClusterDataset 提取搜索所需 payload。

        注意：
        - 当前 PyIVFTensor 接口要求 Python 侧仍传入 cluster_vectors。
        - 这里避免使用 flatten()，优先使用 reshape(-1) 以减少整库复制概率。
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not initialized")

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
            "k": int(n_clusters),
            "offsets": np.asarray(offsets, dtype=np.int32),
            "counts": np.asarray(counts, dtype=np.int32),
        }
        self._reordered_indices = np.asarray(reordered_indices, dtype=np.int32)
        self._centroids = np.asarray(centroids, dtype=np.float32)

        # 避免 flatten() 无脑复制。
        # 如果 reordered_data 本身是连续的，这里通常会返回 view。
        try:
            self._cluster_vectors_flat = reordered_data.reshape(-1)
        except Exception:
            # 极端情况下退化复制，但明确打印告警，便于定位峰值问题。
            print(
                "[PinnedIVFTensor] Warning: reordered_data.reshape(-1) failed; "
                "falling back to np.ascontiguousarray(...).reshape(-1), which may copy full dataset.",
                flush=True,
            )
            self._cluster_vectors_flat = np.ascontiguousarray(reordered_data).reshape(-1)

        if vector_l2_norm is not None and getattr(vector_l2_norm, "size", 0) > 0:
            self._vector_l2_norm = np.asarray(vector_l2_norm, dtype=np.float32)
        else:
            self._vector_l2_norm = None

    def fit(self, X: np.ndarray) -> None:
        """执行 K-means 聚类并构建索引。

        注意：
        - 此路径会把 X 复制到新的 pinned memory，不满足最严格的 1x 主路径目标。
        - 真正推荐的大数据路径是 fit_file() 或 fit_pinned()。
        """
        self._dataset, self._pinned_data = fit_kmeans_pinned(
            X,
            n_clusters=self._n_lists,
            metric=self._metric,
            kmeans_iters=self._kmeans_iters,
            use_minibatch=self._use_minibatch,
            use_interleaved=self._use_interleaved,
            device_id=self._device_id,
        )
        self._extract_index_payload()

    def fit_pinned(self, pinned_data: PyIVFTensor.PinnedDataset) -> None:
        """直接基于已有 PinnedDataset 构建索引。

        这是推荐主路径之一，避免先构造普通 numpy 再复制。
        """
        self._pinned_data = pinned_data
        self._dataset = fit_kmeans_with_existing_pinned(
            pinned=pinned_data,
            n_clusters=self._n_lists,
            metric=self._metric,
            kmeans_iters=self._kmeans_iters,
            use_minibatch=self._use_minibatch,
            use_interleaved=self._use_interleaved,
            device_id=self._device_id,
        )
        self._extract_index_payload()

    def fit_file(self, filepath: str) -> None:
        """从文件直接加载到 pinned memory 并执行聚类。

        这是推荐主路径之一，最接近 1x dataset 主机侧峰值目标。
        """
        self._dataset, self._pinned_data = fit_kmeans_pinned_file(
            filepath,
            n_clusters=self._n_lists,
            metric=self._metric,
            kmeans_iters=self._kmeans_iters,
            use_minibatch=self._use_minibatch,
            use_interleaved=self._use_interleaved,
            device_id=self._device_id,
        )
        self._extract_index_payload()

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
        reuse_query_workspace: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """执行搜索。

        默认启用可复用 pinned query workspace，避免重复申请 pinned query buffer。
        """
        if self._dataset is None:
            raise RuntimeError("Index not fitted. Call fit() first.")
        if self._cluster_info is None:
            raise RuntimeError("Index payload not extracted")
        if self._cluster_vectors_flat is None:
            raise RuntimeError("Cluster vectors not prepared")
        if self._centroids is None:
            raise RuntimeError("Centroids not prepared")

        if n_probes is None:
            n_probes = self._n_lists

        workspace = self._query_workspace if reuse_query_workspace else None

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
            searcher=self._searcher,
            query_workspace=workspace,
        )

    def reset_query_workspace(self) -> None:
        """释放 query workspace 的 Python 引用。

        注：具体 pinned memory 的实际释放时机仍取决于底层对象生命周期。
        """
        self._query_workspace = ReusablePinnedBuffer()

    def get_debug_state(self) -> dict:
        """返回当前对象的重要状态，便于调试内存路径。"""
        return {
            "metric": self._metric,
            "n_lists": self._n_lists,
            "kmeans_iters": self._kmeans_iters,
            "use_minibatch": self._use_minibatch,
            "use_interleaved": self._use_interleaved,
            "device_id": self._device_id,
            "has_dataset": self._dataset is not None,
            "has_pinned_data": self._pinned_data is not None,
            "has_cluster_info": self._cluster_info is not None,
            "has_centroids": self._centroids is not None,
            "has_cluster_vectors_flat": self._cluster_vectors_flat is not None,
            "has_reordered_indices": self._reordered_indices is not None,
            "has_vector_l2_norm": self._vector_l2_norm is not None,
            "query_workspace_capacity_rows": self._query_workspace.capacity_rows,
            "query_workspace_dim": self._query_workspace.dim,
        }