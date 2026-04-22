"""
IVF-Tensor Pinned Memory 工具模块

提供从文件直接加载到 pinned memory 的功能，以及查询 buffer 复用。
这是简化后的版本，仅保留 module.py 所需的接口。
"""

import os
import sys
from typing import Optional

import numpy as np

# 加载 PyIVFTensor
_ivftensor_path = "/home/diy/lzx/ivftensor"
_module_paths = os.path.join(_ivftensor_path, "python/build")
sys.path.insert(0, _module_paths)
sys.path.insert(0, os.path.join(_ivftensor_path, "python"))
import PyIVFTensor


def numpy_to_pinned(arr: np.ndarray) -> PyIVFTensor.PinnedDataset:
    """将 numpy 数组复制到新的 PinnedDataset（兼容路径，非 1x 主路径）。

    Args:
        arr: C-contiguous float32 数组，shape [n, dim]

    Returns:
        PinnedDataset 对象
    """
    if arr.dtype != np.float32:
        raise ValueError(f"Expected float32, got {arr.dtype}")
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError("Array must be C-contiguous")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D")

    n, dim = arr.shape
    pinned = PyIVFTensor.PinnedDataset(n, dim)
    pinned.numpy()[:] = arr
    return pinned


def load_npy_to_pinned(filepath: str) -> PyIVFTensor.PinnedDataset:
    """从 .npy 文件直接加载到 pinned memory。

    使用 numpy.memmap 避免一次性加载到普通内存。
    """
    arr_mmap = np.load(filepath, mmap_mode="r")

    if arr_mmap.dtype != np.float32:
        raise ValueError(f"Expected float32, got {arr_mmap.dtype}")
    if arr_mmap.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr_mmap.ndim}D")

    n, dim = arr_mmap.shape
    pinned = PyIVFTensor.PinnedDataset(n, dim)

    # 分块复制，避免占用过多内存
    block_size = min(100000, n)
    pinned_arr = pinned.numpy()

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        pinned_arr[start:end] = arr_mmap[start:end]

    return pinned


def load_fvecs_to_pinned(filepath: str) -> PyIVFTensor.PinnedDataset:
    """从 .fvecs 文件加载到 pinned memory。"""
    with open(filepath, "rb") as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)

        vec_size = 4 + dim * 4
        n = file_size // vec_size

        pinned = PyIVFTensor.PinnedDataset(n, dim)
        pinned_arr = pinned.numpy()
        block_size = min(100000, n)

        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            for i in range(start, end):
                f.seek(i * vec_size)
                f.seek(4, 1)  # skip dim
                pinned_arr[i] = np.fromfile(f, dtype=np.float32, count=dim)

    return pinned


def load_bvecs_to_pinned(filepath: str, max_n: int = None) -> PyIVFTensor.PinnedDataset:
    """从 .bvecs 文件加载到 pinned memory。

    优化点：
    1. 不再逐向量 seek / fromfile
    2. 不再逐向量 astype 产生临时 float32 数组
    3. 使用 memmap 按块读取，避免一次性把整个文件搬进普通内存
    4. 仅保留一次必要的 uint8 -> float32 转换，直接写入 pinned memory

    Args:
        filepath: bvecs 文件路径
        max_n: 最大读取向量数（None = 读取全部）
    """
    import os
    import numpy as np

    with open(filepath, "rb") as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        if dim <= 0:
            raise ValueError(f"Invalid bvecs dim: {dim}")

        f.seek(0, os.SEEK_END)
        file_size = f.tell()

    vec_size = 4 + dim
    if file_size % vec_size != 0:
        raise ValueError(
            f"Invalid bvecs file size: {file_size} is not divisible by record size {vec_size}"
        )

    total_n = file_size // vec_size
    n = min(total_n, max_n) if max_n is not None else total_n

    print(
        f"[load_bvecs_to_pinned] Loading {n} vectors (dim={dim}) from {filepath}",
        flush=True,
    )

    pinned = PyIVFTensor.PinnedDataset(n, dim)
    pinned_arr = pinned.numpy()

    # 每条记录: 4-byte dim + dim-byte uint8 vector
    record_dtype = np.dtype([
        ("dim", np.int32),
        ("vec", np.uint8, dim),
    ])

    # 零拷贝映射文件，不把整文件先读进普通内存
    records = np.memmap(filepath, mode="r", dtype=record_dtype, shape=(total_n,))

    # 可选一致性检查：只检查首条，避免全文件扫描
    if int(records[0]["dim"]) != dim:
        raise ValueError(
            f"Inconsistent bvecs header: first dim={int(records[0]['dim'])}, expected {dim}"
        )

    block_size = min(100000, n)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)

        # 这里只做一次必要的 uint8 -> float32 转换，直接写入 pinned_arr
        # 不再显式 astype(copy=True)
        np.copyto(
            pinned_arr[start:end],
            records[start:end]["vec"],
            casting="unsafe",
        )

    return pinned


def load_to_pinned(filepath: str) -> PyIVFTensor.PinnedDataset:
    """根据文件扩展名自动选择加载方式。"""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".npy":
        return load_npy_to_pinned(filepath)
    if ext == ".fvecs":
        return load_fvecs_to_pinned(filepath)
    if ext == ".bvecs":
        return load_bvecs_to_pinned(filepath)

    raise ValueError(f"Unsupported file format: {ext}")


class ReusablePinnedBuffer:
    """可复用的 pinned query buffer。

    避免每次 search 都重新申请 pinned memory。
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
        if rows <= 0 or dim <= 0:
            raise ValueError("rows and dim must be > 0")

        if self._pinned is not None and self._dim == dim and self._capacity_rows >= rows:
            return

        # 扩容策略：2倍或实际需求
        new_rows = max(rows, self._capacity_rows * 2 if self._capacity_rows > 0 else rows)
        self._pinned = PyIVFTensor.PinnedDataset(new_rows, dim)
        self._capacity_rows = new_rows
        self._dim = dim

    def stage(self, queries: np.ndarray) -> PyIVFTensor.PinnedDataset:
        """将 queries 写入 pinned workspace 并返回底层 PinnedDataset。"""
        if queries.dtype != np.float32:
            raise ValueError(f"Expected float32, got {queries.dtype}")
        if not queries.flags["C_CONTIGUOUS"]:
            raise ValueError("Array must be C-contiguous")
        if queries.ndim != 2:
            raise ValueError(f"Expected 2D array, got {queries.ndim}D")

        rows, dim = queries.shape
        self.ensure_capacity(rows, dim)
        self._pinned.numpy()[:rows, :] = queries
        return self._pinned
