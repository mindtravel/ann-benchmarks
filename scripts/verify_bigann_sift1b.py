#!/usr/bin/env python3
"""
验证 BigANN 与 SIFT1B 是否为同一份数据：对比 base.1B.u8bin 与 bigann_base.bvecs 前 N 条向量。
"""
import os
import struct
import sys

# 数据目录：命令行参数 > 环境变量 SIFT1B_DIR > raw_data/sift1B > /data/raw_dataset/sift1b
TARGET_DIR = os.environ.get("SIFT1B_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw_data", "sift1B"
)
if not os.path.isdir(TARGET_DIR) and os.path.isdir("/data/raw_dataset/sift1b"):
    TARGET_DIR = "/data/raw_dataset/sift1b"
if len(sys.argv) > 1:
    TARGET_DIR = os.path.abspath(sys.argv[1])
N_COMPARE = 100  # 对比前 100 条向量


def read_u8bin_header(path):
    with open(path, "rb") as f:
        h = f.read(8)
        if len(h) < 8:
            return None, None
        n, d = struct.unpack("<II", h)
        return n, d


def read_u8bin_vectors(path, n_vectors, dim):
    with open(path, "rb") as f:
        f.seek(8)
        size = n_vectors * dim
        data = f.read(size)
        if len(data) != size:
            return None
        return data


def read_bvecs_vectors(path, n_vectors):
    dim = None
    out = []
    with open(path, "rb") as f:
        for _ in range(n_vectors):
            h = f.read(4)
            if len(h) < 4:
                break
            d = struct.unpack("<i", h)[0]
            if dim is None:
                dim = d
            block = f.read(d)
            if len(block) != d:
                break
            out.extend(block)
        if dim is None:
            return None, None
        return dim, bytes(out)


def main():
    print("数据目录:", TARGET_DIR)
    u8bin_path = os.path.join(TARGET_DIR, "base.1B.u8bin")
    bvecs_path = os.path.join(TARGET_DIR, "bigann_base.bvecs")

    if not os.path.exists(u8bin_path):
        print(f"缺少: {u8bin_path}")
        u8bin_path = None
    if not os.path.exists(bvecs_path):
        print(f"缺少: {bvecs_path}")
        bvecs_path = None
    if not u8bin_path or not bvecs_path:
        print("请先下载 base，再运行本脚本验证。")
        print("  下载命令（在 ann-benchmarks 根目录执行）: python scripts/download_sift1b.py")
        print("  或指定数据目录: python scripts/verify_bigann_sift1b.py /path/to/sift1b")
        sys.exit(1)

    print("1) 元数据对比")
    print("-" * 50)
    n_u, d_u = read_u8bin_header(u8bin_path)
    print(f"  base.1B.u8bin:     n={n_u:,}  d={d_u}")
    d_b, data_b = read_bvecs_vectors(bvecs_path, 1)
    if d_b is None:
        print("  bigann_base.bvecs: 无法读取")
        sys.exit(1)
    # 用 bvecs 推断 n：文件大小 / (4+d)
    bvecs_size = os.path.getsize(bvecs_path)
    n_b = bvecs_size // (4 + d_b)
    print(f"  bigann_base.bvecs: n={n_b:,}  d={d_b}")
    if d_u != d_b:
        print("  维度不一致，不是同一格式/数据源。")
        sys.exit(1)
    dim = d_u
    print(f"  维度一致: {dim}（符合 SIFT 描述子 128 维）")
    print()

    print("2) 前 N 条向量逐字节对比")
    print("-" * 50)
    n_compare = min(N_COMPARE, n_u, n_b)
    data_u = read_u8bin_vectors(u8bin_path, n_compare, dim)
    _, data_b = read_bvecs_vectors(bvecs_path, n_compare)
    if data_u is None or data_b is None:
        print("  读取失败")
        sys.exit(1)
    if data_u == data_b:
        print(f"  前 {n_compare} 条向量完全一致 → BigANN 与 SIFT1B base 为同一份数据。")
    else:
        diff = sum(1 for a, b in zip(data_u, data_b) if a != b)
        print(f"  前 {n_compare} 条向量有 {diff} 字节不同")
        if diff > 0:
            for i in range(min(3, n_compare)):
                off = i * dim
                u_vec = data_u[off : off + dim]
                b_vec = data_b[off : off + dim]
                print(f"    向量 {i}: u8bin 前 8 字节 {list(u_vec[:8])}  bvecs 前 8 字节 {list(b_vec[:8])}")
    print()
    print("验证完成。")


if __name__ == "__main__":
    main()
