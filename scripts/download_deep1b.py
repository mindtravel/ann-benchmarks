#!/usr/bin/env python3
"""
下载 Deep1B（Yandex DEEP）到 /data/raw_dataset/deep1b
- base.1B.fbin、query.public.10K.fbin、groundtruth.public.10K.ibin
- 可选：learn.350M.fbin
- 格式：.fbin = [n(uint32), d(uint32), float32 数据]，.ibin = [n, d, int32]
- 支持断点续传；需代理时设置 HTTPS_PROXY。
"""
import os
import sys

TARGET_DIR = "/data/raw_dataset/deep1b"
BASE_URL = "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP"

# 1B 基集约 384GB (10^9 * 96 * 4 bytes)
EXPECTED_BASE_BYTES = 8 + 1_000_000_000 * 96 * 4  # 8 字节头 + 1B * 96 * float32
EXPECTED_QUERY_BYTES = 8 + 10_000 * 96 * 4
EXPECTED_GT_BYTES = 8 + 10_000 * 100 * 4  # 100 近邻索引，int32，约 3.8 MB

os.makedirs(TARGET_DIR, exist_ok=True)
os.chdir(TARGET_DIR)


def get_proxies():
    p = {}
    for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        if os.environ.get(k):
            v = os.environ[k]
            p.setdefault("http", v)
            p.setdefault("https", v)
            break
    return p if p else None


def download_http(url, out_path, proxies=None):
    """带断点续传的 HTTP 下载"""
    try:
        import urllib.request
        proxy = (proxies or {}).get("https") or (proxies or {}).get("http")
        if proxy:
            proxy_handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
            opener = urllib.request.build_opener(proxy_handler)
            urllib.request.install_opener(opener)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        if os.path.exists(out_path):
            size = os.path.getsize(out_path)
            req.add_header("Range", f"bytes={size}-")
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status not in (200, 206):
                return False
            mode = "ab" if (r.status == 206) else "wb"
            with open(out_path, mode) as f:
                chunk = 1024 * 1024
                while True:
                    b = r.read(chunk)
                    if not b:
                        break
                    f.write(b)
                    print(f"\r  {os.path.getsize(out_path) / (1024**2):.1f} MB", end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f" 失败: {e}", file=sys.stderr)
        return False


def main():
    proxies = get_proxies()
    if not proxies:
        print("当前未设置代理。若 Yandex 不可达，可设置：")
        # print("  export HTTPS_PROXY=http://127.0.0.1:7890")
        print("  python scripts/download_deep1b.py")
        print()

    files = [
        ("base.1B.fbin", EXPECTED_BASE_BYTES, "约 384 GB"),
        ("query.public.10K.fbin", EXPECTED_QUERY_BYTES, "约 3.7 MB"),
        ("groundtruth.public.10K.ibin", EXPECTED_GT_BYTES, "约 4 MB"),
    ]
    for name, expected_size, desc in files:
        out_path = os.path.join(TARGET_DIR, name)
        url = f"{BASE_URL}/{name}"
        if os.path.exists(out_path) and os.path.getsize(out_path) >= expected_size:
            print(f"已存在且完整，跳过: {name}")
            continue
        if os.path.exists(out_path):
            print(f"{name} 未下完（当前 {os.path.getsize(out_path) / (1024**3):.1f} GB），断点续传...")
        else:
            print(f"下载 {name} ({desc}) ...")
        if not download_http(url, out_path, proxies):
            print(f"  -> 可设置代理后重试，或从 {BASE_URL} 手动下载到 {TARGET_DIR}", file=sys.stderr)

    # 可选：learn 集（约 350M * 96 * 4 ≈ 134 GB）
    learn_name = "learn.350M.fbin"
    learn_path = os.path.join(TARGET_DIR, learn_name)
    learn_url = f"{BASE_URL}/{learn_name}"
    if not os.path.exists(learn_path):
        print(f"可选：下载 {learn_name}（约 134 GB）？若需要请取消下一行注释并重跑。")
        # if not download_http(learn_url, learn_path, proxies):
        #     print(f"  -> 可选学习集下载失败", file=sys.stderr)
    else:
        print(f"已存在: {learn_name}")

    print("当前目录:", TARGET_DIR)
    for f in sorted(os.listdir(TARGET_DIR)):
        p = os.path.join(TARGET_DIR, f)
        if os.path.isfile(p):
            print(f"  {f}  {os.path.getsize(p) / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()
