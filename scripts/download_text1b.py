#!/usr/bin/env python3
"""
下载 Text1B（Microsoft SPACEV1B 或 Yandex Text-to-Image-1B）到 /data/raw_dataset/text1b

选项 A - Microsoft SPACEV1B（文本向量，100 维 int8）：
- 从 HuggingFace 镜像下载（如 maknee/spacev1b）：vectors.bin, query.bin, truth.bin
- 格式：前 4 字节 n、4 字节 d，随后 int8 数据

选项 B - Yandex Text-to-Image-1B（T2I，跨模态，200 维 float32）：
- base.1B.fbin, query.public.100K.fbin, groundtruth.public.100K.ibin
- 格式：.fbin = [n(uint32), d(uint32), float32], .ibin = [n, d, int32]

默认使用选项 A（SPACEV1B）；可通过环境变量 USE_T2I=1 使用选项 B。
支持断点续传；需代理时设置 HTTPS_PROXY。
"""
import os
import sys

TARGET_DIR = "/data/raw_dataset/text1b"
USE_T2I = os.environ.get("USE_T2I", "").strip() in ("1", "yes", "true")

# SPACEV1B (HF)
HF_MIRROR = "https://hf-mirror.com/datasets/maknee/spacev1b/resolve/main"
SPACEV_FILES = [
    ("vectors.bin", None),   # 约 140GB，1.4B * 100 int8
    ("query.bin", None),
    ("truth.bin", None),
]

# Yandex T2I
T2I_BASE_URL = "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I"
T2I_FILES = [
    ("base.1B.fbin", 8 + 1_000_000_000 * 200 * 4),
    ("query.public.100K.fbin", 8 + 100_000 * 200 * 4),
    ("groundtruth.public.100K.ibin", 8 + 100_000 * 100 * 4),
]

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
        print("当前未设置代理。若 HF/Yandex 不可达，可设置：")
        print("  export HTTPS_PROXY=http://127.0.0.1:7890")
        print("  python scripts/download_text1b.py")
        print()

    if USE_T2I:
        print("使用 Yandex Text-to-Image-1B (T2I)")
        for name, expected_size in T2I_FILES:
            out_path = os.path.join(TARGET_DIR, name)
            url = f"{T2I_BASE_URL}/{name}"
            if os.path.exists(out_path) and expected_size and os.path.getsize(out_path) >= expected_size:
                print(f"已存在且完整，跳过: {name}")
                continue
            if os.path.exists(out_path):
                print(f"{name} 未下完，断点续传...")
            else:
                print(f"下载 {name} ...")
            if not download_http(url, out_path, proxies):
                print(f"  -> 可设置代理后重试，或从 {T2I_BASE_URL} 手动下载", file=sys.stderr)
    else:
        print("使用 Microsoft SPACEV1B（HF 镜像）")
        for remote_name, _ in SPACEV_FILES:
            out_path = os.path.join(TARGET_DIR, remote_name)
            url = f"{HF_MIRROR}/{remote_name}"
            if os.path.exists(out_path):
                print(f"已存在，跳过: {remote_name}")
                continue
            print(f"下载 {remote_name} ...")
            if not download_http(url, out_path, proxies):
                print(f"  -> 可设置代理后重试，或从 https://huggingface.co/datasets/maknee/spacev1b 手动下载到 {TARGET_DIR}", file=sys.stderr)
        print("若 HF 上无此文件，可从 Microsoft SPTAG 或 Azure 获取 SPACEV1B；或使用 USE_T2I=1 下载 Yandex T2I。")

    print("当前目录:", TARGET_DIR)
    for f in sorted(os.listdir(TARGET_DIR)):
        p = os.path.join(TARGET_DIR, f)
        if os.path.isfile(p):
            print(f"  {f}  {os.path.getsize(p) / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()
