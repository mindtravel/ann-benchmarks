#!/usr/bin/env python3
"""
下载 SIFT1B 到 /data/raw_dataset/sift1b
- query/learn/gnd：HF 镜像直链（需代理时设置 HTTPS_PROXY），断点续传。
- base：从 Facebook big-ann-benchmarks 镜像下载 base.1B.u8bin，再转为 bigann_base.bvecs。
"""
import os
import sys
import gzip
import shutil

TARGET_DIR = "/data/raw_dataset/sift1b"
# HF 镜像直链（走代理时可用）
HF_MIRROR = "https://hf-mirror.com/datasets/fzliu/sift1b/resolve/main"

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
        with urllib.request.urlopen(req, timeout=30) as r:
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


def decompress_gz(gz_path, out_path):
    if os.path.exists(out_path):
        return
    if not os.path.exists(gz_path):
        return
    print(f"解压 {os.path.basename(gz_path)} -> {os.path.basename(out_path)} ...")
    with gzip.open(gz_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def u8bin_to_bvecs(u8bin_path, bvecs_path):
    """将 base.1B.u8bin（头 8 字节 n,d + 裸数据）转为 bigann_base.bvecs（每向量 4 字节 dim + 128 字节）"""
    import struct
    chunk_vectors = 2 * 10**6  # 每次处理 200 万向量
    with open(u8bin_path, "rb") as f_in:
        header = f_in.read(8)
        if len(header) < 8:
            raise ValueError("u8bin 文件过短")
        n, d = struct.unpack("<II", header)
        with open(bvecs_path, "wb") as f_out:
            written = 0
            while written < n:
                take = min(chunk_vectors, n - written)
                block = f_in.read(take * d)
                if len(block) != take * d:
                    raise ValueError("u8bin 读取长度不符")
                out = bytearray(take * (4 + d))
                for i in range(take):
                    struct.pack_into("<i", out, i * (4 + d), d)
                    out[i * (4 + d) + 4 : (i + 1) * (4 + d)] = block[i * d : (i + 1) * d]
                f_out.write(out)
                written += take
                print(f"\r  已转换 {written / 1e6:.1f}M / {n / 1e6:.1f}M 向量", end="", flush=True)
    print()
    print(f"  已写出: {bvecs_path}")


def clean_tmp():
    """删除之前失败/中断留下的 .tmp 文件"""
    for name in os.listdir(TARGET_DIR):
        p = os.path.join(TARGET_DIR, name)
        if name.endswith(".tmp") and os.path.isfile(p):
            try:
                os.remove(p)
                print(f"已删除残留: {name}")
            except OSError:
                pass
    gnd_dir = os.path.join(TARGET_DIR, "gnd")
    if os.path.isdir(gnd_dir):
        for name in os.listdir(gnd_dir):
            if name.endswith(".tmp"):
                try:
                    os.remove(os.path.join(gnd_dir, name))
                    print(f"已删除残留: gnd/{name}")
                except OSError:
                    pass


def main():
    clean_tmp()
    proxies = get_proxies()
    if not proxies:
        print("当前未设置代理。若 HF/FTP 不可达，请先设置代理再运行，例如：")
        print("  export HTTPS_PROXY=http://127.0.0.1:7890")
        print("  python scripts/download_sift1b.py")
        print()

    # fzliu/sift1b 仅有 query、learn、gnd，无 base（92GB 需另找）
    files = [
        ("bigann_query.bvecs.gz", "bigann_query.bvecs"),
        ("bigann_learn.bvecs.gz", "bigann_learn.bvecs"),
        ("bigann_gnd.tar.gz", None),
    ]
    for remote_name, local_name in files:
        url = f"{HF_MIRROR}/{remote_name}"
        local_path = os.path.join(TARGET_DIR, remote_name)
        if os.path.exists(local_path) and (not local_name or os.path.exists(os.path.join(TARGET_DIR, local_name))):
            print(f"已存在，跳过: {remote_name}")
            continue
        print(f"下载 {remote_name} ...")
        if download_http(url, local_path, proxies):
            if local_name and remote_name.endswith(".gz"):
                decompress_gz(local_path, os.path.join(TARGET_DIR, local_name))
        else:
            print(f"  -> 请设置代理后重试，或从 https://huggingface.co/datasets/fzliu/sift1b 手动下载到 {TARGET_DIR}", file=sys.stderr)

    # 解压 gnd
    gnd_tar = os.path.join(TARGET_DIR, "bigann_gnd.tar.gz")
    gnd_dir = os.path.join(TARGET_DIR, "gnd")
    if os.path.exists(gnd_tar):
        os.makedirs(gnd_dir, exist_ok=True)
        if not os.listdir(gnd_dir):
            print("解压 bigann_gnd.tar.gz -> gnd/ ...")
            os.system(f"tar -xzf '{gnd_tar}' -C '{TARGET_DIR}'")

    # base：从 Facebook big-ann-benchmarks 镜像用脚本下载（u8bin 格式），再转为 bvecs 供 ann_benchmarks 用
    base_bvecs = os.path.join(TARGET_DIR, "bigann_base.bvecs")
    base_u8bin = os.path.join(TARGET_DIR, "base.1B.u8bin")
    FB_BASE_URL = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin"
    # 完整 base.1B.u8bin 大小：8 字节头 + 10^9 * 128
    EXPECTED_U8BIN_BYTES = 8 + 1_000_000_000 * 128
    need_download = not os.path.exists(base_u8bin) or os.path.getsize(base_u8bin) < EXPECTED_U8BIN_BYTES
    if need_download:
        if os.path.exists(base_u8bin):
            current = os.path.getsize(base_u8bin)
            print(f"base.1B.u8bin 未下完（当前 {current / (1024**3):.1f} GB，需约 119 GB），断点续传...")
        else:
            print("下载 base 集 base.1B.u8bin（约 119GB，可断点续传）...")
        if download_http(FB_BASE_URL, base_u8bin, proxies):
            pass
        else:
            print("  -> 可设置代理后重试，或从 https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/ 手动下载 base.1B.u8bin", file=sys.stderr)
    u8bin_complete = os.path.exists(base_u8bin) and os.path.getsize(base_u8bin) >= EXPECTED_U8BIN_BYTES
    if u8bin_complete and (not os.path.exists(base_bvecs) or os.path.getsize(base_bvecs) < 1_000_000_000 * (4 + 128)):
        print("将 base.1B.u8bin 转为 bigann_base.bvecs（供 ann_benchmarks 使用）...")
        u8bin_to_bvecs(base_u8bin, base_bvecs)
    if not os.path.exists(base_bvecs) or os.path.getsize(base_bvecs) < 1_000_000_000 * (4 + 128):
        if need_download and not u8bin_complete:
            print("  -> base 未下完，请再次运行本脚本继续续传: python scripts/download_sift1b.py", file=sys.stderr)
        elif not os.path.exists(base_bvecs):
            print("  -> 未得到 bigann_base.bvecs，请设置代理后重新运行本脚本", file=sys.stderr)

    print("当前目录:", TARGET_DIR)
    for f in sorted(os.listdir(TARGET_DIR)):
        p = os.path.join(TARGET_DIR, f)
        try:
            if os.path.isfile(p):
                print(f"  {f}  {os.path.getsize(p) / (1024**3):.2f} GB")
            else:
                print(f"  {f}/")
                for g in sorted(os.listdir(p)):
                    gp = os.path.join(p, g)
                    if os.path.isfile(gp):
                        print(f"    {g}  {os.path.getsize(gp) / (1024**3):.2f} GB")
        except OSError:
            print(f"  {f}")


if __name__ == "__main__":
    main()
