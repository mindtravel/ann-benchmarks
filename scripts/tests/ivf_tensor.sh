#!/bin/bash
# ours ivftensor测试脚本（batch_size 在 ann_benchmarks/algorithms/ivf_tensor/config.yml 中配置）
# 用法: ./scripts/tests/ivf_tensor.sh <dataset>
# 与 PyIVFTensor（python3.10 编译）对齐：默认 python3.10，可用 PYTHON_BIN 覆盖
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ANN_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.10 2>/dev/null || command -v python3)}"
DATASET="${1:-SIFT1M-128-euclidean}"
echo "ours ivf_tensor测试开始..."

"$PYTHON_BIN" run.py --local --algorithm ivf_tensor --dataset "$DATASET" --force --runs 1 --batch

echo "ours ivf_tensor测试完成"

# 不重新计算，只想画图用这个命令
# python plot.py --x-scale linear --y-scale log --batch --dataset "SIFT1M-128-euclidean"
