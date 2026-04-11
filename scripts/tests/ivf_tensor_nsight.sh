#!/bin/bash
# 使用 Nsight Systems 采集 ivf_tensor 的 GPU 时间线（含 NVTX 阶段）
# 用法: ./scripts/tests/ivf_tensor_nsight.sh [dataset]
# 例:   ./scripts/tests/ivf_tensor_nsight.sh SIFT1M-128-euclidean
# 输出: nsys-report-ivf_tensor-<dataset>-<timestamp>.nsys-rep
# 用 Nsight Systems GUI 打开 .nsys-rep 可看 Stage0_Prepare / Stage1_Coarse / Stage2_Entry / Stage3_Fine / Stage4_Lookup 及重叠

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ANN_ROOT"

# 与 ivftensor 中 PyIVFTensor 编译所用 Python 一致（默认 3.10）
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.10 2>/dev/null || command -v python3)}"

if ! command -v nsys &>/dev/null; then
    echo "错误: 未找到 nsys。请安装 Nsight Systems 并确保 nsys 在 PATH 中。"
    echo "例如: 安装 CUDA Toolkit 后 nsys 位于 /usr/local/cuda/bin/nsys 或 \$CUDA_HOME/bin/nsys"
    exit 1
fi

DATASET="${1:-SIFT10M-128-euclidean}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="${ANN_ROOT}/report"
REPORT="${REPORT_DIR}/nsys-report-ivf_tensor-${DATASET}-${TIMESTAMP}.nsys-rep"
mkdir -p "$REPORT_DIR"

# NSYS_PROFILE=1 使 benchmark 在主进程执行，这样 nsys 才能采到 CUDA/NVTX（子进程不会被 trace）
# --trace=cuda 采集 CUDA API 和 kernel，nvtx 采集 NVTX 区间（Stage0~Stage4）
# 解析 --nsight 参数，可选是否采集 Nsight Systems 报告
NSIGHT=1
for arg in "$@"; do
    case "$arg" in
        --nsight)
            NSIGHT=1
            ;;
    esac
done

if [ "$NSIGHT" -eq 0 ]; then
    echo "跳过 Nsight Systems 采集，仅运行测试 (无 nsys)..."
    IVF_DEBUG_INDEX_RANGE=0 IVF_BATCH_SERIAL=0 "$PYTHON_BIN" run.py --local --algorithm ivf_tensor --dataset "$DATASET" --force --runs 1 --batch
    exit 0
fi

if [ "$NSIGHT" -eq 1 ]; then
    echo "Nsight Systems: 采集 ivf_tensor 测试 (dataset=$DATASET)"
    echo "报告将保存为: $REPORT"
    echo ""
    nsys profile \
    -o "$REPORT" \
    --trace=cuda,nvtx \
    --force-overwrite=true \
    --stats=true \
    env NSYS_PROFILE=1 IVF_DEBUG_INDEX_RANGE=0 IVF_BATCH_SERIAL=0 python run.py --local --algorithm ivf_tensor --dataset "$DATASET" --force --runs 1 --batch
fi


# python run.py --local --algorithm ivf_tensor --dataset "$DATASET" --force -c-runs 1 --batch

echo ""
echo "采集完成。查看方式："
echo "  1) 无显示器: 导出/拷到本机或用 stats:"
echo "     ./scripts/tests/nsys_export_report.sh $REPORT json"
echo "     nsys stats $REPORT"
