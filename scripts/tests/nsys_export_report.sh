#!/bin/bash
# 将 .nsys-rep 报告导出为 JSON/文本，便于在无显示器服务器上分析或拷到本机查看
# 用法: ./scripts/tests/nsys_export_report.sh <report.nsys-rep> [json|text]
# 例:   ./scripts/tests/nsys_export_report.sh nsys-report-ivf_tensor-SIFT1M-128-euclidean-20260221_205917.nsys-rep json

set -e
REP="${1:?用法: $0 <report.nsys-rep> [json|text]}"
FMT="${2:-json}"

if [[ "$FMT" != "json" && "$FMT" != "text" ]]; then
    echo "格式请选 json 或 text"
    exit 1
fi

OUT="${REP%.nsys-rep}.${FMT}"
echo "导出: $REP -> $OUT (格式: $FMT)"
nsys export --type="$FMT" --force-overwrite=true -o "$OUT" "$REP"
echo "已生成: $OUT"

if command -v nsys &>/dev/null; then
    echo ""
    echo "--- 报告统计摘要 (nsys stats) ---"
    nsys stats "$REP" 2>/dev/null || true
fi
