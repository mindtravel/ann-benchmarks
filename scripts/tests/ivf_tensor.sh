# !/bin/bash
# ours ivftensor测试脚本（batch_size 在 ann_benchmarks/algorithms/ivf_tensor/config.yml 中配置）
# 用法: ./scripts/tests/ivf_tensor.sh <dataset>
echo "ours ivf_tensor测试开始..."

python run.py --local --algorithm ivf_tensor --dataset "$1" --force --runs 1 --batch

echo "ours ivf_tensor测试完成"

# 不重新计算，只想画图用这个命令
# python plot.py --x-scale linear --y-scale log --batch --dataset "SIFT1M-128-euclidean"
