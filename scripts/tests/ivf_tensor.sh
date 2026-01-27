# !/bin/bash
# ours pgvector测试脚本
echo "ours ivf_tensor测试开始..."

# 测试GPU版本
python run.py --local --algorithm ivf_tensor --dataset $1 --force --runs 1 --batch


echo "ours ivf_tensor测试完成"


