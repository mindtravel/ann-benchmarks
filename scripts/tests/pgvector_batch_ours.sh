#!/bin/bash

# 专门测试 PGVectorSingle 的 Ours 版本
# 现在通过 ann_benchmarks.algorithms.pgvector_parallel.PGVectorSingle 实现
# 使用 pgvector 扩展中的 batch_vector_search SQL 进行批量查询

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <dataset> (e.g. glove-100-angular)"
  exit 1
fi

DATASET="$1"

echo "[single-ours-parallel] 编译带 batch_vector_search 的 pgvector 扩展..."
./scripts/tests/compile.sh ours

echo "[single-ours-parallel] 创建数据库 ann 并重建 vector 扩展..."
su - postgres -c "createdb ann" || true
sudo -u postgres psql -d ann -c "DROP EXTENSION IF EXISTS vector CASCADE;"
sudo -u postgres psql -d ann -c "CREATE EXTENSION vector;"
# 增加内存限制以支持批量查询
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET work_mem = '2GB';"
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET maintenance_work_mem = '2GB';"
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET shared_buffers = '4GB';"
sudo service postgresql restart

# ann-benchmarks 连接参数
export ANN_BENCHMARKS_PG_USER=postgres
export ANN_BENCHMARKS_PG_PASSWORD=
export ANN_BENCHMARKS_PG_DBNAME=ann
export ANN_BENCHMARKS_PG_START_SERVICE=false

echo "[single-ours-parallel] 激活 conda 环境并运行 pgvector_ivfflat_batch (PGVectorSingle-parallel + batch_vector_search) 测试..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ann-benchmarks

# 运行测试
python run.py \
  --local \
  --algorithm pgvector_ivfflat_batch \
  --dataset "${DATASET}" \
  --batch \
  --runs 1 \
  --force

echo ""
echo "[single-ours-parallel] 测试完成，正在分析结果..."

# 显示测试结果
python -c "
import h5py
import numpy as np
import os
import glob

# 查找最新的结果文件
result_dir = './results/${DATASET}/10/pgvector_ivfflat_batch-batch/'
if os.path.exists(result_dir):
    files = glob.glob(result_dir + '*version_Ours*.hdf5')
    files.sort(key=os.path.getmtime, reverse=True)
    
    print('=== 测试结果摘要 ===')
    for i, file in enumerate(files[:2]):  # 显示最新的两个结果
        filename = os.path.basename(file)
        # 从文件名提取参数
        parts = filename.replace('.hdf5', '').split('_')
        probes = parts[-1]
        
        with h5py.File(file, 'r') as f:
            if 'times' in f:
                times = f['times'][:]
                avg_time = np.mean(times)
                median_time = np.median(times)
                print(f'Probes={probes}: 平均查询时间={avg_time:.4f}s ({avg_time*1000:.2f}ms), 中位数={median_time:.4f}s ({median_time*1000:.2f}ms)')
            
            if 'recalls' in f:
                recalls = f['recalls'][:]
                avg_recall = np.mean(recalls)
                print(f'Probes={probes}: 平均召回率={avg_recall:.4f} ({avg_recall*100:.2f}%)')
            print()
else:
    print('未找到结果文件')
"

# 生成性能图表
echo "[single-ours-parallel] 生成性能图表..."
python plot.py \
  --dataset "${DATASET}" \
  --output results/pgvector_batch_ours_${DATASET}.png \
  --batch \
  -x k-nn \
  -y qps

echo "[single-ours-parallel] 图表已保存到 results/pgvector_batch_ours_${DATASET}.png"
echo "[single-ours-parallel] 所有任务完成！"
