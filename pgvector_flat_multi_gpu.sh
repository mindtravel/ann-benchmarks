#!/bin/bash

# 多卡多线程pgvector GPU测试脚本
echo "=== 多卡多线程pgvector GPU测试开始 ==="

# 检查GPU可用性
echo "检查GPU状态..."
nvidia-smi > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "GPU可用，将启用GPU加速"
    GPU_AVAILABLE=true
else
    echo "GPU不可用，将使用CPU多线程"
    GPU_AVAILABLE=false
fi

# 步骤1: 编译pgvector
echo "1: 编译pgvector..."
cd /home/zhangyi/pgvector
g++ -std=c++17 -I/usr/include/postgresql -o cpp/vector_search cpp/parallel_flat.cpp -lpq

# 步骤2: 替换PostgreSQL扩展
echo "2: 替换pgvector扩展..."
sudo cp /usr/lib/postgresql/16/lib/vector.so /usr/lib/postgresql/16/lib/vector.so.backup
sudo cp /home/zhangyi/pgvector/vector.so /usr/lib/postgresql/16/lib/vector.so
sudo service postgresql restart

# 步骤3: 配置PostgreSQL多线程参数
echo "3: 配置PostgreSQL多线程参数..."
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET max_parallel_workers_per_gather = 8;"
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET max_parallel_workers = 16;"
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET parallel_tuple_cost = 0.1;"
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET parallel_setup_cost = 1000.0;"
sudo -u postgres psql -d ann -c "SELECT pg_reload_conf();"

# 步骤4: 设置环境变量
echo "4: 设置环境变量..."
cd /home/zongxi/ann-benchmarks
export ANN_BENCHMARKS_PG_USER=postgres
export ANN_BENCHMARKS_PG_PASSWORD=
export ANN_BENCHMARKS_PG_DBNAME=ann
export ANN_BENCHMARKS_PG_START_SERVICE=false

# 步骤5: 运行GPU测试
echo "5: 运行GPU测试..."

if [ "$GPU_AVAILABLE" = true ]; then
    # 强制使用GPU配置
    echo "运行GPU配置: lists-100-workers-4-gpu"
    python run.py --local --algorithm pgvector_ivfflat_multi_ours --batch --force --runs 1
    echo "运行GPU配置: lists-100-workers-8-gpu"
    python run.py --local --algorithm pgvector_ivfflat_multi_ours --batch --force --runs 1
else
    echo "运行CPU配置: lists-100-workers-4"
    python run.py --local --algorithm pgvector_ivfflat_multi_ours --batch --force --runs 1
    
    echo "运行CPU配置: lists-100-workers-8"
    python run.py --local --algorithm pgvector_ivfflat_multi_ours --batch --force --runs 1
fi

echo "=== 多卡多线程pgvector GPU测试完成 ==="

