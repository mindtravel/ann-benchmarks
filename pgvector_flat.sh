#!/bin/bash

# 步骤1: 从源码编译pgvector
echo "1: 编译pgvector..."
cd /home/zhangyi/pgvector
g++ -std=c++17 -I/usr/include/postgresql -o cpp/vector_search cpp/parallel_flat.cpp -lpq

# 步骤2: 替换PostgreSQL的pgvector扩展
echo "2: 替换pgvector扩展..."
sudo cp /usr/lib/postgresql/16/lib/vector.so /usr/lib/postgresql/16/lib/vector.so.backup
sudo cp /home/zhangyi/pgvector/vector.so /usr/lib/postgresql/16/lib/vector.so
sudo service postgresql restart

# 步骤3: 设置环境变量并运行ann-benchmark测试
echo "3: 运行ann-benchmark测试..."
cd /home/zongxi/ann-benchmarks
export ANN_BENCHMARKS_PG_USER=postgres
export ANN_BENCHMARKS_PG_PASSWORD=
export ANN_BENCHMARKS_PG_DBNAME=ann
export ANN_BENCHMARKS_PG_START_SERVICE=false

python run.py --local --algorithm pgvector_ivfflat_ours --batch --force --runs 1

echo "pgvector_flat测试完成"

