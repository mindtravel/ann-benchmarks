#!/bin/bash

# 恢复原版pgvector扩展并测试脚本
echo "=== 恢复原版pgvector扩展并测试 ==="

# 步骤1: 恢复原版pgvector扩展
echo "1: 恢复原版pgvector扩展..."
# sudo apt-get update
sudo apt-get install -y postgresql-16-pgvector

# 步骤2: 重启PostgreSQL服务
echo "2: 重启PostgreSQL服务..."
sudo service postgresql restart

# 步骤3: 验证原版扩展
echo "3: 验证原版扩展..."
su - postgres -c "createdb ann"
sudo -u postgres psql -d ann -c "DROP EXTENSION IF EXISTS vector CASCADE;"
sudo -u postgres psql -d ann -c "CREATE EXTENSION vector;"
sudo -u postgres psql -d ann -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"

# 步骤4: 设置环境变量并运行原版测试
echo "4: 运行原版pgvector ann-benchmark测试..."
# cd /home/zongxi/ann-benchmarks
export ANN_BENCHMARKS_PG_USER=postgres
export ANN_BENCHMARKS_PG_PASSWORD=
export ANN_BENCHMARKS_PG_DBNAME=ann
export ANN_BENCHMARKS_PG_START_SERVICE=false

# python run.py --local --algorithm pgvector_ivfflat_multi_ours --batch --force --runs 1
python run.py --local --algorithm pgvector_ivfflat --batch --force --runs 1

echo "原版pgvector测试完成"






