!/bin/bash

# 多卡多线程pgvector测试脚本
echo "多卡多线程pgvector测试开始..."

# 步骤1: 编译pgvector
echo "1: 编译pgvector..."
cd /home/zhangyi/pgvector
# g++ -std=c++17 -I/usr/include/postgresql -o cpp/vector_search cpp/parallel_flat.cpp -lpq
make
cd /home/zongxi/ann-benchmarks

# 步骤2: 替换PostgreSQL扩展
echo "2: 替换pgvector扩展..."
sudo cp /home/zhangyi/pgvector/vector.so /usr/lib/postgresql/16/lib/vector.so
sudo service postgresql restart

# 步骤3: 设置PostgreSQL多线程参数
echo "3: 配置PostgreSQL多线程参数..."
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET max_parallel_workers_per_gather = 20;"
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET max_parallel_workers = 20;"
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET parallel_tuple_cost = 0.1;"
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET parallel_setup_cost = 1000.0;"
sudo -u postgres psql -d ann -c "SELECT pg_reload_conf();"

# 步骤4: 设置环境变量并运行多线程测试
echo "4: 运行多线程ann-benchmark测试..."
export ANN_BENCHMARKS_PG_USER=postgres
export ANN_BENCHMARKS_PG_PASSWORD=
export ANN_BENCHMARKS_PG_DBNAME=ann
export ANN_BENCHMARKS_PG_START_SERVICE=false

# 使用多进程运行测试
# python run.py --local --algorithm pgvector_ivfflat_multi_ours --batch --force --runs 1
python run.py --local --algorithm pgvector_ivfflat_multi_ours --force --runs 1

echo "多卡多线程pgvector测试完成"


