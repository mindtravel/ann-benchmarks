# !/bin/bash
# ours pgvector测试脚本
echo "ours pgvector测试开始..."

# ./scripts/tests/compile.sh ours
# ./scripts/tests/compile.sh baseline

# 步骤3: 设置PostgreSQL多线程参数
echo "3: 配置PostgreSQL多线程参数..."
sudo -u postgres psql -d ann -c "DROP EXTENSION IF EXISTS vector CASCADE;"
sudo -u postgres psql -d ann -c "CREATE EXTENSION vector;"
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET max_parallel_workers_per_gather = 20;"
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET max_parallel_workers = 20;"
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET parallel_tuple_cost = 0.1;"
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET parallel_setup_cost = 1000.0;"

# 设置日志配置
echo "设置日志配置..."
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET log_min_messages = 'info';"
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET log_destination = 'stderr';"
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET logging_collector = on;"
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET log_filename = 'postgresql.log';"
sudo -u postgres psql -d ann -c "SELECT pg_reload_conf();"
sudo service postgresql restart

# 验证日志配置是否生效
echo "验证日志配置..."
sudo -u postgres psql -d ann -c "SHOW log_min_messages;"
sudo -u postgres psql -d ann -c "SHOW log_destination;"
sudo -u postgres psql -d ann -c "SHOW logging_collector;"

# 验证vector_array类型是否存在
echo "验证vector_array类型..."
sudo -u postgres psql -d ann -c "SELECT typname, oid FROM pg_type WHERE typname = 'vector_batch';"

# 步骤4: 设置环境变量并运行多线程测试
echo "4: 运行多线程ann-benchmark测试..."
export ANN_BENCHMARKS_PG_USER=postgres
export ANN_BENCHMARKS_PG_PASSWORD=
export ANN_BENCHMARKS_PG_DBNAME=ann
export ANN_BENCHMARKS_PG_START_SERVICE=false

# 使用多进程运行测试
python run.py --local --algorithm pgvector_ivfflat_gpu --dataset $1 --force --runs 1 --batch
# python run.py --local --algorithm pgvector_ivfflat_ours --dataset $1 --force --runs 1 --batch
# python run.py --local --algorithm pgvector_ivfjl_ours --dataset $1 --force --runs 1 --batch

echo "ours pgvector测试完成"


