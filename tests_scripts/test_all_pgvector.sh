conda activate ann
cd /home/zongxi/ann-benchmarks/scripts_for_tests
# 测试原版pgvector
# ./pgvector_flat_origin.sh
# 测试改良版pgvector（多线程）
./pgvector_flat_multi.sh

# 切换到根目录运行plot.py
cd /home/zongxi/ann-benchmarks
python plot.py --x-scale logit --y-scale log