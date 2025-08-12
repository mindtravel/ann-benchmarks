conda activate ann
# 测试原版pgvector
./pgvector_flat_origin.sh
# 测试改良版pgvector（单线程）
./pgvector_flat.sh
# 测试改良版pgvector（多线程）
./pgvector_flat_multi.sh

