# 编译原版pgvector或ours pgvector

if [ "$1" = "baseline" ]; then \
    # 编译恢复原版pgvector扩展
    # sudo apt-get update
    # sudo apt-get install -y postgresql-16-pgvector
    ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        OPTFLAGS="-march=native -msve-vector-bits=512"; \
    elif [ "$ARCH" = "x86_64" ]; then \
        OPTFLAGS="-march=native -mprefer-vector-width=512"; \
    else \
        OPTFLAGS="-march=native"; \
    fi && \
    cd /tmp/pgvector && \
    make clean && \
    make OPTFLAGS="$OPTFLAGS" && \
    make install
    cd /home/zongxi/ann-benchmarks
elif  [ "$1" = "ours" ]; then \
    echo "编译ours pgvector"
    cd ../pgvector
    # g++ -std=c++17 -I/usr/include/postgresql -o cpp/vector_search cpp/parallel_flat.cpp -lpq
    make
    cd ../ann-benchmarks
    # 替换PostgreSQL扩展
    sudo cp ../pgvector/vector.so /usr/lib/postgresql/16/lib/vector.so
    sudo service postgresql restart
else \
    echo "未知的编译选项"

fi

# 重启PostgreSQL服务
sudo service postgresql restart
