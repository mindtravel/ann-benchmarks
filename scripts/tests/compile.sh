# 设置正确的语言环境
export LC_ALL=C.utf8 && export LANG=C.utf8 && export LANGUAGE=C.utf8
# 停止冲突的postgresql实例
# pkill -f postgres
sudo service postgresql stop
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
elif  [ "$1" = "ours" ]; then \
    echo "编译ours pgvector"
    cd ../pgvector
    # g++ -std=c++17 -I/usr/include/postgresql -o cpp/vector_search cpp/parallel_flat.cpp -lpq
    make clean
    make
    make install
    # 替换PostgreSQL扩展
    sudo cp /home/zongxi/pgvector/vector.so /usr/lib/postgresql/16/lib/vector.so
else \
    echo "未知的编译选项"

fi

cd /home/zongxi/ann-benchmarks

# 重启PostgreSQL服务
sudo service postgresql restart
