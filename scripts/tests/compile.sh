# 编译原版pgvector或ours pgvector

# 设置正确的语言环境
export LC_ALL=C.utf8 && export LANG=C.utf8 && export LANGUAGE=C.utf8
# 停止冲突的postgresql实例
# pkill -f postgres
sudo service postgresql stop

if [ "$1" = "baseline" ]; then \
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
    cd ../pgvector-baseline && \
    make OPTFLAGS="$OPTFLAGS"
elif  [ "$1" = "ours" ]; then \
    cd ../pgvector

else \
    echo "未知的编译选项"
fi  

make clean
make
make install

# 替换PostgreSQL扩展
if [ "$1" = "ours" ]; then \
    sudo cp ../pgvector/vector.so /usr/lib/postgresql/16/lib/vector.so
fi

cd ../ann-benchmarks

# 重启PostgreSQL服务
sudo service postgresql restart
