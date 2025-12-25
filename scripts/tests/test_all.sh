# 用于训练所有的三个数据集
# 测试原版pgvector: 编译选项baseline，循环内取消pgvector_origin.sh注释
# 测试我们优化的pgvector: 编译选项oues，循环内取消pgvector_ours.sh注释

./scripts/tests/compile.sh ours # 可选 ours, baseline

# 用规定的三个数据集测试
for dataset in "sift-128-euclidean" "deep-image1M-96-angular" "text1M-200-angular" "sift-128-euclidean" "sift-128-euclidean" 

do
    # 测试原版pgvector
    # ./scripts/tests/pgvector_origin.sh $dataset

    # 测试我们的pgvector
    ./scripts/tests/pgvector_ours.sh $dataset 

    python plot.py --x-scale linear --y-scale log --batch --no-pareto-frontier --dataset $dataset
done

