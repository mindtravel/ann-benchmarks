# 用甲方给的三个数据集测试（Text还不支持）
for dataset in "sift-128-euclidean" "deep-image-96-angular" 
# "text10m-200-euclidean"

do
    # 测试原版pgvector
    # ./scripts/tests/pgvector_origin.sh $dataset

    # 测试我们的pgvector
    ./scripts/tests/pgvector_ours.sh $dataset

    # 测试cuvs
    # ./scripts/tests/cuvs.sh $dataset

    # TODO:添加pgvector的其他改良版本
    
    python plot.py --x-scale linear --y-scale log --batch --dataset $dataset
done