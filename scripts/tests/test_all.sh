# 用甲方给的三个数据集测试
# for dataset in "sift-128-euclidean" "deep-image-96-angular" "TEXT500k-200-angular"
# for dataset in "Deep-image1M-96-angular" "TEXT500k-200-angular"
# for dataset in "SIFT10M-128-euclidean"  # 测试 SIFT1B 数据集（1M 规模）
for dataset in "SIFT100M-128-euclidean"  # 测试 SIFT 数据集

do
    # 测试原版pgvector
    # ./scripts/tests/pgvector_origin.sh $dataset

    # 测试我们的pgvector
    # ./scripts/tests/pgvector_ours.sh $dataset
    
    # 测试ivf_tensor
    ./scripts/tests/ivf_tensor.sh $dataset

    # 测试cuvs
    # ./scripts/tests/cuvs.sh $dataset

    # TODO:添加pgvector的其他改良版本
    
    python plot.py --x-scale linear --y-scale log --batch --dataset $dataset
done