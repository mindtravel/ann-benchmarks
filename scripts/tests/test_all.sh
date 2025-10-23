# 用合同中的三个数据集测试
for dataset in "TEXT1M-200-angular" "sift-128-euclidean" "deep-image-96-angular" 
do
    # 测试原版pgvector
    ./scripts/tests/pgvector_origin.sh $dataset
    
    python plot.py --x-scale linear --y-scale log --batch --dataset $dataset
done