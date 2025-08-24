for dataset in "sift-128-euclidean" "deep-image-96-angular" 
# "text10m-200-euclidean"
do
    # 测试原版pgvector
    ./scripts/tests/pgvector_ivfflat.sh $dataset
    # ./scripts/tests/pgvector_ivfflat_ours.sh $dataset
    ./scripts/tests/cuvs.sh $dataset
    # TODO:添加pgvector的其他改良版本
    python plot.py --x-scale linear --y-scale log --batch --dataset $dataset
done