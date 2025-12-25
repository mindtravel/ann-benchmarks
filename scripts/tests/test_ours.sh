./scripts/tests/compile.sh ours

# 用规定的三个数据集测试
for dataset in "sift-128-euclidean" "deep-image1M-96-angular" "text1M-200-angular" 

do
    ./scripts/tests/pgvector_ours.sh $dataset 

    python plot.py --x-scale linear --y-scale log --batch --no-pareto-frontier --dataset $dataset
done

