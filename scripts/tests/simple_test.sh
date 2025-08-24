dataset="glove-100-angular"

# ./scripts/tests/pgvector_origin.sh $dataset
./scripts/tests/pgvector_ours.sh $dataset
# ./scripts/tests/cuvs.sh $dataset
# TODO:添加pgvector的其他改良版本
python plot.py --x-scale linear --y-scale log --batch --dataset $dataset