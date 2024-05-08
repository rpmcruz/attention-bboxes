#!/bin/bash

DATASETS="Birds StanfordCars StanfordDogs"
PENALTIES1="0 0.1 1"
PENALTIES2="0 0.0001 0.001"

for DATASET in $DATASETS; do
NAME="model-$DATASET-Baseline.pth"
python3 test.py $DATASET $NAME

# without softmax
for PENALTY1 in $PENALTIES1; do
for PENALTY2 in $PENALTIES2; do
    NAME="model-$DATASET-GaussianGrid-penalty-$PENALTY1-$PENALTY2-usesoftmax-0.pth"
    python3 test.py $DATASET $NAME
done
done

# with softmax
for PENALTY1 in $PENALTIES1; do
for PENALTY2 in $PENALTIES2; do
    NAME="model-$DATASET-GaussianGrid-penalty-$PENALTY1-$PENALTY2-usesoftmax-1.pth"
    python3 test.py $DATASET $NAME
done
done

done
