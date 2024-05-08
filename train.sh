#!/bin/bash

DATASETS="Birds StanfordCars StanfordDogs"
EPOCHS=100
PENALTIES1="0 0.1 1"
PENALTIES2="0 0.0001 0.001"

for DATASET in $DATASETS; do
OUT="model-$DATASET-Baseline.pth"
if [ ! -f $OUT ]; then
    echo $OUT
    python3 train.py $DATASET Baseline $OUT --epochs $EPOCHS
fi

for PENALTY1 in $PENALTIES1; do
for PENALTY2 in $PENALTIES2; do
    echo "Train for penalty $PENALTY1 $PENALTY2"
    OUT="model-$DATASET-GaussianGrid-penalty-$PENALTY1-$PENALTY2-usesoftmax-0.pth"
    if [ ! -f $OUT ]; then
        echo $OUT
        python3 train.py $DATASET GaussianGrid $OUT --epochs $EPOCHS --penalty1 $PENALTY1 --penalty2 $PENALTY2 &
    fi
    OUT="model-$DATASET-GaussianGrid-penalty-$PENALTY1-$PENALTY2-usesoftmax-1.pth"
    if [ ! -f $OUT ]; then
        echo $OUT
        python3 train.py $DATASET GaussianGrid $OUT --epochs $EPOCHS --penalty1 $PENALTY1 --penalty2 $PENALTY2 --use-softmax &
    fi
    wait
done
done
done
