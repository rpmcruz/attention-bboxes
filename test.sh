#!/bin/bash

DATASET=Birds
PENALTIES="0 0.1 1 100"

python3 test.py $DATASET "model-$DATASET-Baseline.pth"

# without softmax
for PENALTY1 in $PENALTIES; do
for PENALTY2 in $PENALTIES; do
    python3 test.py $DATASET "model-$DATASET-GaussianGrid-penalty-$PENALTY1-$PENALTY2-usesoftmax-0.pth"
done
done

# with softmax
for PENALTY1 in $PENALTIES; do
for PENALTY2 in $PENALTIES; do
    python3 test.py $DATASET "model-$DATASET-GaussianGrid-penalty-$PENALTY1-$PENALTY2-usesoftmax-1.pth"
done
done
