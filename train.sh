#!/bin/bash
PENALTIES1="0 0.01 0.1 1 10 100"
PENALTIES2="0 0.01 0.1 1 10 100"
for PENALTY1 in $PENALTIES1; do
for PENALTY2 in $PENALTIES2; do
    echo "Train for penalty $PENALTY1 $PENALTY2"
    python3 train.py Birds GaussianGrid --epochs 100 --penalty1 $PENALTY1 --penalty2 $PENALTY2
    python3 train.py Birds GaussianGrid --epochs 100 --penalty1 $PENALTY1 --penalty2 $PENALTY2 --use-softmax
done
done
