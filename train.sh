#!/bin/bash
PENALTIES="0 0.1 1 10"
for PENALTY in $PENALTIES; do
    echo "Train for penalty $PENALTY"
    python3 train.py GaussianGrid --epochs 100 --penalty $PENALTY
done
