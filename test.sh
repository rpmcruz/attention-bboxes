#!/bin/bash

DATASET=Birds
PENALTIES="0 0.001 0.1 1 10"

MODEL=OnlyClass
python3 test.py model-$DATASET-$MODEL.pth $DATASET --visualize

MODEL=ProtoPNet
python3 test.py model-$DATASET-$MODEL.pth $DATASET --visualize

MODEL=Heatmap
for PENALTY in $PENALTIES; do
python3 test.py model-$DATASET-$MODEL-l1-$PENALTY-softmax.pth $DATASET --visualize
python3 test.py model-$DATASET-$MODEL-l1-$PENALTY-sigmoid.pth $DATASET --visualize
done

MODEL="SimpleDet"
for PENALTY in $PENALTIES; do
python3 test.py model-$DATASET-$MODEL-l1-$PENALTY-softmax.pth $DATASET --visualize
python3 test.py model-$DATASET-$MODEL-l1-$PENALTY-sigmoid.pth $DATASET --visualize
done
