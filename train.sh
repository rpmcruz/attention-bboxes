#!/bin/bash

DATASET=Birds
# #MODELS="OnlyClass Heatmap"
# MODELS=""
# for MODEL in $MODELS; do
# echo "model-$DATASET-$MODEL"
# python3 train.py model-$DATASET-$MODEL.pth $DATASET $MODEL
# done

MODEL=Heatmap
#python3 train.py model-$DATASET-$MODEL-softmax.pth $DATASET $MODEL
python3 train.py model-$DATASET-$MODEL-sigmoid.pth $DATASET $MODEL --sigmoid

# # softmax
# MODELS="SimpleDet"
# PENALTIES="0 0.001 0.1 1 10"
# for MODEL in $MODELS; do
# for PENALTY in $PENALTIES; do
# echo "model-$DATASET-$MODEL-$PENALTY"
# python3 train.py model-$DATASET-$MODEL-l1-$PENALTY-softmax.pth $DATASET $MODEL --penalty-l1 $PENALTY
# done
# done

# sigmoid
MODELS="SimpleDet"
PENALTIES="0 0.001 0.1 1 10"
for MODEL in $MODELS; do
for PENALTY in $PENALTIES; do
echo "model-$DATASET-$MODEL-$PENALTY"
python3 train.py model-$DATASET-$MODEL-l1-$PENALTY-sigmoid.pth $DATASET $MODEL --penalty-l1 $PENALTY --sigmoid
done
done
