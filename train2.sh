#!/bin/bash
# train different model types

DATASET=$1
MODEL=$2
HEATMAP=$3
OCCLUSION=$4
PENALTIES="0 0.001 0.01 0.1 1 10"

# non-adversarial
RESUME=none
EPOCHS=100
for PENALTY in $PENALTIES; do
NAME="model-$DATASET-$MODEL-l1-$PENALTY-heatmap-$HEATMAP-occlusion-$OCCLUSION-sigmoid.pth"
if [ ! -f $NAME ]; then
echo $NAME
python train.py $NAME $DATASET $MODEL --l1 $PENALTY --heatmap $HEATMAP --occlusion $OCCLUSION --sigmoid --epochs $EPOCHS --resume $RESUME
fi
RESUME=$NAME
EPOCHS=50
done

# adversarial
RESUME=none
EPOCHS=100
for PENALTY in $PENALTIES; do
NAME="model-$DATASET-$MODEL-l1-$PENALTY-heatmap-$HEATMAP-occlusion-$OCCLUSION-sigmoid-adversarial.pth"
if [ ! -f $NAME ]; then
echo $NAME
python train.py $NAME $DATASET $MODEL --l1 $PENALTY --heatmap $HEATMAP --occlusion $OCCLUSION --sigmoid --epochs $EPOCHS --resume $RESUME --adversarial
fi
RESUME=$NAME
EPOCHS=50
done
