#!/bin/bash
# train different model types

DATASET=$1
MODEL=$2
HEATMAP=$3
OCCLUSION=$4
PENALTIES="0 0.001 0.01 0.1 1 10 100 1000"

# non-adversarial
RESUME=none
EPOCHS=1000
for PENALTY in $PENALTIES; do
NAME="model-$DATASET-$MODEL-l1-$PENALTY-heatmap-$HEATMAP-occlusion-$OCCLUSION-sigmoid-1.pth"
if [ ! -f $NAME ]; then
echo $NAME
python train.py $NAME $DATASET $MODEL --l1 $PENALTY --heatmap $HEATMAP --occlusion $OCCLUSION --epochs $EPOCHS --resume $RESUME
else
echo "already exists: $NAME"
fi
RESUME=$NAME
EPOCHS=100
done

# adversarial
RESUME=none
EPOCHS=1000
for PENALTY in $PENALTIES; do
NAME="model-$DATASET-$MODEL-l1-$PENALTY-heatmap-$HEATMAP-occlusion-$OCCLUSION-sigmoid-1-adversarial-1.pth"
if [ ! -f $NAME ]; then
echo $NAME
python train.py $NAME $DATASET $MODEL --l1 $PENALTY --heatmap $HEATMAP --occlusion $OCCLUSION --epochs $EPOCHS --resume $RESUME --adversarial
fi
RESUME=$NAME
EPOCHS=100
done
