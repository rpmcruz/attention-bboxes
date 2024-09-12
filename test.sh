#!/bin/bash

#DATASETS="Birds StanfordCars StanfordDogs"
DATASETS="$1"
PENALTIES="0 0.001 0.1 1 10"
XAI="CAM GradCAM DeepLIFT"
MODELS="SimpleDet FasterRCNN FCOS DETR"
HEATMAPS="GaussHeatmap LogisticHeatmap"
# only encoder occlusion due to time constrains
OCCLUSIONS="image encoder"

echo "model,dataset,xai,crop,acc,pg,degscore,sparsity"

for dataset in $DATASETS; do
# baseline: no heatmap (only class)
model="model-$dataset-OnlyClass.pth"
python3 test.py $model $dataset

# baseline: xai
for xai in $XAI; do
python3 test.py $model $dataset --xai $xai
done

# baseline: protopnet
model="model-$dataset-ProtoPNet.pth"
python3 test.py $model $dataset --protopnet
python3 test.py $model $dataset --protopnet --crop

# proposal (with ablation)
for penalty in $PENALTIES; do
    for occlusion in $OCCLUSIONS; do
        model="model-$dataset-Heatmap-l1-$penalty-occlusion-$occlusion-sigmoid.pth"
        python3 test.py $model $dataset
        model="model-$dataset-Heatmap-l1-$penalty-occlusion-$occlusion-sigmoid-adversarial.pth"
        python3 test.py $model $dataset
        for objdet in $MODELS; do
            for heatmap in $HEATMAPS; do
                model="model-$dataset-$objdet-l1-$penalty-heatmap-$heatmap-occlusion-$occlusion-sigmoid.pth"
                python3 test.py $model $dataset
                model="model-$dataset-$objdet-l1-$penalty-heatmap-$heatmap-occlusion-$occlusion-sigmoid-adversarial.pth"
                python3 test.py $model $dataset
            done
        done
    done
done
done  # $dataset