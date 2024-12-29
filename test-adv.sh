#!/bin/bash

#DATASETS="Birds StanfordCars StanfordDogs"
DATASETS="$1"
PENALTIES="0 0.001 0.01 0.1 1 10 100 1000"
XAI="CAM GradCAM DeepLIFT Occlusion IBA"
MODELS="SimpleDet FasterRCNN FCOS DETR"
HEATMAPS="GaussHeatmap LogisticHeatmap"
OCCLUSIONS="encoder"
#OCCLUSIONS="encoder image"

echo "model,dataset,xai,crop,acc,degscore,pg,density,totalvariance,entropy"

for DATASET in $DATASETS; do
if [ $DATASET = Birds ]; then NCLASSES="10 50 100 200"; fi
if [ $DATASET = StanfordCars ]; then NCLASSES="10 50 100 196"; fi
if [ $DATASET = StanfordDogs ]; then NCLASSES="10 50 80 120"; fi

# baseline: protopnet
MODEL="model-$DATASET-ProtoPNet.pth"
#python test.py $MODEL $DATASET --protopnet
#python test.py $MODEL $DATASET --protopnet --crop
# baseline: protopnet crop
MODEL="model-$DATASET-ProtoPNet-crop.pth"
#python test.py $MODEL $DATASET --protopnet
#python test.py $MODEL $DATASET --protopnet --crop

for NCLASS in $NCLASSES; do
# baseline: no heatmap (only class)
MODEL="model-$DATASET$NCLASS-OnlyClass.pth"
#python test.py $MODEL $DATASET$NCLASS

# baseline: xai
for xai in $XAI; do
  : #python test.py $MODEL "$DATASET$NCLASS" --xai $xai
done

# baseline: ViT
MODEL="model-$DATASET$NCLASS-ViTb.pth"
#python test.py $MODEL "$DATASET$NCLASS"
MODEL="model-$DATASET$NCLASS-ViTl.pth"
#python test.py $MODEL "$DATASET$NCLASS"
MODEL="model-$DATASET$NCLASS-ViTr.pth"
#python test.py $MODEL "$DATASET$NCLASS"

# proposal (with ablation)
for occlusion in $OCCLUSIONS; do
    for penalty in $PENALTIES; do
        MODEL="model-$DATASET$NCLASS-Heatmap-l1-$penalty-heatmap-none-occlusion-$occlusion-sigmoid-1.pth"
        #python test.py $MODEL "$DATASET$NCLASS"
        model="model-$DATASET$NCLASS-Heatmap-l1-$penalty-heatmap-none-occlusion-$occlusion-sigmoid-1-adversarial-1.pth"
        python test.py $MODEL "$DATASET$NCLASS"
        for objdet in $MODELS; do
            for heatmap in $HEATMAPS; do
                MODEL="model-$DATASET$NCLASS-$objdet-l1-$penalty-heatmap-$heatmap-occlusion-$occlusion-sigmoid-1.pth"
                #python test.py $MODEL "$DATASET$NCLASS"
                MODEL="model-$DATASET$NCLASS-$objdet-l1-$penalty-heatmap-$heatmap-occlusion-$occlusion-sigmoid-1-adversarial-1.pth"
                python test.py $MODEL "$DATASET$NCLASS"
            done
        done
    done
done

done
done  # $dataset
