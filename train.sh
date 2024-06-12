#!/bin/bash

python3 train.py model-vit.pth Birds --vit --epochs 10
python3 train.py models-fasterrcnn-gauss.pth Birds --detection FasterRCNN --heatmap GaussHeatmap --epochs 10
