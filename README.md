# attention-bboxes
An in-model xAI technique based on predicting intermediate bounding boxes.

The most important part of the code is `models.py` (with the several modules being proposed) and `train.py` (that combines and trains them).

Possible approaches:

* Occlusion
```
                      ┌─────────┐               ┌──────────────┐                                 
         ┌───────┐    │detection│               │differentiable│                     ┌──────────┐
image───►│encoder├───►│head     ├────►bboxes───►│transformation├──► heatmap ──► ⊗ ──►│classifier│
         └───┬───┘    └─────────┘               └──────────────┘                ▲    └──────────┘
             │                                                                  │                
             └──────────────────────────────────────────────────────────────────┘                
```

Min_{encoder,detection,classifier} Loss

* Adversarial
```
         ┌───────┐                                                      ┌──────────┐
image───►│encoder├─────────────────────────────────────────────────────►│classifier│
         └───┬───┘                                                      └──────────┘
             │                                                                ▲     
             │       ┌─────────┐     ┌──────────────┐                         │     
             ├──────►│detection├────►│differentiable├───► 1-heatmap ──► ⊗ ────┘     
             │       │head     │     │transformation│                   ▲           
             │       └─────────┘     └──────────────┘                   │           
             │                                                          │           
             └──────────────────────────────────────────────────────────┘           
```

Min_{encoder,classifier} Loss ; Max_{detection} Loss
