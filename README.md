# attention-bboxes
An in-model xAI technique based on predicting intermediate bounding boxes.

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

**Baselines:**
* [Data-efficient and weakly supervised computational pathology on whole-slide images](https://www.nature.com/articles/s41551-020-00682-w) (CLAM)
* [Black-box Explanation of Object Detectors via Saliency Maps](https://www.computer.org/csdl/proceedings-article/cvpr/2021/450900l1438/1yeLVLXB5289) (D-RISE)
* ProtoPNet
* ViT

**Other interesting papers:**
* [Beyond the Label Itself](https://arxiv.org/abs/2312.08234) proposes a formula to convert bounding boxes to heatmaps in a differentiable manner.
