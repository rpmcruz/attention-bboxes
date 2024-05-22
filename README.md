# attention-bboxes
An in-model xAI technique based on predicting intermediate bounding boxes.

Possible approaches:

* Occlusion
```
                                ┌──────────────┐                                 
         ┌───────┐              │differentiable│                     ┌──────────┐
image───►│encoder├──► bboxes───►│transformation├──► heatmap ──► ⊗ ──►│classifier│
         └───┬───┘              └──────────────┘                ▲    └──────────┘
             │                                                  │                
             └──────────────────────────────────────────────────┘                  
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

Preliminary results using BasicGrid (dataset=STL10):

![Preliminary results](results/basicgrid-1.png)

Preliminary results using GaussianGridGrid using top-5 bounding boxes and confidence=1.5σ (dataset=STL10):

![Preliminary results](results/gaussiangrid-1.png)

![Preliminary results](results/gaussiangrid-2.png)

![Preliminary results](results/gaussiangrid-3.png)

Issues: (1) it errs a lot (but that's not an issue with the model itself); (2) the spatial scores are on the low side; (3) how to measure how good an explanation is?
