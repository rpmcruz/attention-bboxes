from torchmetrics import Metric
import torch

class PointingGame(Metric):
    # https://link.springer.com/article/10.1007/s11263-017-1059-x
    def __init__(self):
        super().__init__()
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        preds = torch.nn.functional.interpolate(preds, target.shape[-2:], mode='nearest-exact')
        for p, t in zip(preds, target):
            i = p.argmax()
            x = i % p.shape[1]
            y = i // p.shape[1]
            self.correct += t[0, y, x]
        self.total += len(preds)

    def compute(self):
        return self.correct / self.total
