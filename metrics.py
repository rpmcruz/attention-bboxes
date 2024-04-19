import torchmetrics
import torch

class PointingGame(torchmetrics.Metric):
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
            self.correct += t[0, y, x] != 0
        self.total += len(preds)

    def compute(self):
        return self.correct / self.total

class GradCAM:
    # Selvaraju, Ramprasaath R., et al. "Grad-CAM: Why did you say that?."
    # arXiv preprint arXiv:1611.07450 (2016).
    def fhook(self, module, args, output):
        self.activation = output

    def __call__(self, model, layer, x, y):
        handle = layer.register_forward_hook(self.fhook)
        ypred = model(x)
        handle.remove()
        grad_out = torch.zeros_like(self.activation)
        grad = torch.autograd.grad([ypred[:, y].sum()], [self.activation], [grad_out])[0]
        alpha = torch.mean(grad, (2, 3), True)
        return torch.nn.functional.relu(torch.sum(alpha * self.activation, 1))

class DegradationScore(torchmetrics.Metric):
    def __init__(self, resnet, loss, score='acc'):
        self.model = resnet
        if score == 'acc':
            self.score = lambda ypred, y: (ypred == y).float().mean()
        else:
            raise Exception(f'Unknown score: {score}')
        self.add_state('areas', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, batch, y):
        x, y = batch
        e = GradCAM()(self.model, self.model.layer4[1].conv2, x, y)
        for b in range(len(x)):
            lerf = self.degradation_curve('lerf', self.model, self.score, x[b], y[b], e[b])
            morf = self.degradation_curve('morf', self.model, self.score, x[b], y[b], e[b])
            self.areas += (lerf - morf).mean()
        self.count += len(x)

    def compute(self):
        return self.areas / self.count

    def degradation_curve(self, curve_type, model, score_fn, x, y, e):
        # Given an explanation map, occlude by 8x8 creating two curves: least
        # relevant removed first (LeRF) and most relevant removed first (MoRF),
        # where the score is computed for a given metric. The result is the area
        # between the two curves.
        #
        # Schulz, Karl, et al. "Restricting the flow: Information bottlenecks for
        # attribution." arXiv preprint arXiv:2001.00396 (2020).
        #
        # e is the explanation map.
        # like in the paper occlude by 8x8 from the worst to the best (and
        # vice-versa)
        assert curve_type in ['lerf', 'morf']
        assert len(x.shape) == 3
        assert len(e.shape) == 2
        x = x.clone()
        descending = curve_type == 'morf'
        ix = torch.argsort(e.flatten(), descending=descending)
        xx = ix % e.shape[1]
        yy = ix // e.shape[1]
        xscale = x.shape[2] // e.shape[1]
        yscale = x.shape[1] // e.shape[0]
        curve = []
        for i in range(e.shape[0]*e.shape[1]-1):
            yslice = slice(yy[i]*yscale, (yy[i]+1)*yscale)
            xslice = slice(xx[i]*xscale, (xx[i]+1)*xscale)
            x[:, yslice, xslice] = 0
            with torch.no_grad():
                ypred = model(x[None])
                s = score_fn(ypred, y)
                curve.append(s)
        return torch.tensor(curve)
