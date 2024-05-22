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
        preds = preds.view(len(preds), -1)
        target = target.view(len(target), -1)
        self.correct += sum(target[range(len(target)), torch.argmax(preds, 1)] != 0)
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
    def __init__(self, model, score='acc'):
        super().__init__()
        self.model = model
        if score == 'acc':
            self.score = lambda ypred, y: (ypred == y).float()
        else:
            raise Exception(f'Unknown score: {score}')
        self.add_state('areas', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, images, true_classes, heatmaps):
        for image, true_class, heatmap in zip(images, true_classes, heatmaps):
            lerf = self.degradation_curve('lerf', self.model, self.score, image, true_class, heatmap)
            morf = self.degradation_curve('morf', self.model, self.score, image, true_class, heatmap)
            self.areas += (lerf - morf).mean()
        self.count += len(images)

    def compute(self):
        return self.areas / self.count

    def degradation_curve(self, curve_type, model, score_fn, image, true_class, heatmap):
        # Given an explanation map, occlude by 8x8 creating two curves: least
        # relevant removed first (LeRF) and most relevant removed first (MoRF),
        # where the score is computed for a given metric. The result is the area
        # between the two curves.
        # Schulz, Karl, et al. "Restricting the flow: Information bottlenecks for
        # attribution." arXiv preprint arXiv:2001.00396 (2020).
        assert curve_type in ['lerf', 'morf']
        assert len(image.shape) == 3
        assert len(heatmap.shape) == 3 and heatmap.shape[0] == 1
        heatmap = heatmap[0]
        descending = curve_type == 'morf'
        ix = torch.argsort(heatmap.view(-1), descending=descending)[:-1]
        cc = ix % heatmap.shape[1]
        rr = ix // heatmap.shape[1]
        xscale = image.shape[2] // heatmap.shape[1]
        yscale = image.shape[1] // heatmap.shape[0]
        # create occlusions covered by the heatmap
        occlusions = image[None].repeat(len(ix), 1, 1, 1)
        for i, (c, r) in enumerate(zip(cc, rr)):
            occlusions[i:, c*yscale:(c+1)*yscale, r*xscale:(r+1)*xscale] = 0
        with torch.no_grad():
            ypred = model(occlusions)['class'].argmax(1)
        return score_fn(ypred, true_class)
