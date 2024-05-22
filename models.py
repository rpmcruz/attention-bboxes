import torch
import torchvision

def cxcywh_to_xyxy(bboxes):
    return torch.stack((
        torch.clamp(bboxes[:, 0] - bboxes[:, 2]/2, min=0),
        torch.clamp(bboxes[:, 1] - bboxes[:, 3]/2, min=0),
        torch.clamp(bboxes[:, 0] + bboxes[:, 2]/2, max=1),
        torch.clamp(bboxes[:, 1] + bboxes[:, 3]/2, max=1),
    ), 1)

####################### OBJ DETECT MODELS #######################
# the bounding boxes are of the type xyxy (normalized 0-1)

class OneStage(torch.nn.Module):
    def __init__(self, use_softmax=True, bboxes_normalized=False):
        super().__init__()
        self.bboxes = torch.nn.LazyConv2d(4, 1)
        self.scores = torch.nn.LazyConv2d(1, 1)
        self.use_softmax = use_softmax
        self.bboxes_normalized = bboxes_normalized

    def forward(self, grid):
        device = grid.device
        bboxes = self.bboxes(grid)
        scores = self.scores(grid)
        scores = torch.softmax(scores, 1) if self.use_softmax else torch.sigmoid(scores)
        if self.bboxes_normalized:
            bboxes = torch.sigmoid(bboxes)
            xstep = 1/grid.shape[3]
            ystep = 1/grid.shape[2]
            xx = torch.arange(0, 1, xstep, device=device)
            yy = torch.arange(0, 1, ystep, device=device)
        else:
            bboxes = torch.stack((
                torch.sigmoid(bboxes[:, 0]), torch.sigmoid(bboxes[:, 1]),
                torch.nn.functional.softplus(bboxes[:, 2]),
                torch.nn.functional.softplus(bboxes[:, 3]),
            ), 1)
            xstep = ystep = 1
            xx = torch.arange(grid.shape[3], device=device)
            yy = torch.arange(grid.shape[2], device=device)
        xx, yy = torch.meshgrid(xx, yy, indexing='xy')
        bboxes = torch.stack((
            xx + bboxes[:, 0]*xstep,
            yy + bboxes[:, 1]*ystep,
            bboxes[:, 2], bboxes[:, 3]
        ), 1)
        return torch.flatten(bboxes, 2), torch.flatten(scores, 1)

class FCOS(torch.nn.Module):
    # https://arxiv.org/abs/1904.01355
    # TODO: this model still needs to be adapted
    def __init__(self):
        super().__init__()
        backbone = torchvision.models.resnet50(weights='DEFAULT')
        backbone = list(backbone.children())[:-2]
        self.backbone = torch.nn.Sequential(*backbone[:-3])
        # C3, C4, C5 = the last 3 layers of the backbone
        # P3 = conv(C3, 1x1) + upsample(P4, 2)
        # P4 = conv(C4, 1x1) + upsample(P5, 2)
        # P5 = conv(C5, 1x1)
        # P6 = conv(P5, 1x1, stride=2)
        # P7 = conv(P6, 1x1, stride=2)
        self.C3 = backbone[-3]
        self.C4 = backbone[-2]
        self.C5 = backbone[-1]
        self.P3 = torch.nn.Conv2d(512, 256, 1)
        self.P4 = torch.nn.Conv2d(1024, 256, 1)
        self.P5 = torch.nn.Conv2d(2048, 256, 1)
        self.P6 = torch.nn.Conv2d(256, 256, 1, stride=2)
        self.P7 = torch.nn.Conv2d(256, 256, 1, stride=2)
        # head is shared across feature levels
        # we use class_head as our score_head
        self.class_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.Conv2d(256, 1, 3, padding=1),
        )
        self.reg_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.Conv2d(256, 4, 3, padding=1),
        )
        # instead of exp(x), use exp(s_ix) with a trainable scalar s_i for each P_i
        self.s = torch.nn.parameter.Parameter(torch.ones([5]))

    def forward(self, x):
        C2 = self.backbone(x)
        C3 = self.C3(C2)
        C4 = self.C4(C3)
        C5 = self.C5(C4)
        upsample = torch.nn.functional.interpolate
        P5 = self.P5(C5)
        P6 = self.P6(P5)
        P7 = self.P7(P6)
        P4 = self.P4(C4) + upsample(P5, scale_factor=2, mode='nearest-exact')
        P3 = self.P3(C3) + upsample(P4, scale_factor=2, mode='nearest-exact')
        return (self.class_head(P3), torch.exp(self.s[0]*self.reg_head(P3))), \
            (self.class_head(P4), torch.exp(self.s[1]*self.reg_head(P4))), \
            (self.class_head(P5), torch.exp(self.s[2]*self.reg_head(P5))), \
            (self.class_head(P6), torch.exp(self.s[3]*self.reg_head(P6))), \
            (self.class_head(P7), torch.exp(self.s[4]*self.reg_head(P7)))

class DETR(torch.nn.Module):
    # https://github.com/facebookresearch/detr
    def __init__(self, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6, bboxes_normalized=False):
        super().__init__()
        self.conv = torch.nn.LazyConv2d(hidden_dim, 1)
        self.transformer = torch.nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, batch_first=True)
        self.bboxes = torch.nn.Linear(hidden_dim, 4)
        self.query_pos = torch.nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = torch.nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = torch.nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.bboxes_normalized = bboxes_normalized

    def forward(self, x):
        h = self.conv(x)
        N, _, H, W = h.shape
        pos = torch.cat([
            self.col_embed[None, :W].repeat(H, 1, 1),
            self.row_embed[:H, None].repeat(1, W, 1),
        ], -1).flatten(0, 1)[None]
        h = self.transformer(pos + h.flatten(2).permute(0, 2, 1), self.query_pos[None].repeat(N, 1, 1))
        bboxes = torch.sigmoid(self.bboxes(h))
        # assume it predicts directly xyxy
        # ensure that 01 is to the left of 23
        # predicted bounding boxes are in cxcywh format
        bboxes = self.bboxes(h)
        if self.bboxes_normalized:
            bboxes = torch.sigmoid(bboxes)
        else:
            bboxes = torch.stack((
                torch.sigmoid(bboxes[:, 0]), torch.sigmoid(bboxes[:, 1]),
                torch.nn.functional.softplus(bboxes[:, 2]),
                torch.nn.functional.softplus(bboxes[:, 3]),
            ), 1)
        return bboxes, None

############################# GLUE #############################

class OcclusionModel(torch.nn.Module):
    def __init__(self, backbone, classifier, object_detection, bboxes2heatmap, occlusion_level, is_adversarial):
        super().__init__()
        self.backbone = backbone
        self.object_detection = object_detection
        self.bboxes2heatmap = bboxes2heatmap
        self.classifier = classifier
        self.occlusion_level = occlusion_level
        self.is_adversarial = is_adversarial

    def forward(self, images):
        embed = self.backbone(images)
        if self.occlusion_level == 'none':
            return {'class': self.classifier(embed)}
        bboxes, scores = self.object_detection(embed)
        heatmap_shape = embed.shape[2:] if self.occlusion_level == 'encoder' else images.shape[2:]
        heatmap = self.bboxes2heatmap(heatmap_shape, bboxes, scores)
        if scores != None:
            bboxes = [bb[:, ss >= 0.5] for bb, ss in zip(bboxes, scores)]
        if self.is_adversarial:
            if self.occlusion_level == 'encoder':
                min_embed = heatmap * embed
                max_embed = (1-heatmap) * embed
            else:  # image
                min_embed = self.backbone(heatmap * images)
                max_embed = self.backbone((1-heatmap) * images)
            return {
                'class': self.classifier(embed),
                'min_class': self.classifier(min_embed),
                'max_class': self.classifier(max_embed),
                'heatmap': heatmap, 'bboxes': bboxes
            }
        if self.occlusion_level == 'encoder':
            embed = heatmap * embed
            return {
                'class': self.classifier(embed),
                'heatmap': heatmap, 'bboxes': bboxes
            }

class SimpleModel(OcclusionModel):
    def __init__(self, backbone, classifier):
        super().__init__(backbone, classifier, None, None, 'none', False)

class ImageOcclusionModel(OcclusionModel):
    def __init__(self, backbone, classifier, object_detection, bboxes2heatmap, is_adversarial):
        super().__init__(backbone, classifier, object_detection, bboxes2heatmap, 'image', is_adversarial)

class EncoderOcclusionModel(OcclusionModel):
    def __init__(self, backbone, classifier, object_detection, bboxes2heatmap, is_adversarial):
        super().__init__(backbone, classifier, object_detection, bboxes2heatmap, 'encoder', is_adversarial)

def Backbone():
    # image 96x96 ----> grid 6x6 if [:-3]
    # image 224x224 --> grid 7x7 if [:-2]
    return torch.nn.Sequential(*list(torchvision.models.resnet50(weights='DEFAULT').children())[:-2])

class Classifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.output = torch.nn.LazyLinear(num_classes)

    def forward(self, x):
        x = torch.sum(x, (2, 3))  # global pooling
        return self.output(x)

########################### PROPOSALS ###########################

class Heatmap(torch.nn.Module):
    def forward(self, output_shape, bboxes, scores, bboxes_normalized=False):
        device = bboxes.device
        if bboxes_normalized:
            xstep = 1/output_shape[1]
            ystep = 1/output_shape[0]
            xx = torch.arange(0, 1, xstep, device=device)
            yy = torch.arange(0, 1, ystep, device=device)
        else:
            xstep = ystep = 1
            xx = torch.arange(output_shape[1], device=device)
            yy = torch.arange(output_shape[0], device=device)
        xx, yy = torch.meshgrid(xx, yy, indexing='xy')
        xprob = self.f(xx[None, None], bboxes[:, 0][..., None, None], bboxes[:, 2][..., None, None])
        yprob = self.f(yy[None, None], bboxes[:, 1][..., None, None], bboxes[:, 3][..., None, None])
        # avoid the pdf being too big for a single pixel
        probs = torch.clamp(xprob*yprob, max=1)
        if scores is None:
            r = torch.mean(probs, 1, True)
            print('r2:', r.min().detach(), r.max().detach())
            return r
        #scores = scores / scores.max()
        print('scores:', scores.min(), scores.max())
        heatmap = torch.sum(scores[..., None, None]*probs, 1, True)
        print('heatmap:', heatmap.min().detach(), heatmap.max().detach())
        return heatmap

class GaussHeatmap(Heatmap):
    def f(self, x, x1, x2):
        stdev_eps = 1e-6
        avg = x1
        # divided by 100 to make it smaller
        stdev = x2 + stdev_eps
        sqrt2pi = 2.5066282746310002
        return (1/(stdev*sqrt2pi)) * torch.exp(-0.5*(((x-avg)/stdev)**2))

class LogisticHeatmap(Heatmap):
    def f(self, x, x1, x2):
        k = 1
        x1 = x1 - x2/2
        x2 = x1 + x2/2
        logistic0 = 1/(1+torch.exp(-k*(x-x1)))
        logistic1 = 1 - 1/(1+torch.exp(-k*(x-x2)))
        #print('logistic0:', logistic0.min().detach(), logistic0.max().detach())
        #print('logistic1:', logistic1.min().detach(), logistic1.max().detach())
        r = logistic0 * logistic1
        r = r / r.amax(1, True)  # divide by max so it's not smaller than 1
        # FIXME: maybe divide by sum or max ?
        #print('r:', r.min().detach(), r.max().detach())
        #print('x1:', x1[0, :5, 0, 0].detach())
        #print('x2:', x2[0, :5, 0, 0].detach())
        #print('r:', r[0, :5, 0, 0].detach())
        #print()
        return r
