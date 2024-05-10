import torch
import torchvision

####################### OBJ DETECT MODELS #######################

class OneStage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bboxes = torch.nn.LazyConv2d(4, 1)
        self.scores = torch.nn.LazyConv2d(1, 1)

    def forward(self, grid):
        bboxes, scores = self.output(grid)
        bboxes = torch.sigmoid(bboxes)
        scores = torch.sigmoid(scores)
        # center x/y move to relative coordinates
        device = grid.device
        xstep = 1/grid.shape[3]
        ystep = 1/grid.shape[2]
        xx = torch.arange(0, 1, xstep, device=device)
        yy = torch.arange(0, 1, ystep, device=device)
        xx, yy = torch.meshgrid(xx, yy, indexing='xy')
        bboxes[:, 0] = xx + bboxes[:, 0]/xstep
        bboxes[:, 1] = yy + bboxes[:, 1]/ystep
        return torch.flatten(bboxes, 1), torch.flatten(scores, 1)

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
    def __init__(self, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        self.conv = torch.nn.LazyConv2d(hidden_dim, 1)
        self.transformer = torch.nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, batch_first=True)
        self.linear_bboxes = torch.nn.Linear(hidden_dim, 4)
        self.query_pos = torch.nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = torch.nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = torch.nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, x):
        h = self.conv(x)
        N, _, H, W = h.shape
        pos = torch.cat([
            self.col_embed[None, :W].repeat(H, 1, 1),
            self.row_embed[:H, None].repeat(1, W, 1),
        ], -1).flatten(0, 1)[None]
        h = self.transformer(pos + h.flatten(2).permute(0, 2, 1), self.query_pos[None].repeat(N, 1, 1))
        bboxes = torch.sigmoid(self.linear_bboxes(h))
        return bboxes, None

############################# GLUE #############################

class Model(torch.nn.Module):
    def __init__(self, backbone, classifier, object_detection=None, bboxes2heatmap=None):
        super().__init__()
        self.backbone = backbone
        self.object_detection = object_detection
        self.bboxes2heatmap = bboxes2heatmap
        self.classifier = classifier

    def forward(self, images):
        embed = self.backbone(images)
        heatmap = bboxes = None
        if self.object_detection is not None:
            bboxes, scores = self.object_detection(embed)
            heatmap = self.bboxes2heatmap(embed.shape[2:], bboxes, scores)
            embed = heatmap * embed
            if scores != None:
                bboxes = [bb[ss >= 0.5] for bb, ss in zip(bboxes, scores)]
        return self.classifier(embed), heatmap, bboxes

def Backbone():
    # image 96x96 ---> grid 6x6
    # sometimes people use [:-2] because they use 224x224 images (7x7 grid)
    return torch.nn.Sequential(*list(torchvision.models.resnet50(weights='DEFAULT').children())[:-3])

class Classifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.output = torch.nn.LazyLinear(num_classes)

    def forward(self, x):
        x = torch.sum(x, (2, 3))  # global pooling
        return self.output(x)

########################### PROPOSALS ###########################

class Heatmap(torch.nn.Module):
    def forward(self, output_shape, bboxes, scores):
        device = bboxes.device
        xstep = 1/output_shape[1]
        ystep = 1/output_shape[0]
        xx = torch.arange(0, 1, xstep, device=device)
        yy = torch.arange(0, 1, ystep, device=device)
        xx, yy = torch.meshgrid(xx, yy, indexing='xy')
        xprob = self.f(xx[None, None], bboxes[:, 0][..., None, None], bboxes[:, 2][..., None, None])
        yprob = self.f(yy[None, None], bboxes[:, 1][..., None, None], bboxes[:, 3][..., None, None])
        if scores is None:
            return torch.mean(xprob*yprob, 1, True)
        return torch.sum(scores[..., None, None]*xprob*yprob, 1, True)

class GaussHeatmap(Heatmap):
    def f(self, x, avg, stdev):
        stdev_eps = 0.01
        stdev = stdev + stdev_eps
        sqrt2pi = 2.5066282746310002
        return (1/(stdev*sqrt2pi)) * torch.exp(-0.5*(((x-avg)/stdev)**2))

class LogisticHeatmap(Heatmap):
    def f(self, x, center, size):
        k = 1
        x0 = center-size/2
        x1 = center+size/2
        logistic0 = 1/(1+torch.exp(-k*(x-x0)))
        logistic1 = 1 - 1/(1+torch.exp(-k*(x-x1)))
        return logistic0 + logistic1