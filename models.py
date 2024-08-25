import torch, torchvision

############################# BASICS #############################

class Occlusion(torch.nn.Module):
    def __init__(self, backbone, classifier, detection, bboxes2heatmap, occlusion_level, is_adversarial):
        super().__init__()
        self.backbone = backbone
        self.detection = detection
        self.bboxes2heatmap = bboxes2heatmap
        self.classifier = classifier
        self.occlusion_level = occlusion_level
        self.is_adversarial = is_adversarial
        self.return_dict = True

    def forward(self, images):
        features = self.backbone(images)
        embed = features[-1]
        if self.occlusion_level == 'none':
            if not getattr(self, 'return_dict', True):
                # this is a hack because captum wants models to return class names
                return self.classifier(embed)
            return {'class': self.classifier(embed)}
        det = self.detection(features)
        if 'heatmap' not in det:
            heatmap_shape = embed.shape[2:] if self.occlusion_level == 'encoder' else images.shape[2:]
            scale = embed.shape[-1] / images.shape[-1]
            det['heatmap'] = self.bboxes2heatmap(heatmap_shape, det['bboxes'], det['scores'])
        heatmap = det['heatmap']
        if self.is_adversarial:
            if self.occlusion_level == 'encoder':
                min_embed = heatmap[:, None] * embed
                max_embed = (1-heatmap)[:, None] * embed
            else:  # image
                min_embed = self.backbone(heatmap[:, None] * images)
                max_embed = self.backbone((1-heatmap)[:, None] * images)
            return {
                'class': self.classifier(embed),
                'min_class': self.classifier(min_embed),
                'max_class': self.classifier(max_embed),
                **det
            }
        if self.occlusion_level == 'encoder':
            embed = heatmap[:, None] * embed
            return {
                'class': self.classifier(embed), **det
            }

class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50(weights='DEFAULT')

    def forward(self, x):
        layers = [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]
        x = self.resnet.maxpool(self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x))))
        ret = []
        for layer in layers:
            x = layer(x)
            ret.append(x)
        return ret

class Classifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.output = torch.nn.LazyLinear(num_classes)

    def forward(self, x):
        x = torch.mean(x, (2, 3))  # global pooling
        return self.output(x)

####################### OBJ DETECT MODELS #######################
# output format = cxcywh (normalized 0-1)

class Heatmap(torch.nn.Module):
    def __init__(self, use_sigmoid):
        super().__init__()
        self.conv = torch.nn.Conv2d(2048, 1, 3, padding=1)
        self.use_sigmoid = use_sigmoid

    def forward(self, features):
        heatmap = self.conv(features[-1])
        if self.use_sigmoid:
            heatmap = torch.sigmoid(heatmap)[:, 0]
        else:
            heatmap = torch.softmax(heatmap.reshape(len(heatmap), -1), 1).reshape(len(heatmap), heatmap.shape[2], heatmap.shape[3])
            heatmap = heatmap / torch.amax(heatmap, (1, 2), True)
        return {'heatmap': heatmap}

class SimpleDet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bboxes = torch.nn.LazyConv2d(2, 1)
        self.scores = torch.nn.LazyConv2d(1, 1)

    def forward(self, features):
        grid = features[-1]
        _, _, h, w = grid.shape
        xx, yy = torch.meshgrid(
            torch.arange(0, 1, 1/w, device=grid.device),
            torch.arange(0, 1, 1/h, device=grid.device),
            indexing='xy')
        bboxes = torch.sigmoid(self.bboxes(grid))
        bboxes = torch.flatten(torch.stack((
            (1/w)/2 + xx[None].repeat(len(grid), 1, 1), (1/h)/2 + yy[None].repeat(len(grid), 1, 1),
            bboxes[:, 0], bboxes[:, 1]), 1), 2)
        scores = torch.flatten(self.scores(grid), 1)
        return {'bboxes': bboxes, 'scores': scores}

class FasterRCNN(torch.nn.Module):
    # https://arxiv.org/abs/1506.01497
    # implementation based on
    # https://towardsdatascience.com/understanding-and-implementing-faster-r-cnn-a-step-by-step-guide-11acfff216b0
    def __init__(self, hidden_dim=512, dropout_p=0.3):
        super().__init__()
        # RPN
        # in Faster-RCNN, RPN also produces a score to filter bounding boxes
        self.rpn_conv = torch.nn.LazyConv2d(hidden_dim, 3, padding=1)
        self.rpn_dropout = torch.nn.Dropout(dropout_p)
        self.offsets = torch.nn.Conv2d(hidden_dim, 9*4, 1)
        #self.scores = torch.nn.Conv2d(hidden_dim, 9*1, 1)
        # classifier
        self.clf_hidden = torch.nn.LazyLinear(hidden_dim)
        self.clf_dropout = torch.nn.Dropout(dropout_p)
        self.clf_head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, features):
        grid = features[-1]

        # RPN
        grid = self.rpn_dropout(self.rpn_conv(grid))
        offsets = self.offsets(grid).reshape(grid.shape[0], 4, 9, grid.shape[2], grid.shape[3])
        offsets = torch.sigmoid(offsets)
        #scores = torch.sigmoid(self.scores(scores))

        _, _, h, w = grid.shape
        xx, yy = torch.meshgrid(
            torch.arange(0, 1, 1/w, device=grid.device),
            torch.arange(0, 1, 1/h, device=grid.device),
            indexing='xy')

        # in the paper, the anchors are for an image of minimum side=600,
        # therefore we normalize since we are using 0-1 predictions
        scales = [128/600, 256/600, 512/600]
        ratios = [0.5, 1, 2]
        anchors = [(scale*(ratio**0.5), scale/(ratio**0.5)) for scale in scales for ratio in ratios]
        anchors = torch.tensor(anchors, device=grid.device).T[None, ..., None, None]
        bboxes = torch.flatten(torch.stack((
            #xx + anchors[:, 0]/2 + offsets[:, 0]*anchors[:, 0],
            #yy + anchors[:, 1]/2 + offsets[:, 1]*anchors[:, 1],
            #torch.exp(offsets[:, 2])*anchors[:, 0],
            #torch.exp(offsets[:, 3])*anchors[:, 1]
            # due to unstability, I changed to
            xx + offsets[:, 0]/w,
            yy + offsets[:, 1]/h,
            offsets[:, 2]*anchors[:, 0],
            offsets[:, 3]*anchors[:, 1]
        ), 1), 2)  # (B, 4, 9*H*W)
        # ops.roi_pool() requires [L, 4] with size B
        # ops.roi_pool() wants xyxy (not xywh)
        roi_bboxes = list(bboxes.permute(0, 2, 1))
        rescale = torch.tensor([grid.shape[3], grid.shape[2]]*2, device=grid.device)
        roi_bboxes = [torchvision.ops.box_convert(bb*rescale[None], 'xywh', 'xyxy') for bb in roi_bboxes]
        rois = torchvision.ops.roi_pool(grid, roi_bboxes, (grid.shape[2], grid.shape[3]), 1)
        rois = rois.reshape(grid.shape[0], -1, grid.shape[1], grid.shape[2], grid.shape[3])

        # classifier
        rois = torch.mean(rois, [-1, -2])
        scores = self.clf_head(torch.nn.functional.relu(self.clf_dropout(self.clf_hidden(rois))))
        scores = scores[..., 0]
        return {'bboxes': bboxes, 'scores': scores}

class FCOS(torch.nn.Module):
    # implemented from scratch based on the paper
    # https://arxiv.org/abs/1904.01355
    def __init__(self):
        super().__init__()
        self.P3 = torch.nn.Conv2d(512, 256, 1)
        self.P4 = torch.nn.Conv2d(1024, 256, 1)
        self.P5 = torch.nn.Conv2d(2048, 256, 1)
        self.P6 = torch.nn.Conv2d(256, 256, 1, stride=2)
        self.P7 = torch.nn.Conv2d(256, 256, 1, stride=2)
        # head is shared across feature levels
        # we use class_head as our score_head
        self.reg_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.Conv2d(256, 4, 3, padding=1),
        )
        self.clf_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.Conv2d(256, 1, 3, padding=1),
        )
        # instead of exp(x), use exp(s_ix) with a trainable scalar s_i for each P_i
        #self.s = torch.nn.parameter.Parameter(torch.ones([5]))

    def forward(self, features):
        _, C3, C4, C5 = features
        upsample = torch.nn.functional.interpolate
        P5 = self.P5(C5)
        P6 = self.P6(P5)
        P7 = self.P7(P6)
        P4 = self.P4(C4) + upsample(P5, scale_factor=2, mode='nearest-exact')
        P3 = self.P3(C3) + upsample(P4, scale_factor=2, mode='nearest-exact')

        _bboxes = []
        _scores = []
        for P in [P3, P4, P5, P6, P7]:
            _, _, h, w = P.shape
            xx, yy = torch.meshgrid(
                torch.arange(0, 1, 1/w, device=P.device),
                torch.arange(0, 1, 1/h, device=P.device),
                indexing='xy')
            #bboxes = torch.exp(self.s[i]*self.reg_head(P))
            bboxes = torch.sigmoid(self.reg_head(P))
            bboxes = torch.stack((
                xx + bboxes[:, 0]/w, yy + bboxes[:, 1]/h,
                #bboxes[:, 2]-bboxes[:, 0], bboxes[:, 3]-bboxes[:, 1]
                bboxes[:, 2], bboxes[:, 3]
            ), 1)
            scores = self.clf_head(P)
            _bboxes.append(torch.flatten(bboxes, 2))
            _scores.append(torch.flatten(scores, 1))
        bboxes = torch.cat(_bboxes, 2)
        scores = torch.cat(_scores, 1)
        return {'bboxes': bboxes, 'scores': scores}

class DETR(torch.nn.Module):
    # implementation from the paper
    # https://arxiv.org/abs/2005.12872
    def __init__(self, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        self.conv = torch.nn.LazyConv2d(hidden_dim, 1)
        self.transformer = torch.nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, batch_first=True)
        self.bboxes = torch.nn.Linear(hidden_dim, 4)
        self.scores = torch.nn.Linear(hidden_dim, 1)
        self.query_pos = torch.nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = torch.nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = torch.nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, features):
        grid = features[-1]
        h = self.conv(grid)
        N, _, H, W = h.shape
        pos = torch.cat([
            self.col_embed[None, :W].repeat(H, 1, 1),
            self.row_embed[:H, None].repeat(1, W, 1),
        ], -1).flatten(0, 1)[None]
        h = self.transformer(pos + h.flatten(2).permute(0, 2, 1), self.query_pos[None].repeat(N, 1, 1))
        scores = self.scores(h)[..., 0]
        bboxes = torch.sigmoid(self.bboxes(h)).permute(0, 2, 1)
        return {'bboxes': bboxes, 'scores': scores}

########################### PROPOSALS ###########################

class Bboxes2Heatmap(torch.nn.Module):
    def __init__(self, use_sigmoid):
        super().__init__()
        self.use_sigmoid = use_sigmoid

    def forward(self, output_shape, bboxes, scores):
        device = bboxes.device
        xx = torch.arange(output_shape[1], device=device)
        yy = torch.arange(output_shape[0], device=device)
        xx, yy = torch.meshgrid(xx, yy, indexing='xy')
        scale = torch.tensor([output_shape[1], output_shape[0]]*2, device=bboxes.device)
        bboxes = scale[None, :, None]*bboxes
        xprob = self.f(xx[None, None], bboxes[:, 0][..., None, None], bboxes[:, 2][..., None, None])
        yprob = self.f(yy[None, None], bboxes[:, 1][..., None, None], bboxes[:, 3][..., None, None])
        # avoid the pdf being too big for a single pixel
        #probs = torch.clamp(xprob*yprob, max=1)
        probs = xprob*yprob
        if self.use_sigmoid:
            scores = torch.sigmoid(scores)
            heatmap = torch.sum(scores[..., None, None]*probs, 1)
            heatmap = heatmap / torch.amax(heatmap, [1, 2], True)  # ensure 0-1
        else:
            scores = torch.softmax(scores, 1)
            heatmap = torch.sum(scores[..., None, None]*probs, 1)
            heatmap = heatmap / torch.amax(heatmap, [1, 2], True)  # ensure 0-1
        return heatmap

class GaussHeatmap(Bboxes2Heatmap):
    def f(self, x, cx, bw):
        avg = cx
        stdev = bw + 1e-6
        assert stdev.amin() > 0, 'bbox width or height has negative value ' + str(stdev.amin().item())
        sqrt2pi = (2*torch.pi)**0.5
        return (1/(stdev*sqrt2pi)) * torch.exp(-0.5*(((x-avg)/stdev)**2))

class LogisticHeatmap(Bboxes2Heatmap):
    def f(self, x, cx, bw):
        k = 1
        x1 = cx - bw/2
        x2 = cx + bw/2
        logistic0 = 1/(1+torch.exp(-k*(x-x1)))
        logistic1 = 1 - 1/(1+torch.exp(-k*(x-x2)))
        r = logistic0 * logistic1
        r = r / r.amax(1, True)  # divide by max so it's not smaller than 1
        return r
