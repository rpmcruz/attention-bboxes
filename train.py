import argparse
parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('dataset', choices=['Birds', 'StanfordCars', 'StanfordDogs'])
parser.add_argument('--protopnet', action='store_true')
parser.add_argument('--vit', action='store_true')
parser.add_argument('--detection', choices=['Simple', 'FasterRCNN', 'FCOS', 'DETR'])
parser.add_argument('--heatmap', choices=['GaussHeatmap', 'LogisticHeatmap'])
parser.add_argument('--penalty', type=float, default=0)
parser.add_argument('--nstdev', type=float, default=1)
parser.add_argument('--occlusion', default='encoder', choices=['none', 'encoder', 'image'])
parser.add_argument('--adversarial', action='store_true')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()
assert (args.detection == None) == (args.heatmap == None), 'Must enable both or neither detection/heatmap'

import torch
from torchvision.transforms import v2
from time import time
import data, models, utils
import baseline_protopnet, baseline_vit
device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################# DATA #############################

transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(0.2, 0.2),
    v2.ToDtype(torch.float32, True),
])
ds = getattr(data, args.dataset)
tr = ds('/data/toys', 'train', transforms)
tr = torch.utils.data.DataLoader(tr, args.batchsize, True, num_workers=4, pin_memory=True)

############################# MODEL #############################

backbone = models.Backbone()
classifier = models.Classifier(ds.num_classes)
detection = getattr(models, args.detection)() if args.detection else None
heatmap = getattr(models, args.heatmap)() if args.heatmap else None

if args.protopnet:
    model = baseline_protopnet.ProtoPNet(backbone, ds.num_classes)
elif args.vit:
    model = baseline_vit.ViT(ds.num_classes)
elif detection is None or args.occlusion == 'none':
    model = models.SimpleModel(backbone, classifier)
elif args.occlusion == 'encoder':
    model = models.EncoderOcclusionModel(backbone, classifier, detection, heatmap, args.adversarial)
else:  # image
    model = models.ImageOcclusionModel(backbone, classifier, detection, heatmap, args.adversarial)
model.to(device)
if args.protopnet:
    opt = torch.optim.Adam(list(model.backbone.parameters()) + list(model.prototype_layer.parameters()))
    opt2 = torch.optim.Adam(model.fc_layer.parameters())
else:
    opt = torch.optim.Adam(model.parameters())

############################# LOOP #############################

debug_batch = next(iter(tr))

model.train()
for epoch in range(args.epochs):
    tic = time()
    avg_loss = 0
    avg_sparsity = 0
    avg_bbox_size = 0
    avg_adv_loss = 0
    avg_acc = 0
    avg_pg = 0
    for x, masks, y in tr:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = torch.nn.functional.cross_entropy(pred['class'], y)
        if args.protopnet:
            loss += baseline_protopnet.stage1_loss(pred['min_distances_per_class'], y)
        if 'heatmap' in pred:
            loss += args.penalty * pred['heatmap'].mean()
            normalized_heatmap = pred['heatmap'] / torch.sum(pred['heatmap'], (1, 2), True)
            avg_sparsity += float(torch.mean(-normalized_heatmap*torch.log2(normalized_heatmap))) / len(tr)
        if 'bboxes' in pred:
            avg_bbox_size += float(torch.mean(pred['bboxes'][:, 2:])) / len(tr)
        opt.zero_grad()
        loss.backward(retain_graph=args.adversarial)
        opt.step()
        if args.protopnet:
            model.backbone.eval()
            pred = model(x)
            # (stage2) projection of prototypes
            baseline_protopnet.stage2_projection(model, pred['min_features'])
            # (stage3) convex optimization of last layer
            loss2 = torch.nn.functional.cross_entropy(pred['class'], y)
            loss2 += baseline_protopnet.stage3_loss(model)
            model.backbone.train()
            opt2.zero_grad()
            loss2.backward()
            opt2.step()
            loss += loss2
        if args.adversarial:
            # temporarily disable gradients for backbone and classifier
            for module in [backbone, classifier]:
                for param in module.parameters():
                    param.requires_grad = False
            adv_loss = torch.nn.functional.cross_entropy(pred['min_class'], y) + \
                -torch.nn.functional.cross_entropy(pred['max_class'], y)
            adv_loss.backward()
            for module in [backbone, classifier]:
                for param in module.parameters():
                    param.requires_grad = True
            avg_adv_loss += float(adv_loss) / len(tr)
        avg_loss += float(loss) / len(tr)
        avg_acc += (y == pred['class'].argmax(1)).float().mean() / len(tr)
        if 'heatmap' in pred:
            masks = masks.to(device)
            heatmaps = pred['heatmap'][:, None]
            heatmaps = torch.nn.functional.interpolate(heatmaps, masks.shape[-2:], mode='nearest-exact')
            avg_pg += float(torch.mean((masks.view(len(masks), -1)[range(len(masks)), torch.argmax(heatmaps.view(len(heatmaps), -1), 1)] != 0).float())) / len(tr)
    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Avg loss: {avg_loss} - Avg acc: {avg_acc}' + (f' - Avg adversarial loss: {avg_adv_loss}' if args.adversarial else '') + f' - Avg pg: {avg_pg} - Avg sparsity: {avg_sparsity} - Avg bbox size: {avg_bbox_size}')
    if args.debug:
        with torch.no_grad():
            x, _, y = debug_batch
            x = x[:4].to(device)
            y = y[:4].to(device)
            pred = model(x)
        import matplotlib.pyplot as plt
        plt.rcParams['figure.figsize'] = (18, 8)
        plt.clf()
        for i in range(4):
            plt.subplot(2, 4, i+1)
            utils.draw_bboxes(x[i], pred['bboxes'][i].detach(), args.nstdev)
            plt.title(f"y={y[i]} ŷ={pred['class'][i].argmax()}")
            plt.subplot(2, 4, i+4+1)
            utils.draw_heatmap(x[i], pred['heatmap'][i].detach())
        plt.suptitle(f'{args.output[:-4]} epoch={epoch+1}')
        plt.savefig(f'{args.output}-epoch-{epoch+1}.png')

torch.save(model.cpu(), args.output)
