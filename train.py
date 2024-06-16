import argparse
parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('dataset', choices=['Birds', 'StanfordCars', 'StanfordDogs'])
parser.add_argument('model', choices=['ProtoPNet', 'ViT', 'OnlyClass', 'Heatmap', 'SimpleDet', 'FasterRCNN', 'FCOS', 'DETR'])
parser.add_argument('--heatmap', choices=['GaussHeatmap', 'LogisticHeatmap'], default='GaussHeatmap')
parser.add_argument('--penalty', type=float, default=0)
parser.add_argument('--nstdev', type=float, default=2)
parser.add_argument('--occlusion', default='encoder', choices=['none', 'encoder', 'image'])
parser.add_argument('--adversarial', action='store_true')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--fast', action='store_true')
args = parser.parse_args()

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
if args.fast:
    tr = torch.utils.data.Subset(tr, [0]*args.batchsize)
tr = torch.utils.data.DataLoader(tr, args.batchsize, True, num_workers=4, pin_memory=True)

############################# MODEL #############################

if args.model == 'ProtoPNet':
    backbone = models.Backbone()
    model = baseline_protopnet.ProtoPNet(backbone, ds.num_classes)
    slow_opt = torch.optim.Adam(backbone.parameters(), 1e-4)
    fast_opt = torch.optim.Adam(model.prototype_layer.parameters())
    late_opt = torch.optim.Adam(model.fc_layer.parameters())
elif args.model == 'ViT':
    model = baseline_vit.ViT(ds.num_classes)
    slow_opt = torch.optim.Adam([], 1e-4)
    fast_opt = torch.optim.Adam(model.parameters())
else:
    backbone = models.Backbone()
    classifier = models.Classifier(ds.num_classes)
    detection = getattr(models, args.model)() if args.model != 'OnlyClass' else None
    heatmap = getattr(models, args.heatmap)()
    occlusion = 'none' if args.model == 'OnlyClass' else args.occlusion
    model = models.Occlusion(backbone, classifier, detection, heatmap, occlusion, args.adversarial)
    slow_opt = torch.optim.Adam(list(backbone.parameters()) + list(detection.parameters()), 1e-4)
    fast_opt = torch.optim.Adam(classifier.parameters())
model.to(device)

############################# LOOP #############################

visualize_batch = next(iter(tr))

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
        if args.model == 'ProtoPNet':
            loss += baseline_protopnet.stage1_loss(model, pred['features'], y)
        if 'heatmap' in pred:
            norm_heatmap = pred['heatmap'] / torch.sum(pred['heatmap'], (1, 2), True)
            entropy = torch.mean(-norm_heatmap*torch.log2(norm_heatmap+1e-7))
            loss += args.penalty * entropy
            avg_sparsity += float(entropy) / len(tr)
        if 'bboxes' in pred:
            avg_bbox_size += float(torch.mean(pred['bboxes'][:, 2:])) / len(tr)
        fast_opt.zero_grad()
        slow_opt.zero_grad()
        loss.backward(retain_graph=args.adversarial)
        fast_opt.step()
        slow_opt.step()
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
    if args.model == 'ProtoPNet':
        # protopnet has two more stages: in the paper they do this after a few epochs
        model.backbone.eval()
        # (stage2) projection of prototypes
        all_features = [[] for _ in range(ds.num_classes)]
        all_distances = [[] for _ in range(ds.num_classes)]
        for x, _, y in tr:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                z = model.features(model.backbone(x)[-1])
                z = torch.flatten(z, 2).permute(0, 2, 1)
                for k in range(ds.num_classes):
                    ix = y == k
                    if ix.sum() > 0:
                        zk = z[ix]
                        pk = model.prototype_layer.prototypes[:, k]
                        distances = torch.cdist(zk, pk)
                        all_features[k].append(torch.flatten(zk, 0, 1))
                        all_distances[k].append(torch.flatten(distances, 0, 1))
        for k in range(ds.num_classes):
            num_prototypes = model.prototype_layer.prototypes.shape[2]
            ix = torch.argsort(torch.cat(all_distances[k]))
            # projection
            model.prototype_layer.prototypes[:, k] = torch.cat(all_features[k])[ix[:num_prototypes]]
        # (stage3) convex optimization of last layer
        for x, _, y in tr:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss2 = torch.nn.functional.cross_entropy(pred['class'], y)
            loss2 += baseline_protopnet.stage3_loss(model)
            late_opt.zero_grad()
            loss2.backward()
            late_opt.step()
        loss += float(loss2)
        model.backbone.train()
    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Avg loss: {avg_loss} - Avg acc: {avg_acc}' + (f' - Avg adversarial loss: {avg_adv_loss}' if args.adversarial else '') + f' - Avg pg: {avg_pg} - Avg sparsity: {avg_sparsity} - Avg bbox size: {avg_bbox_size}')
    if args.visualize:
        with torch.no_grad():
            x, _, y = visualize_batch
            x = x[:4].to(device)
            y = y[:4].to(device)
            pred = model(x)
        import matplotlib.pyplot as plt
        plt.rcParams['figure.figsize'] = (18, 8)
        plt.clf()
        for i in range(4):
            plt.subplot(2, 4, i+1)
            if 'bboxes' in pred:
                utils.draw_bboxes(x[i], pred['bboxes'][i].detach(), pred['scores'][i].detach(), args.nstdev)
            plt.title(f"y={y[i]} ŷ={pred['class'][i].argmax()}")
            plt.subplot(2, 4, i+4+1)
            utils.draw_heatmap(x[i], pred['heatmap'][i].detach())
        plt.suptitle(f'{args.output[:-4]} epoch={epoch+1}')
        plt.savefig(f'{args.output}-epoch-{epoch+1}.png')

torch.save(model.cpu(), args.output)
