from scipy.io import loadmat
import xml.etree.ElementTree as ET
import torch, torchvision
import torchvision.tv_tensors
import os

class Birds:
    # https://www.vision.caltech.edu/datasets/cub_200_2011/
    num_classes = 200
    def __init__(self, root, split, transform=None):
        self.root = os.path.join(root, 'CUB_200_2011')
        files = open(os.path.join(self.root, 'images.txt'))
        split = open(os.path.join(self.root, 'train_test_split.txt'))
        labels = open(os.path.join(self.root, 'image_class_labels.txt'))
        # I don't know if split=0 is test and split=1 is train but I am assuming
        # that since split=0 49% and split=1 51%
        train = int(split == 'train')
        self.files = [f.split()[1] for f, s in zip(files, split) if int(s.split()[1]) == train]
        self.labels = [int(l.split()[1])-1 for l in labels]
        self.class_names = [line.split()[1][4:-1] for line in open(os.path.join(self.root, 'classes.txt'))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]
        label = self.labels[i]
        image = os.path.join(self.root, 'images', fname)
        image = torchvision.io.read_image(image, torchvision.io.ImageReadMode.RGB)
        mask = os.path.join(self.root, 'segmentations', fname[:-3] + 'png')
        mask = torchvision.tv_tensors.Mask(torchvision.io.read_image(mask, torchvision.io.ImageReadMode.GRAY))
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask, label

class StanfordCars:
    # https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset
    num_classes = 196
    def __init__(self, root, split, transform=None):
        root = os.path.join(root, 'stanford_cars')
        data = loadmat(os.path.join(root, 'devkit', f'cars_{split}_annos.mat'), simplify_cells=True)
        self.class_names = list(loadmat(os.path.join(root, 'devkit', 'cars_meta.mat'), simplify_cells=True)['class_names'])
        self.data = data['annotations']
        self.root = os.path.join(root, 'cars_train')
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        image = torchvision.io.read_image(os.path.join(self.root, d['fname']))
        mask = torch.zeros(1, image.shape[1], image.shape[2], dtype=bool)
        mask[0, d['bbox_y1']:d['bbox_y2']+1, d['bbox_x1']:d['bbox_x2']+1] = True
        label = d['class']-1
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask, label

class StanfordDogs:
    # http://vision.stanford.edu/aditya86/ImageNetDogs/
    num_classes = 120
    def __init__(self, root, split, transform=None):
        self.root = os.path.join(root, 'stanford_dogs')
        l = loadmat(os.path.join(self.root, f'{split}_list.mat'), simplify_cells=True)
        self.files = l['file_list']
        self.labels = l['labels']-1
        self.class_names = ['-'.join(species.split('-')[1:]) for species in sorted(os.listdir(os.path.join(root, 'stanford_dogs', 'Images')))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]
        label = self.labels[i]
        image = torchvision.io.read_image(os.path.join(self.root, 'Images', fname))
        ann = ET.parse(os.path.join(self.root, 'Annotation', fname[:-4]))
        bndboxes = [{c.tag: int(c.text) for c in bbox} for bbox in ann.findall('.//bndbox')]
        mask = torch.zeros(1, image.shape[1], image.shape[2], dtype=bool)
        for bndbox in bndboxes:
            mask[0, bndbox['ymin']:bndbox['ymax']+1, bndbox['xmin']:bndbox['xmax']+1] = True
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask, label

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    args = parser.parse_args()
    ds = globals()[args.dataset]('/data/toys', 'train')
    import matplotlib.pyplot as plt
    for i, (image, mask, label) in enumerate(ds):
        plt.subplot(1, 2, 1)
        plt.imshow(image.permute(1, 2, 0))
        plt.subplot(1, 2, 2)
        plt.imshow(mask[0])
        if hasattr(ds, 'class_names'):
            label = f'{ds.class_names[label]} {label}'
        plt.suptitle(str(label))
        plt.show()
