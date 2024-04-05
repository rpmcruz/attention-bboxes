from scipy.io import loadmat
import torch, torchvision
import os

class Birds:
    # https://www.vision.caltech.edu/datasets/cub_200_2011/
    num_classes = 200
    def __init__(self, root, split, transform=None):
        self.root = os.path.join(root, 'CUB_200_2011')
        split = int(split == 'test')
        which = [int(line.split()[1]) == split for line in open(os.path.join(self.root, 'train_test_split.txt')).readlines()]
        files = [(label, folder, image) for label, folder in enumerate(sorted(os.listdir(os.path.join(self.root, 'images')))) for image in sorted(os.listdir(os.path.join(self.root, 'images', folder)))]
        assert len(which) == len(files), f'Split information ({len(which)}) differs from true files ({len(files)})'
        self.files = [f for w, f in zip(which, files) if w]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        label, folder, fname = self.files[i]
        image = os.path.join(self.root, 'images', folder, fname)
        image = torchvision.io.read_image(image, torchvision.io.ImageReadMode.RGB)
        mask = os.path.join(self.root, 'segmentations', folder, fname[:-3] + 'png')
        mask = torchvision.tv_tensors.Mask(torchvision.io.read_image(mask, torchvision.io.ImageReadMode.GRAY))
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask, label

class StanfordCars:
    # https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset
    # FIXME: the dataset from kaggle seems completely broken
    num_classes = 196
    def __init__(self, root, split, transform=None):
        root = os.path.join(root, 'stanford_cars')
        data = loadmat(os.path.join(root, 'cars_annos.mat'), simplify_cells=True)
        data = data['annotations']
        self.root = os.path.join(root, f'cars_{split}', f'cars_{split}')
        split = int(split == 'test')
        self.data = [d for d in data if d['test'] == split]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        image = torchvision.io.read_image(os.path.join(self.root, d['relative_im_path'].split('/')[1][1:]))
        mask = torch.zeros((image.shape[1], image.shape[2]), dtype=bool)
        mask[d['bbox_y1']:d['bbox_y2'], d['bbox_x1']:d['bbox_x2']] = True
        label = d['class']-1
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
        plt.suptitle(f'Class: {label}')
        plt.show()
