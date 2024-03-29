import torchvision

def STL10(root, split, transform=None):
    return torchvision.datasets.STL10(root, split, transform=transform)