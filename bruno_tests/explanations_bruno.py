import torch
from model_bruno import *
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip()      
])

STL10_set = datasets.STL10(root='C:/Users/Bruno/OneDrive/Documents/Datasets/STL10', split='train', transform=transformation, download=False)
STL10_loader = torch.utils.data.DataLoader(STL10_set, batch_size=1, shuffle=True,  pin_memory=False)

model = torch.load('model100epochsLRe4.pt')
model = model.to(device)
model.eval()

for x,y in STL10_loader:
    pred, att, boxes = model(x)
    if pred.argmax(1) == y:
        print('Correct prediction')
    else:
        print('Wrong prediction')

    visualize_boxes(img=x, bboxes=boxes, atts=att, n_boxes=10)    