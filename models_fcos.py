import torch, torchvision

class MyFCOS(torch.nn.Module):
    def __init__(self, num_classes):
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
        self.class_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.Conv2d(256, 1+num_classes, 3, padding=1),
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

if __name__ == '__main__':
    m = MyFCOS(5)
    r = m(torch.zeros(10, 3, 800, 1024))
    for i, (ch, rh) in enumerate(r):
        print(f'P{i+3} {ch.shape} {rh.shape}')
