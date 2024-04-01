import torch

'''
(Part 1)
image      grid     scores (softmax)
□□□□□       □□□       □□□
□□□□□  ==>  □□□  ==>  □□□
□□□□□       □□□       □□□
□□□□□

(Part 2)
grid ⊙ scores      (hidden)   output
    □□□               □
    □□□         ==>   □   ==>  □
    □□□               □
'''

class BasicGrid(torch.nn.Module):
    def __init__(self, num_classes, use_softmax):
        super().__init__()
        # image 96x96 ---> grid 6x6
        self.grid = torch.nn.Sequential(        # 96x96
            torch.nn.LazyConv2d(128, 3, 2, 1),  # 48x48
            torch.nn.LazyConv2d(256, 3, 2, 1),  # 24x24
            torch.nn.LazyConv2d(512, 3, 2, 1),  # 12x12
            torch.nn.LazyConv2d(1024, 3, 2, 1), # 6x6
        )
        self.scores = torch.nn.LazyConv2d(1, 1)
        self.output = torch.nn.LazyLinear(num_classes)
        self.use_softmax = use_softmax

    def forward(self, images):
        grid = self.grid(images)
        scores = self.scores(grid)
        # temporarily flatten scores matrix because pytorch softmax can only be
        # done across one single dimension
        if self.use_softmax:
            scores = torch.softmax(torch.flatten(scores, 2), 2).view(*scores.shape)
        else:
            scores = torch.sigmoid(scores)
        hidden = torch.sum(grid * scores, (2, 3))
        return self.output(hidden), scores, None

'''
(Part 1)
image      grid bboxes (cx,cy,log(w),log(h),score)
□□□□□       □□□
□□□□□  ==>  □□□
□□□□□       □□□
□□□□□

(Part 2)
image      extract bboxes     latent
□□□□□       □ □ □             □ □ □
□□□□□  ==>  □ □ □        ==>  □ □ □
□□□□□       □ □ □             □ □ □
□□□□□

(Part 3)
latent ⊙ scores    (hidden)   output
    □□□               □
    □□□         ==>   □   ==>  □
    □□□               □
'''

def extract_roi(images, bboxes, region_size):
    bboxes = bboxes.int()
    regions = torch.zeros((bboxes.shape[0], 3, bboxes.shape[2], bboxes.shape[3]))
    for i, (image, image_bboxes) in enumerate(zip(images, bboxes)):
        for j, bbox in enumerate(image_bboxes):
            region = image[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
            region = torch.nn.functional.interpolate(region, region_size)
            regions[i, j] = region
    return regions

class BboxGrid(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # image 96x96 ---> grid 6x6
        self.grid = torch.nn.Sequential(        # 96x96
            torch.nn.LazyConv2d(128, 3, 2, 1),  # 48x48
            torch.nn.LazyConv2d(256, 3, 2, 1),  # 24x24
            torch.nn.LazyConv2d(512, 3, 2, 1),  # 12x12
            torch.nn.LazyConv2d(5, 3, 2, 1),    # 6x6
        )
        # region 12x12 ---> latent 256
        self.regions = torch.nn.Sequential(     # 12x12
            torch.nn.LazyConv2d(128, 3, 2, 1),  # 6x6
            torch.nn.LazyConv2d(256, 3, 2, 1),  # 3x3
        )
        self.output = torch.nn.LazyLinear(num_classes)

    def forward(self, images):
        grid = self.grid(images)
        # grid is (cy, cx, h, w, scores)
        rr, cc = torch.meshgrid(torch.arange(grid.shape[2], device=images.device), torch.arange(grid.shape[3], device=images.device), indexing='ij')
        centers_rescale = (torch.tensor(images.shape[[2, 3]])/torch.tensor(grid.shape[[2, 3]]))[None, :, None, None]
        sizes_rescale = torch.tensor(images.shape[[2, 3]])[None, :, None, None]
        centers = centers_rescale * (torch.sigmoid(grid[:, 0:2]) + torch.stack((rr, cc))[None, :, None, None])
        sizes = sizes_rescale * torch.sigmoid(grid[:, 2:4])
        # our bboxes are (y1, x1, y2, x2)
        bboxes = torch.cat((centers-sizes/2, centers+sizes/2), 1)
        bboxes[:, [0, 1]] = torch.clamp(bboxes[:, [0, 1]], min=0)
        bboxes[:, 2] = torch.clamp(bboxes[:, 2], max=images.shape[2]-1)
        bboxes[:, 3] = torch.clamp(bboxes[:, 3], max=images.shape[3]-1)
        scores = grid[:, [4]]
        # temporarily flatten scores matrix because pytorch softmax can only be
        # done across one single dimension
        scores = torch.softmax(torch.flatten(scores, 2), 2).view(*scores.shape)
        # extract bounding boxes
        regions = extract_roi(images, bboxes, (12, 12))
        # another cnn on each region
        latent = self.regions(regions)
        # attention
        hidden = torch.sum(latent * scores, (2, 3))
        return self.output(hidden), scores, None

'''
(Part 1)
image      grid latent    gaussian (cx,cy,log(varx),log(vary))
□□□□□       □□□             □□□
□□□□□  ==>  □□□       =>    □□□
□□□□□       □□□             □□□
□□□□□

(Part 2)
grid gaussian      pdf gaussian
    □□□               □□□
    □□□         ==>   □□□
    □□□               □□□

(Part 3)
latent ⊙ pdf      (hidden)   output
    □□□               □
    □□□         ==>   □   ==>  □
    □□□               □
'''

def gaussian_pdf(x, avg, stdev):
    sqrt2pi = 2.5066282746310002
    return (1/(stdev*sqrt2pi)) * torch.exp(-0.5*(((x-avg)/stdev)**2))

class GaussianGrid(torch.nn.Module):
    # use_softmax is ignored
    def __init__(self, num_classes, use_softmax):
        super().__init__()
        # image 96x96 ---> grid 6x6
        self.grid = torch.nn.Sequential(        # 96x96
            torch.nn.LazyConv2d(128, 3, 2, 1),  # 48x48
            torch.nn.LazyConv2d(256, 3, 2, 1),  # 24x24
            torch.nn.LazyConv2d(512, 3, 2, 1),  # 12x12
            torch.nn.LazyConv2d(1024, 3, 2, 1), # 6x6
        )
        self.gaussian = torch.nn.LazyConv2d(5, 1)
        self.output = torch.nn.LazyLinear(num_classes)

    def forward(self, images):
        grid = self.grid(images)
        gauss = self.gaussian(grid)
        xx, yy = torch.meshgrid(torch.arange(grid.shape[3], device=images.device), torch.arange(grid.shape[2], device=images.device), indexing='xy')
        x_gauss_avg = torch.flatten(torch.sigmoid(gauss[:, 0]) + xx, 1)
        y_gauss_avg = torch.flatten(torch.sigmoid(gauss[:, 1]) + yy, 1)
        x_gauss_stdev = torch.flatten(torch.exp(gauss[:, 2]), 1)
        y_gauss_stdev = torch.flatten(torch.exp(gauss[:, 3]), 1)
        score_gauss = torch.flatten(torch.sigmoid(gauss[:, 4]), 1)
        x_prob = gaussian_pdf(xx[None, None], x_gauss_avg[..., None, None], x_gauss_stdev[..., None, None])
        y_prob = gaussian_pdf(yy[None, None], y_gauss_avg[..., None, None], y_gauss_stdev[..., None, None])
        scores = torch.sum(score_gauss[:, :, None, None] * (x_prob*y_prob), 1, True)
        hidden = torch.sum(scores * grid, [2, 3])
        # 68.27% of the data falls below 1 stddev of the mean
        # 95.00% of the data falls below 2 stddevs of the mean
        n = 1
        xscale = images.shape[3] / grid.shape[3]
        yscale = images.shape[2] / grid.shape[2]
        bboxes = torch.stack((
            xscale * torch.clamp(x_gauss_avg-n*x_gauss_stdev, min=0),
            yscale * torch.clamp(y_gauss_avg-n*y_gauss_stdev, min=0),
            xscale * torch.clamp(x_gauss_avg+n*x_gauss_stdev, max=grid.shape[3]),
            yscale * torch.clamp(y_gauss_avg+n*y_gauss_stdev, max=grid.shape[2])
        ), 2)
        # filter those bounding boxes below a certain level of confidence
        conf = 0.25
        bboxes = [[bbox for bbox, score in zip(bboxes[i], score_gauss[i]) if score >= conf] for i in range(bboxes.shape[0])]
        return self.output(hidden), scores, bboxes

if __name__ == '__main__':
    m = GaussianGrid(5)
    out = m(torch.zeros(5, 3, 256, 256))
