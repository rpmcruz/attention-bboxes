# code adapted from the ProtoPNet paper authors
# https://github.com/cfchen-duke/ProtoPNet

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('dataset', choices=['Birds', 'StanfordCars', 'StanfordDogs'])
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batchsize', type=int, default=8)
args = parser.parse_args()

import torch, torchvision
from torchvision.transforms import v2
from skimage.io import imsave
from time import time
import data
device = 'cuda' if torch.cuda.is_available() else 'cpu'

######################################### DATA #########################################

class TransformProb(torch.nn.Module):
    def __init__(self, transform, prob):
        super().__init__()
        self.transform = transform
        self.prob = prob
    def forward(self, *x):
        if torch.rand(()) < self.prob:
            x = self.transform(*x)
        return x

train_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(),
    TransformProb(v2.RandomRotation(15), 0.33),
    TransformProb(v2.RandomAffine(0, shear=10), 0.33),
    v2.RandomPerspective(0.2, 0.33),
    v2.ColorJitter(0.2, 0.2),
    v2.ToDtype(torch.float32, True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
test_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
train_dataset = getattr(data, args.dataset)('/data/toys', 'train', train_transforms)
train_push_dataset = getattr(data, args.dataset)('/data/toys', 'train', test_transforms)
test_dataset = getattr(data, args.dataset)('/data/toys', 'test', test_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, args.batchsize, True, num_workers=4, pin_memory=True)
train_push_loader = torch.utils.data.DataLoader(train_push_dataset, args.batchsize, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, args.batchsize, num_workers=4, pin_memory=True)

######################################### HELPER #########################################

def compute_rf_protoL_at_spatial_location(img_size, height_index, width_index, protoL_rf_info):
    n = protoL_rf_info[0]
    j = protoL_rf_info[1]
    r = protoL_rf_info[2]
    start = protoL_rf_info[3]
    assert height_index < n
    assert width_index < n

    center_h = start + (height_index*j)
    center_w = start + (width_index*j)

    rf_start_height_index = max(int(center_h - (r/2)), 0)
    rf_end_height_index = min(int(center_h + (r/2)), img_size)

    rf_start_width_index = max(int(center_w - (r/2)), 0)
    rf_end_width_index = min(int(center_w + (r/2)), img_size)
    return [rf_start_height_index, rf_end_height_index, rf_start_width_index, rf_end_width_index]

def compute_rf_prototype(img_size, prototype_patch_index, protoL_rf_info):
    img_index = prototype_patch_index[0]
    height_index = prototype_patch_index[1]
    width_index = prototype_patch_index[2]
    rf_indices = compute_rf_protoL_at_spatial_location(img_size, height_index, width_index, protoL_rf_info)
    return [img_index, rf_indices[0], rf_indices[1], rf_indices[2], rf_indices[3]]

def compute_layer_rf_info(layer_filter_size, layer_stride, layer_padding, previous_layer_rf_info):
    import math
    n_in = previous_layer_rf_info[0] # input size
    j_in = previous_layer_rf_info[1] # receptive field jump of input layer
    r_in = previous_layer_rf_info[2] # receptive field size of input layer
    start_in = previous_layer_rf_info[3] # center of receptive field of input layer

    if layer_padding == 'SAME':
        n_out = math.ceil(float(n_in) / float(layer_stride))
        if (n_in % layer_stride == 0):
            pad = max(layer_filter_size - layer_stride, 0)
        else:
            pad = max(layer_filter_size - (n_in % layer_stride), 0)
        assert(n_out == math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1) # sanity check
        assert(pad == (n_out-1)*layer_stride - n_in + layer_filter_size) # sanity check
    elif layer_padding == 'VALID':
        n_out = math.ceil(float(n_in - layer_filter_size + 1) / float(layer_stride))
        pad = 0
        assert(n_out == math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1) # sanity check
        assert(pad == (n_out-1)*layer_stride - n_in + layer_filter_size) # sanity check
    else:
        # layer_padding is an int that is the amount of padding on one side
        pad = layer_padding * 2
        n_out = math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1

    pL = math.floor(pad/2)

    j_out = j_in * layer_stride
    r_out = r_in + (layer_filter_size - 1)*j_in
    start_out = start_in + ((layer_filter_size - 1)/2 - pL)*j_in
    return [n_out, j_out, r_out, start_out]

def compute_proto_layer_rf_info_v2(img_size, layer_filter_sizes, layer_strides, layer_paddings, prototype_kernel_size):
    assert len(layer_filter_sizes) == len(layer_strides)
    assert len(layer_filter_sizes) == len(layer_paddings)

    rf_info = [img_size, 1, 1, 0.5]
    for i in range(len(layer_filter_sizes)):
        filter_size = layer_filter_sizes[i]
        stride_size = layer_strides[i]
        padding_size = layer_paddings[i]
        rf_info = compute_layer_rf_info(layer_filter_size=filter_size, layer_stride=stride_size,
            layer_padding=padding_size, previous_layer_rf_info=rf_info)

    proto_layer_rf_info = compute_layer_rf_info(layer_filter_size=prototype_kernel_size,
        layer_stride=1, layer_padding='VALID', previous_layer_rf_info=rf_info)
    return proto_layer_rf_info

def find_high_activation_crop(activation_map, percentile=0.95):
    threshold = torch.quantile(activation_map, percentile)
    mask = torch.ones_like(activation_map)
    mask[activation_map < threshold] = 0
    for lower_y in range(mask.shape[0]):
        if torch.amax(mask[lower_y]) > 0.5:
            break
    for upper_y in reversed(range(mask.shape[0])):
        if torch.amax(mask[upper_y]) > 0.5:
            break
    for lower_x in range(mask.shape[1]):
        if torch.amax(mask[:, lower_x]) > 0.5:
            break
    for upper_x in reversed(range(mask.shape[1])):
        if torch.amax(mask[:, upper_x]) > 0.5:
            break
    return lower_y, upper_y+1, lower_x, upper_x+1

######################################### MODEL #########################################

class PPNet(torch.nn.Module):
    def __init__(self, img_size, prototype_shape, num_classes, proto_layer_rf_info):
        super().__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        assert(self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1
        self.prototype_class_identity = torch.nn.Parameter(self.prototype_class_identity, False)

        # this has to be named features to allow the precise loading
        self.backbone = torch.nn.Sequential(*(list(torchvision.models.resnet50(weights='DEFAULT').children())[:-2]))

        # FIXME: this is a difference between mine and this. in mine, I always added two layers
        # with the same shape as the prototypes.
        # I also was not using sigmoid ...
        current_in_channels = 2048
        add_on_layers = []
        while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
            current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
            add_on_layers.append(torch.nn.Sequential(
                torch.nn.Conv2d(current_in_channels, current_out_channels, 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(current_out_channels, current_out_channels, 1),
                torch.nn.ReLU() if current_out_channels > self.prototype_shape[1] else torch.nn.Sigmoid()
            ))
            current_in_channels = current_in_channels // 2
        self.add_on_layers = torch.nn.Sequential(*add_on_layers)
       
        self.proto_layer_rf_info = proto_layer_rf_info
        self.prototype_vectors = torch.nn.Parameter(torch.rand(self.prototype_shape))
        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = torch.nn.Parameter(torch.ones(self.prototype_shape), False)
        self.last_layer = torch.nn.Linear(self.num_prototypes, self.num_classes, bias=False) # do not use bias

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        '''
        input2 = input ** 2
        input_patch_weighted_norm2 = torch.nn.functional.conv2d(input2, weights)

        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = torch.nn.functional.conv2d(input, weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = torch.nn.functional.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2
        x2_patch_sum = torch.nn.functional.conv2d(x2, self.ones)

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, (1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2[:, None, None]

        xp = torch.nn.functional.conv2d(x, self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = torch.nn.functional.relu(x2_patch_sum + intermediate_result)
        return distances

    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        conv_features = self.add_on_layers(self.backbone(x))
        distances = self._l2_convolution(conv_features)
        return distances

    def distance_2_similarity(self, distances):
        return torch.log((distances + 1) / (distances + self.epsilon))

    def forward(self, x):
        distances = self.prototype_distances(x)
        '''
        we cannot refactor the lines below for similarity scores
        because we need to return min_distances
        '''
        # global min pooling
        min_distances = -torch.nn.functional.max_pool2d(-distances, (distances.shape[2], distances.shape[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)
        logits = self.last_layer(prototype_activations)
        return logits, min_distances

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        conv_output = self.add_on_layers(self.backbone(x))
        distances = self._l2_convolution(conv_output)
        return conv_output, distances

    def prune_prototypes(self, prototypes_to_prune):
        '''
        prototypes_to_prune: a list of indices each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        '''
        prototypes_to_keep = list(set(range(self.num_prototypes)) - set(prototypes_to_prune))

        self.prototype_vectors = torch.nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...])

        self.num_prototypes = self.prototype_vectors.shape[0]

        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are consistent
        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        # self.ones is nn.Parameter
        self.ones = torch.nn.Parameter(self.ones.data[prototypes_to_keep, ...], False)
        # self.prototype_class_identity is torch tensor
        # so it does not need .data access for value update
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = self.prototype_class_identity.T
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

# resnet sizes
layer_filter_sizes = [7, 3, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1]
layer_strides = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
layer_paddings = [3, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
model = PPNet(224, (10*train_dataset.num_classes, 128, 1, 1), train_dataset.num_classes,
    compute_proto_layer_rf_info_v2(224, layer_filter_sizes, layer_strides, layer_paddings, 1))
model.to(device)

####################################### OPTIMIZER #######################################

# define optimizer
joint_optimizer = torch.optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-4, 'weight_decay': 1e-3},
    {'params': model.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
    {'params': model.prototype_vectors, 'lr': 3e-3},
])
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, 5)

warm_optimizer = torch.optim.Adam([
    {'params': model.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
    {'params': model.prototype_vectors, 'lr': 3e-3},
])

last_layer_optimizer = torch.optim.Adam([
    {'params': model.last_layer.parameters(), 'lr': 1e-4}
])

######################################### LOOP #########################################

def warm_only(model):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True

def joint(model):
    for p in model.backbone.parameters():
        p.requires_grad = True
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True

def last_only(model):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = False
    model.prototype_vectors.requires_grad = False
    for p in model.last_layer.parameters():
        p.requires_grad = True

def train(model, dataloader, optimizer):
    model.train()
    avg_acc = 0
    avg_loss = 0
    for image, _, label in dataloader:
        input = image.to(device)
        target = label.to(device)
        # nn.Module has implemented __call__() function
        # so no need to call .forward
        output, min_distances = model(input)
        # compute loss
        cross_entropy = torch.nn.functional.cross_entropy(output, target)
        max_dist = model.prototype_shape[1]*model.prototype_shape[2]*model.prototype_shape[3]

        # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
        # calculate cluster cost
        prototypes_of_correct_class = model.prototype_class_identity[:, label].T
        inverted_distances = torch.amax((max_dist - min_distances) * prototypes_of_correct_class, 1)
        cluster_cost = torch.mean(max_dist - inverted_distances)

        # calculate separation cost
        prototypes_of_wrong_class = 1 - prototypes_of_correct_class
        inverted_distances_to_nontarget_prototypes = \
            torch.amax((max_dist - min_distances) * prototypes_of_wrong_class, 1)
        separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

        # calculate avg cluster cost
        avg_separation_cost = \
            torch.sum(min_distances * prototypes_of_wrong_class, 1) / torch.sum(prototypes_of_wrong_class, 1)
        avg_separation_cost = torch.mean(avg_separation_cost)

        l1_mask = 1 - model.prototype_class_identity.T
        l1 = (model.last_layer.weight * l1_mask).norm(p=1)

        # evaluation statistics
        predicted = torch.argmax(output.data, 1)
        avg_acc += (predicted == target).sum().item() / len(dataloader)

        # compute gradient and do SGD step
        loss = cross_entropy + 0.8*cluster_cost - 0.08*separation_cost + 1e-4*l1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += float(loss) / len(dataloader)
    return avg_loss, avg_acc

def test(model, dataloader):
    model.eval()
    avg_acc = 0
    for image, _, label in dataloader:
        input = image.to(device)
        target = label.to(device)
        with torch.no_grad():
            output, _ = model(input)
            predicted = torch.argmax(output.data, 1)
            avg_acc += (predicted == target).sum().item() / len(dataloader)
    return avg_acc

# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader, prototype_network, prototype_layer_stride=1):
    prototype_network.eval()
    prototype_shape = prototype_network.prototype_shape
    n_prototypes = prototype_network.num_prototypes
    # saves the closest distance seen so far
    global_min_proto_dist = torch.full([n_prototypes], torch.inf)
    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = torch.zeros([n_prototypes, prototype_shape[1], prototype_shape[2], prototype_shape[3]])

    num_classes = prototype_network.num_classes
    for search_batch_input, _, search_y in dataloader:
        update_prototypes_on_batch(search_batch_input, prototype_network,
            global_min_proto_dist, global_min_fmap_patches,
            search_y, num_classes, prototype_layer_stride)

    prototype_update = global_min_fmap_patches.reshape(prototype_shape)
    prototype_network.prototype_vectors.data.copy_(prototype_update)

# update each prototype for current search batch
def update_prototypes_on_batch(search_batch, prototype_network,
        # these four will be updated
        global_min_proto_dist, global_min_fmap_patches,
        search_y, num_classes, prototype_layer_stride=1):
    prototype_network.eval()
    with torch.no_grad():
        # this computation currently is not parallelized
        protoL_input, proto_dist = prototype_network.push_forward(search_batch.to(device))

    class_to_img_index_dict = {key: [] for key in range(num_classes)}
    # img_y is the image's integer label
    for img_index, img_y in enumerate(search_y):
        class_to_img_index_dict[img_y.item()].append(img_index)

    prototype_shape = prototype_network.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]

    for j in range(n_prototypes):
        # target_class is the class of the class_specific prototype
        target_class = torch.argmax(prototype_network.prototype_class_identity[j]).item()
        # if there is not images of the target_class from this batch
        # we go on to the next prototype
        if len(class_to_img_index_dict[target_class]) == 0:
            continue
        proto_dist_j = proto_dist[class_to_img_index_dict[target_class]][:,j,:,:]

        batch_min_proto_dist_j = torch.amin(proto_dist_j)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = list(torch.unravel_index(torch.argmin(proto_dist_j), proto_dist_j.shape))
            '''
            change the argmin index from the index among
            images of the target class to the index in the entire search
            batch
            '''
            batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]

            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w

            batch_min_fmap_patch_j = protoL_input[img_index_in_batch, :,
                fmap_height_start_index:fmap_height_end_index, fmap_width_start_index:fmap_width_end_index]

            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j
            
            # get the receptive field boundary of the image patch
            # that generates the representation
            protoL_rf_info = prototype_network.proto_layer_rf_info
            rf_prototype_j = compute_rf_prototype(search_batch.shape[2], batch_argmin_proto_dist_j, protoL_rf_info)
            
            # get the whole image
            original_img_j = search_batch[rf_prototype_j[0]]
            original_img_j = torch.permute(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]
            # reverse vgg normalization
            original_img_j = original_img_j * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])

            # crop out the receptive field
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                      rf_prototype_j[3]:rf_prototype_j[4], :]
            
            # find the highly activated region of the original image
            proto_dist_img_j = proto_dist[img_index_in_batch, j, :, :]
            proto_act_img_j = torch.log((proto_dist_img_j + 1) / (proto_dist_img_j + prototype_network.epsilon))
            upsampled_act_img_j = torch.nn.functional.interpolate(proto_act_img_j[None, None], (original_img_size, original_img_size), mode='bicubic')[0, 0]
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
            # crop out the image patch with high activation as prototype image
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1], proto_bound_j[2]:proto_bound_j[3]]

            # save the whole image containing the prototype as png
            imsave(f'{args.output}-prototype-{j}-original.png', (255*original_img_j).to(torch.uint8))
            # overlay (upsampled) self activation on original image and save the result
            # ricardo: I modified a little how they do the heatmap
            rescaled_act_img_j = upsampled_act_img_j - torch.amin(upsampled_act_img_j)
            rescaled_act_img_j = rescaled_act_img_j / torch.amax(rescaled_act_img_j)
            heatmap = rescaled_act_img_j[:, :, None].cpu()
            overlayed_original_img_j = original_img_j*heatmap
            imsave(f'{args.output}-prototype-{j}-original_with_self_act.png', (255*overlayed_original_img_j).to(torch.uint8))

            # if different from the original (whole) image, save the prototype receptive field as png
            if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[1] != original_img_size:
                imsave(f'{args.output}-prototype-{j}-receptive_field.png', (255*rf_img_j).to(torch.uint8))
                overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2], rf_prototype_j[3]:rf_prototype_j[4]]
                imsave(f'{args.output}-prototype-{j}-receptive_field_with_self_act.png', (255*overlayed_rf_img_j).to(torch.uint8))

            # save the prototype image (highly activated region of the whole image)
            imsave(f'{args.output}-prototype-{j}.png', (255*proto_img_j).to(torch.uint8))

for epoch in range(args.epochs):
    tic = time()
    if epoch < 5:
        warm_only(model)
        avg_train_loss, avg_train_acc = train(model, train_loader, warm_optimizer)
    else:
        joint(model)
        avg_train_loss, avg_train_acc = train(model, train_loader, joint_optimizer)
        joint_lr_scheduler.step()
    toc = time()
    avg_test_acc = test(model, test_loader)
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Train loss: {avg_train_loss} - Train acc: {avg_train_acc} - Test acc: {avg_test_acc}')

    if (epoch+1) % 10 == 0:
        print('2nd stage')
        tic = time()
        push_prototypes(train_push_loader, model)
        last_only(model)
        for i in range(20):
            print(f'3rd stage (iteration {i+1}/20)')
            avg_train_loss, avg_train_acc = train(model, train_loader, last_layer_optimizer)
        toc = time()
        avg_test_acc = test(model, test_loader)
        print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Train loss: {avg_train_loss} - Train acc: {avg_train_acc} - Test acc: {avg_test_acc}')
    torch.save(model, args.output)