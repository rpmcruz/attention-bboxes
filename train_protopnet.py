import argparse
parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('dataset', choices=['Birds', 'StanfordCars', 'StanfordDogs'])
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batchsize', type=int, default=8)
args = parser.parse_args()

import torch, torchvision
from torchvision.transforms import v2
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

######################################### MODEL #########################################

class PPNet(torch.nn.Module):
    def __init__(self, img_size, prototype_shape, num_classes):
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

        # this has to be named features to allow the precise loading
        self.backbone = torchvision.models.resnet50(weights='DEFAULT')

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
        self.add_on_layers = torch.nn.Sequential(*add_on_layers)
       
        self.prototype_vectors = torch.nn.Parameter(torch.rand(self.prototype_shape))
        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = torch.nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)
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
        min_distances = -torch.nn.functional.max_pool2d(-distances,
            (distances.shape[2], distances.shape[3]))
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

        self.prototype_shape = list(self.prototype_vectors.size())
        self.num_prototypes = self.prototype_shape[0]

        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are consistent
        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        # self.ones is nn.Parameter
        self.ones = torch.nn.Parameter(self.ones.data[prototypes_to_keep, ...], requires_grad=False)
        # self.prototype_class_identity is torch tensor
        # so it does not need .data access for value update
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

model = PPNet(224, (2000, 128, 1, 1), train_dataset.num_classes)
model.to(device)

####################################### OPTIMIZER #######################################

# define optimizer
joint_optimizer = torch.optim.Adam([
    {'params': model.features.parameters(), 'lr': 1e-4, 'weight_decay': 1e-3},
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
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

def joint(model):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

def last_only(model):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

def train_or_test(model, dataloader, optimizer, class_specific, use_l1_mask=True):
    is_train = optimizer is not None
    tic = time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    for image, _, label in enumerate(dataloader):
        input = image.to(device)
        target = label.to(device)
        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances = model(input)
            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)
            if class_specific:
                max_dist = model.module.prototype_shape[1]*model.module.prototype_shape[2]*model.module.prototype_shape[3]

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                prototypes_of_correct_class = model.module.prototype_class_identity[:, label].t
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
                if use_l1_mask:
                    l1_mask = 1 - model.module.prototype_class_identity.t
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1) 
            else:
                min_distance = torch.amin(min_distances, 1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            predicted = torch.argmax(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    toc = time()
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(torch.sum((torch.unsqueeze(p, 2) - torch.unsqueeze(p.t, 0))**2, 1))
    print(f'* Epoch {epoch} - {toc-tic:0f}s - cross ent: {total_cross_entropy/n_batches} - cluster: {total_cluster_cost/n_batches}' +
        (f' - separaton: {total_separation_cost/n_batches} - avg separation: {total_avg_separation_cost/n_batches}' if class_specific else '') + 
        f' - accu: {n_correct/n_examples} - l1: {model.module.last_layer.weight.norm(p=1).item()}' +
        f' - p dist pair: {p_avg_pair_dist}')
    return n_correct/n_examples

for epoch in range(args.epochs):
    if epoch < 5:
        warm_only(model)
        model.train()
        train_or_test(model, train_loader, warm_optimizer, True)
    else:
        joint(model)
        joint_lr_scheduler.step()
        model.train()
        train_or_test(model, train_loader, joint_optimizer, True)

    model.eval()
    accu = train_or_test(model, test_loader, None, True)
    if (epoch+1) % 10 == 0:
        push.push_prototypes(
            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        model.eval()
        accu = train_or_test(model, test_loader, None, True)

        last_only(model)
        for i in range(20):
            model.train()
            train_or_test(model, train_loader, last_layer_optimizer, True)
            model.eval()
            accu = train_or_test(model, test_loader, None, True)
    torch.save(model, args.output)
