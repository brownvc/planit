import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torchvision
from torch.autograd import Variable
from loc_dataset import *
from PIL import Image
import scipy.misc as m
import numpy as np
import math
import utils
from matplotlib import cm

# FiLM stuff comes from here
from models.utils import *

# ---------------------------------------------------------------------------------------
# ResNet stuff

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def inverse_xform_img(img, loc, orient, scale=1):
    batch_size = img.shape[0]
    matrices = torch.zeros(batch_size, 2, 3).cuda()
    cos = orient[:, 0]
    sin = orient[:, 1]
    matrices[:, 0, 0] = cos
    matrices[:, 1, 1] = cos
    matrices[:, 0, 1] = -sin
    matrices[:, 1, 0] = sin
    matrices[:, 0, 2] = loc[:, 1]
    matrices[:, 1, 2] = loc[:, 0]
    scale_matrices = torch.stack([torch.eye(3, 3) for i in range(0, batch_size)]).cuda()
    scale_matrices[:, 0, 0] = scale
    scale_matrices[:, 1, 1] = scale
    matrices = torch.matmul(matrices, scale_matrices)
    img = img.unsqueeze(1).float() if len(img.shape) == 3 else img
    out_size = torch.Size((batch_size, 1, img.shape[1], img.shape[2])) if len(img.shape) == 3 \
               else torch.Size(img.shape)
    grid = F.affine_grid(matrices, out_size)
    return F.grid_sample(img, grid)

def forward_xform_img(img, loc, orient, scale=1):
    # First, build the inverse rotation matrices
    batch_size = img.shape[0]
    inv_rot_matrices = torch.zeros(batch_size, 3, 3).cuda()
    cos = orient[:, 0]
    sin = orient[:, 1]
    inv_rot_matrices[:, 0, 0] = cos
    inv_rot_matrices[:, 1, 1] = cos
    inv_rot_matrices[:, 0, 1] = sin
    inv_rot_matrices[:, 1, 0] = -sin
    inv_rot_matrices[:, 2, 2] = 1
    # Then, build the inverse translation matrices
    # (Apparently, x and y need to be swapped. I don't know why...)
    inv_trans_matrices = [torch.eye(3, 3) for i in range(0, batch_size)]
    inv_trans_matrices = torch.stack(inv_trans_matrices).cuda()
    inv_trans_matrices[:, 0, 2] = -loc[:, 1]
    inv_trans_matrices[:, 1, 2] = -loc[:, 0]
    # Build scaling transform matrices
    scale_matrices = torch.stack([torch.eye(3, 3) for i in range(0, batch_size)]).cuda()
    scale_matrices[:, 0, 0] = 1/scale
    scale_matrices[:, 1, 1] = 1/scale
    # Multiply them to get the full affine matrix
    inv_matrices = torch.matmul(scale_matrices, inv_rot_matrices)
    inv_matrices = torch.matmul(inv_matrices, inv_trans_matrices)
    # Discard the last row (affine_grid expects 2x3 matrices)
    inv_matrices = inv_matrices[:, 0:2, :]
    # Finalize
    img = img.unsqueeze(1).float() if len(img.shape) == 3 else img
    out_size = torch.Size((batch_size, 1, img.shape[1], img.shape[2])) if len(img.shape) == 3 \
               else torch.Size(img.shape)
    grid = F.affine_grid(inv_matrices, out_size)
    return F.grid_sample(img, grid)

def doublesize_zero_padding(img):
    w, h = img.shape[-2:]
    nw, nh = w//2, h//2
    return F.pad(img, (nh, nh, nw, nw), "constant", 0)

class Coords(nn.Module):

    def __init__(self, add_r=True):
        super().__init__()
        self.add_r = add_r

    def forward(self, x):
        """
        Args:
            x: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = x.size()

        x_coord = torch.arange(x_dim).float().repeat(1, y_dim, 1)
        y_coord = torch.arange(y_dim).float().repeat(1, x_dim, 1).transpose(1, 2)

        x_coord = x_coord / (x_dim - 1)
        y_coord = y_coord / (y_dim - 1)

        x_coord = x_coord * 2 - 1
        y_coord = y_coord * 2 - 1

        x_coord = x_coord.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        y_coord = y_coord.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            x,
            x_coord.type_as(x),
            y_coord.type_as(x)], dim=1)

        if self.add_r:
            rr = torch.sqrt(torch.pow(x_coord.type_as(x) - 0.5, 2) + torch.pow(y_coord.type_as(x) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2, num_input_channels=17, use_fc=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=4, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc
        if use_fc:
            self.fc = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x

class ResBlock(nn.Module):

    def __init__(self, inplanes):
        super(ResBlock, self).__init__()
        self.inplanes = inplanes
        block = BasicBlock
        self.layer1 = self._make_layer(block, inplanes, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        return x

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

# ---------------------------------------------------------------------------------------
# The rest of the model

# I'm implementing UpConvBlock not as a module, but as a function that returns a list of
#    modules. These will be used to construct a FiLMSequential module.
# This is because the way FiLMSequential works is to look for FiLM modules in its list of
#    submodules, and it doesn't have a way to recursively check those submodules
def film_upconv_block(inplanes, outplanes, coord=True):
    upsamp = nn.Upsample(mode='nearest', scale_factor=2)
    if coord:
        inplanes = inplanes+3
        coords = Coords(add_r=True)
    conv = nn.Conv2d(inplanes, outplanes, stride=1, kernel_size=3, padding=1)
    bn = nn.BatchNorm2d(outplanes)
    film = FiLM(outplanes)
    act = nn.LeakyReLU()
    if coord:
        return upsamp, coords, conv, bn, film, act
    else:
        return upsamp, conv, bn, film, act

class EdgeTypeEmbedding(nn.Module):
    def __init__(self, edge_embedding_size):
        super(EdgeTypeEmbedding, self).__init__()
        self.num_distance_types = 3
        self.num_base_types = 5
        embed_size = 5
        self.base_embed = nn.Embedding(self.num_base_types, embed_size)
        self.distance_embed = nn.Embedding(self.num_distance_types, embed_size)
        self.linear = nn.Linear(2*embed_size, edge_embedding_size)
    
    def forward(self, x):
        dist_index = x % self.num_distance_types
        base_index = (x - dist_index) // self.num_distance_types
        embed = torch.cat((self.base_embed(base_index), self.distance_embed(dist_index)), -1)
        return self.linear(F.relu(embed))

class Model(nn.Module):
    def __init__(self, num_classes, num_input_channels, edge_embedding_size=10):
        super(Model, self).__init__()

        # This net takes an integer label for a type of relationship edge, and it produces an
        #    embedding vector for that label
        # I'm currently doing this using an nn.Embedding layer. You could also use FC layers...
        # self.cond_input_net = nn.Embedding(15, edge_embedding_size)
        self.cond_input_net = EdgeTypeEmbedding(edge_embedding_size)

        # This is the main network. It assumes it receives an edge embedding as an extra input,
        #    which gets threaded through all the submodules and is used to drive the FiLM layers.
        self.main_net = FiLMSequential(edge_embedding_size,
            #### Encoder
            #nn.Dropout(p=0.2),
            resnet34(num_input_channels=num_input_channels),

            #### Decoder
            nn.Dropout(p=0.1),
            *film_upconv_block(512, 256, coord=False),
            nn.Dropout(p=0.2),
            *film_upconv_block(256, 128, coord=False),
            nn.Dropout(p=0.2),
            *film_upconv_block(128, 64),
            *film_upconv_block(64, 32),
            *film_upconv_block(32, 16),
            *film_upconv_block(16, 8),
            nn.Dropout(p=0.1),
            *film_upconv_block(8, 4),
            nn.Conv2d(4,num_classes,1,1)
        )

        # The overall model combines both of the above networks: it takes as input the scene image +
        #    the edge type label.
        self.model = FiLMNetworkPair(self.cond_input_net, self.main_net)

    def forward(self, x, edge_type_label):
        return self.model(x, edge_type_label)
        
# ---------------------------------------------------------------------------------------
# Customize dataloader
def test_collate(batch):
    # This is only used in *TEST* time
    # We are assuming only batch_size == 1 is used
    # batch - list of (inputs, target, anchor_loc, anchor_orient, torch.Tensor(edge_type), torch.Tensor(target_node_id))
    assert (len(batch) == 1), f"A test dataloader must have batch_size == 1, but get {len(batch)}"
    batch = batch[0]
    inputs = [item[0] for item in batch]
    target = [item[1] for item in batch]
    loc = [item[2] for item in batch]
    orient = [item[3] for item in batch]
    edge_type = [item[4] for item in batch]
    node_id = [item[5] for item in batch]

    return [torch.stack(x, 0) for x in [inputs, target, loc, orient, edge_type, node_id]]

# ---------------------------------------------------------------------------------------
# Training

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Location Training with Auxillary Tasks')
    parser.add_argument('--data-folder', type=str, default="bedroom_fin_256_obb_dims_coll", metavar='S')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N')
    parser.add_argument('--last-epoch', type=int, default=-1, metavar='N')
    parser.add_argument('--train-size', type=int, default=5000, metavar='N')
    parser.add_argument('--save-dir', type=str, default="fcn_final", metavar='S')
    parser.add_argument('--ablation', type=str, default=None, metavar='S')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N')
    parser.add_argument('--eps', type=float, default=1e-6, metavar='N')
    parser.add_argument('--centroid-weight', type=float, default=10, metavar="N")

    parser.add_argument('--test', action="store_true")
    parser.add_argument('--test-random', action="store_true")
    parser.add_argument('--test-single', action="store_true")
    args = parser.parse_args()

    from tensorboardX import SummaryWriter
    
    writer_fname = '/runs/test/' if args.test else '/runs/FiLM/'
    writer_fname = args.save_dir + writer_fname + args.data_folder

    save_dir = args.save_dir + "/FiLM/" + args.data_folder

    utils.ensuredir(save_dir)
    batch_size = 16
    data_root_dir = utils.get_data_root_dir()

    writer = SummaryWriter(writer_fname)

    categories = ObjectCategories().all_non_arch_categories(data_root_dir, args.data_folder)
    num_categories = len(categories)

    num_input_channels = num_categories+9 # 1 additional mask channel

    logfile = open(f"{save_dir}/log_location.txt", 'w')
    def LOG(msg):
        print(msg)
        logfile.write(msg + '\n')
        logfile.flush()

    LOG('Building model...')
    model = Model(num_classes=num_categories+1, num_input_channels=num_input_channels)

    weight = [args.centroid_weight for i in range(num_categories+1)]
    weight[0] = 1
    #print(weight)

    weight = torch.from_numpy(np.asarray(weight)).float().cuda()
    cross_entropy = nn.CrossEntropyLoss(weight=weight)
    #cross_entropy = nn.CrossEntropyLoss()
    #mse = nn.MSELoss()
    #softmax = nn.Softmax()

    LOG('Converting to CUDA...')
    model.cuda()
    cross_entropy.cuda()

    LOG('Building dataset...')

    train_dataset = FCNDatasetGraph(
        scene_indices = (0, args.train_size),
        data_folder = args.data_folder,
        data_root_dir = data_root_dir
    )
    validation_dataset = FCNDatasetGraph(
        scene_indices = (args.train_size, args.train_size+batch_size),
        data_folder = args.data_folder,
        data_root_dir = data_root_dir
    )

    LOG('Building data loader...')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        num_workers = args.num_workers,
        shuffle = True
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size = batch_size,
        num_workers = args.num_workers,
        shuffle = False
    )

    ### ------------
    # Test time code goes here
    LOG('Prepare for testing...')
    # Controal the behavior of testing
    test_size = batch_size
    test_seed = None if args.test_random else 1
    test_mode = 1 if args.test_single else 0 # 0 for multiple scene, 1 for single scene
    test_index = args.train_size
    if args.test and args.last_epoch == -1:
        args.last_epoch = int(input("Specify last epoch for testing: "))
    
    def build_test_data():
        # Build Test dataset
        test_multiple_scene_dataset = FCNDatasetGraph2(
            scene_indices = (test_index, test_index+test_size),
            data_folder = args.data_folder,
            data_root_dir = data_root_dir,
            seed = test_seed
        )
        test_single_scene_dataset = FCNDatasetGraphOneScene(
            scene_indices = test_index,
            data_folder = args.data_folder,
            data_root_dir = data_root_dir
        )

        test_dataset = test_multiple_scene_dataset if test_mode == 0 else test_single_scene_dataset

        # Build Test dataloader
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size = 1,
            collate_fn=test_collate,
            num_workers = 1, # args.num_workers
            shuffle = False
        )
        return test_loader

    test_loader = build_test_data()
    # Test time misc
    test_cat2idx = test_loader.dataset.get_scene(0).cat_to_index
    test_idx2cat = dict(zip(test_cat2idx.values(), test_cat2idx.keys()))

    if args.test:
        print("\nAll possible categories:")
        print(test_idx2cat)
        test_id = int(input("Enter ID (-1 for exact match): "))
    # --------------

    LOG('Building optimizer...')
    optimizer = optim.Adam(model.parameters(),
        lr = args.lr,
        weight_decay = 1e-8,
    )

    if args.last_epoch < 0:
        load = False
        starting_epoch = 0
    else:
        load = True
        last_epoch = args.last_epoch

    if load:
        LOG('Loading saved models...')
        model.load_state_dict(torch.load(f"{save_dir}/location_{last_epoch}.pt"))
        optimizer.load_state_dict(torch.load(f"{save_dir}/location_optim_backup.pt"))
        starting_epoch = last_epoch + 1
    
    current_epoch = starting_epoch
    num_seen = last_epoch * 10000 if load else 0

    model.train()
    LOG(f'=========================== Epoch {current_epoch} ===========================')

    def train():
        global num_seen, current_epoch
        for batch_idx, (data, target, loc, orient, edge_type) \
                       in enumerate(train_loader):
            data, target, edge_type = data.cuda(), target.cuda(), edge_type.long().cuda()
            
            optimizer.zero_grad()
            output = model(data, edge_type)
            loss = cross_entropy(output,target.long())
            
            loss.backward()
            optimizer.step()

            num_seen += batch_size
            writer.add_scalar('iterations/loss', loss, num_seen)
            if num_seen % 800 == 0:
                LOG(f'Examples {num_seen}/10000')
            if num_seen % 10000 == 0:
                # validate()
                test()
                model.train()
                current_epoch += 1
                LOG(f'=========================== Epoch {current_epoch} ===========================')
                if current_epoch % 10 == 0:
                    torch.save(model.state_dict(), f"{save_dir}/location_{current_epoch}.pt")
                    torch.save(optimizer.state_dict(), f"{save_dir}/location_optim_backup.pt")

    def validate():
        model.eval()
        for batch_idx, (data, target, loc, orient, edge_type) in enumerate(validation_loader):
            data, target, edge_type = data.cuda(), target.cuda(), edge_type.long().cuda()
            loc, orient = loc.cuda(), orient.cuda()

            output = model(data, edge_type)
            output = F.softmax(output)
            
        in_wall = forward_xform_img(data[:,1:2,...], loc, orient, 2)
        #in_wall = utils.nearest_downsample(in_wall, in_wall.shape[2]//(target.shape[2]))
        in_wall = utils.nearest_downsample(in_wall, in_wall.shape[2]//(target.shape[2]//2))
        in_wall = doublesize_zero_padding(in_wall)

        target = forward_xform_img(target, loc/2, orient, 1).repeat(1,3,1,1)

        # 0 channel for nothing
        # ('wardrobe_cabinet': 0, 'double_bed': 2, 'desk': 4, 'single_bed': 7) + 1
        wardrobe_output = forward_xform_img(output[:,1:2,...], loc/2, orient, 1).repeat(1,3,1,1)
        dbed_output = forward_xform_img(output[:,3:4,...], loc/2, orient, 1).repeat(1,3,1,1)
        sbed_output = forward_xform_img(output[:,8:9,...], loc/2, orient, 1).repeat(1,3,1,1)
        # combine wall and bed location
        target[:,0:1,:,:] += in_wall
        wardrobe_output[:,0:1,:,:] += in_wall*0.05
        dbed_output[:,0:1,:,:] += in_wall*0.05
        sbed_output[:,0:1,:,:] += in_wall*0.025

        in_wall_img = vutils.make_grid(in_wall, nrow=8, normalize=False, scale_each=False)
        #target_img = vutils.make_grid(target.unsqueeze(1), nrow=8, normalize=False, scale_each=True)
        target_img = vutils.make_grid(target, nrow=8, normalize=False, scale_each=True)
        wardrobe_img = vutils.make_grid(wardrobe_output, nrow=8, normalize=True, scale_each=True)
        dbed_img = vutils.make_grid(dbed_output, nrow=8, normalize=True, scale_each=True)
        sbed_img = vutils.make_grid(sbed_output, nrow=8, normalize=True, scale_each=True)
        #writer.add_image('input_wall', in_wall_img, num_seen)
        writer.add_image('target_distribution', target_img, num_seen)
        writer.add_image('wardrobe_distribution', wardrobe_img, num_seen)
        #writer.add_image('double_bed_distribution', dbed_img, num_seen)
        #writer.add_image('single_bed_distribution', sbed_img, num_seen)
    
    def test(object_id=-1):
        # object_id: The id of object to visualize. It is as stored in node["category"].
        #   When you actually slice out the output/target, use object_id + 1 because 0 is used as "no object" there
        #   Use -1 to slice out the channel of target for each batch

        model.eval()
        input_masks, outputs, target_nids = [], [], []
        count = 0
        for batch_idx, (inputs, target, anchor_locs, anchor_orients, edge_types, target_nid) in enumerate(test_loader):
            inputs, target, edge_types = inputs.cuda(), target.cuda(), edge_types.long().cuda()
            anchor_locs, anchor_orients = anchor_locs.cuda(), anchor_orients.cuda()

            # Test time code here
            num_transforms = inputs.shape[0]
            out = None
            anchor_mask = None
            for i in range(num_transforms):
                data, out_target, edge_type = inputs[i:(i+1), ...], target[i:(i+1), ...], edge_types[i, ...]
                loc, orient = anchor_locs[i:(i+1), ...], anchor_orients[i:(i+1), ...]
                
                # Compute and combine outputs
                out_logits = model(data, edge_type)

                ### Save figure
                save_figure_local(data, F.softmax(out_logits), target_nid+1, count, num_seen)
                count += 1
                ###

                out_logits = forward_xform_img(out_logits, loc/2, orient, 1)
                out = out_logits if out is None else out + out_logits
                # out_prop = F.softmax(out_logits)
                # out = out_prop if out is None else out * out_prop
                

                # Prepare data for visualization
                mask = anchor_mask_colored(data[:,-1,...].unsqueeze(1), edge_type)
                mask = forward_xform_img(mask, loc, orient, 2)
                anchor_mask = mask if anchor_mask is None else mask + anchor_mask

            # Save data of rooms into list for visualization
            # Masks have shape [batch_size, 6, H, W], where the six channels are
            #     room_mask, wall_mask, target_object, target_direction, anchor distance mask, anchor base mask
            data = forward_xform_img(data, loc, orient, 2)
            input_masks.append(torch.cat((data[:,[0,3],...], out_target, anchor_mask), 1))
            out = F.softmax(out)
            outputs.append(out)
            target_nids.append(target_nid[-1])

            ###
            # Save some figures
            save_figure(data, out, target_nid[0]+1, batch_idx, num_seen)
            ###
        
        ### ----- Visualization code
        input_masks, outputs, target_nids = torch.cat(input_masks, 0), torch.cat(outputs, 0), torch.cat(target_nids, 0)

        input_masks = utils.nearest_downsample(input_masks, input_masks.shape[-1]//(outputs.shape[-1]//2))
        input_masks = doublesize_zero_padding(input_masks)

        # 0 channel for nothing, thus we need to pulse 1
        # ('wardrobe_cabinet': 0, 'stand': 1, 'double_bed': 2, 'desk': 4, 'single_bed': 7, 'armchair': 9) + 1
        obj_output, obj_layout = combine_masks(input_masks.clone(), outputs, object_id+1, target_nids+1)
        # Put them side-by-side
        obj_output = torch.cat((obj_output, obj_layout), -1)

        obj_img = vutils.make_grid(obj_output, nrow=1, normalize=True)
        object_name = test_idx2cat[object_id] if object_id != -1 else "match"
        writer.add_image(f'test_{object_name}_distribution', obj_img, num_seen)
        
        utils.ensuredir(f"test_fcn_edge/{num_seen}/")
        torchvision.utils.save_image(obj_img, f"test_fcn_edge/{num_seen}/result.png")

    while True:
        if args.test:
            num_seen = test_index
            test(object_id=test_id)
            test_index = test_index + 1 if test_mode == 1 else test_index + test_size
            test_loader = build_test_data()
            input('...Load next test scene ...')
        else:
            train()
