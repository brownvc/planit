from torch.utils import data
import torch
import torchvision
import random
import math
from data import ObjectCategories, RenderedScene
import copy
import numpy as np
import utils
import os
from collections import defaultdict
import pickle
import sys
from functools import cmp_to_key

## ------------------------------------------------------------------------------------------------

# Utilities for generating rotated versions of scenes (for data augmentation)
# These operator on C x H x W tensors

# NOTE: These flip_ functions assume that you've got a build of PyTorch newer than 0.4, or
#    you've manually implemented the fix to CPU-side flip() described here:
# https://github.com/pytorch/pytorch/issues/9147

def flip_horiz(x):
    return x.flip(2)

def flip_vert(x):
    return x.flip(1)

def transpose(x):
    return torch.transpose(x, 1, 2)

def rot_90(x):
    return flip_vert(transpose(x))

def rot_180(x):
    return flip_horiz(flip_vert(x))

def rot_270(x):
    return flip_horiz(transpose(x))

def create_transformed_composite(scene, objects, rot):
    assert (rot in [0, 90, 180, 270])

    # Create the transformed composite
    composite = scene.create_composite()

    # Add the rotation to all the object transformation matrices (so that the correct sin/cos
    #    values get written to the composite image)
    if rot != 0:
        rot_rad = math.radians(rot)
        c, s = math.cos(rot_rad), math.sin(rot_rad)
        # Rotation is about y = up
        rot_mat = np.array([[c, 0, -s, 0],
                            [0, 1, 0, 0],
                            [s, 0, c, 0],
                            [0, 0, 0, 1]])
        # The full process is: translate center of scene to origin, rotate, translate back
        hw = scene.size / 2
        trans_mat1 = np.array([[1, 0, 0, -hw],
                               [0, 1, 0, 0],
                               [0, 0, 1, -hw],
                               [0, 0, 0, 1]])
        trans_mat2 = np.array([[1, 0, 0, hw],
                               [0, 1, 0, 0],
                               [0, 0, 1, hw],
                               [0, 0, 0, 1]])
        full_mat = np.matmul(trans_mat2, np.matmul(rot_mat, trans_mat1))

    for obj in objects:
        if rot != 0:
            # Update the transform (used to compute object rotations)
            o_mat = np.array(obj['transform']).reshape(4, 4)
            t_mat = np.matmul(rot_mat, o_mat)
            obj['transform'] = t_mat.reshape(16).tolist()
        # Compute the location
        xmin, _, ymin, _ = obj["bbox_min"]
        xmax, _, ymax, _ = obj["bbox_max"]
        x = (xmin+xmax)/2
        y = (ymin+ymax)/2
        loc = np.array([x, 0, y, 1])
        if rot != 0:
            loc = np.matmul(full_mat, loc)
        obj['location'] = np.array([loc[0], loc[2]])

    # Add these objects in
    # composite.add_nodes(objects)
    composite.add_nodes_obb(objects)    # Use OBBs instead of full geometry

    # Get the composite image
    img = composite.get_composite(num_extra_channels=0)


    # Rotate the image
    if rot == 0:
        t_img = img
    elif rot == 90:
        t_img = rot_90(img)
    elif rot == 180:
        t_img = rot_180(img)
    elif rot == 270:
        t_img = rot_270(img)

    return t_img

## ------------------------------------------------------------------------------------------------


class LatentDataset(data.Dataset):

    def __init__(self, scene_indices=(0,4000), data_folder="bedroom_fin_256", data_root_dir=None, seed=None,
        do_rotation_augmentation=False, cat_only=False, use_same_category_batches=False, importance_order=False,
        epoch_size=None):
        super(LatentDataset, self).__init__()
        self.category_map = ObjectCategories()
        self.seed = seed
        self.data_folder = data_folder
        self.data_root_dir = data_root_dir
        self.scene_indices = scene_indices
        self.do_rotation_augmentation = do_rotation_augmentation
        self.cat_only = cat_only

        self.cat_name2index = None

        self.cat_index2scenes = None

        if self.data_root_dir is None:
            self.data_root_dir = utils.get_data_root_dir()
        
        self.catnames = self.category_map.all_non_arch_categories(self.data_root_dir, data_folder)
        self.cat_name2index = {self.catnames[i]: i for i in range(0, len(self.catnames))}
        self.n_categories = len(self.catnames)

        self.build_cat2scene()
        self.build_cats_in_scene_indices()
        
        self.cat_importances = self.category_map.all_non_arch_category_importances(self.data_root_dir, data_folder)

        # See 'prepare_same_category_batches' below for info
        self.use_same_category_batches = use_same_category_batches
        if use_same_category_batches:
            self.same_category_batch_indices = []
            assert(epoch_size is not None)
            self.epoch_size = epoch_size
        else:
            self.same_category_batch_indices = None

        self.importance_order = importance_order

    # Build a map from category index to the scene indices that contain an instance of that category
    # This ignores scene_indices and does it for the whole data folder
    def build_cat2scene(self):
        self.cat_index2scenes = defaultdict(list)
        data_root_dir = self.data_root_dir or utils.get_data_root_dir()
        data_dir = f'{data_root_dir}/{self.data_folder}'
        filename = f'{data_dir}/cat_index2scenes'
        # Create new cached map file
        if not os.path.exists(filename):
            print('Building map of category to scenes containing an instance...')
            pkls = [path for path in os.listdir(data_dir) if path.endswith('.pkl')]
            pklnames = [os.path.splitext(path)[0] for path in pkls]
            # Only get the .pkl files which are numbered scenes
            indices = [int(pklname) for pklname in pklnames if pklname.isdigit()]
            i = 0
            for idx in indices:
                i += 1
                sys.stdout.write(f'   {i}/{len(indices)}\r')
                sys.stdout.flush()
                scene = RenderedScene(idx, self.data_folder, self.data_root_dir)
                object_nodes = scene.object_nodes
                for node in object_nodes:
                    self.cat_index2scenes[node['category']].append(idx)
            pickle.dump(self.cat_index2scenes, open(filename, 'wb'))
            print('')
        # Load an existing cached map file from disk
        else:
            self.cat_index2scenes = pickle.load(open(filename, 'rb'))

    def __len__(self):
        if self.use_same_category_batches:
            return self.epoch_size
        else:
            return self.scene_indices[1]-self.scene_indices[0]

    # First, find the set of categories that occur within the scene indices
    # We do this because it's possible that there might be some category that
    #    occurs in the dataset, but only in the test set...
    def build_cats_in_scene_indices(self):
        cats_seen = {}
        for cat,scene_indices in self.cat_index2scenes.items():
            scenes = [idx for idx in scene_indices if \
                (idx >= self.scene_indices[0] and idx < self.scene_indices[1])]
            if len(scenes) > 0:
                cats_seen[cat] = True
        cats_seen = list(cats_seen.keys())
        self.cats_seen = cats_seen

    # Use at the beginning of each epoch to support loading batches of all the same category
    # NOTE: The data loader must have shuffle set to False for this to work
    def prepare_same_category_batches(self, batch_size):
        # Build a random list of category indices (grouped by batch_size)
        # This requires than length of dataset is a multiple of batch_size
        assert(len(self) % batch_size == 0)
        num_batches = len(self) // batch_size
        self.same_category_batch_indices = []
        for i in range(num_batches):
            # cat_index = random.randint(0, self.n_categories-1)
            cat_index = random.choice(self.cats_seen)
            for j in range(batch_size):
                self.same_category_batch_indices.append(cat_index)

    # 'importance' = a function of both size and observation frequency
    def sort_object_nodes_by_importance(self, object_nodes, noise=None, swap_prob=None):
        # Build list of pairs of (index, importance)
        index_imp_pairs = []
        for i in range(0, len(object_nodes)):
            node = object_nodes[i]
            cat = node["category"]
            imp = self.cat_importances[cat]
            index_imp_pairs.append((i, imp))

        # Optionally, add noise to these importance scores
        # Noise is expressed as a multiple of the standard deviation of the importance scores
        # A typical value might be really small, e.g. 0.05(?)
        if noise is not None:
            imps = [pair[1] for pair in index_imp_pairs]
            istd = np.array(imps).std()
            index_imp_pairs = [(index, imp + noise*random.normalvariate(0, istd)) for index, imp in index_imp_pairs]
        
        # Sort based on importance
        index_imp_pairs.sort(key=lambda tup: tup[1], reverse=True)

        sorted_nodes = [object_nodes[tup[0]] for tup in index_imp_pairs]

        # Optionally, swap nodes with some probabilitiy
        if swap_prob is not None:
            indices = list(range(len(sorted_nodes)))
            for i in range(len(indices)):
                if random.random() < swap_prob:
                    indices_ = list(range(len(sorted_nodes)))
                    idx1 = random.choice(indices_)
                    indices_.remove(idx1)
                    idx2 = random.choice(indices_)
                    tmp = indices[idx1]
                    indices[idx1] = indices[idx2]
                    indices[idx2] = tmp
                    tmp = sorted_nodes[idx1]
                    sorted_nodes[idx1] = sorted_nodes[idx2]
                    sorted_nodes[idx2] = tmp

        return sorted_nodes

    def order_object_nodes(self, object_nodes):
        if self.importance_order:
            object_nodes = self.sort_object_nodes_by_importance(object_nodes)
        else:
            object_nodes = object_nodes[:]
            random.shuffle(object_nodes)

        # The following extra sorting passes only apply to datasets that have second-tier objects
        # We can check for this by looking for the presence of certain object properties e.g. 'parent'
        if 'parent' in object_nodes[0]:
            # Make sure that all second-tier objects come *after* first tier ones
            def is_second_tier(node):
                return (node['parent'] != 'Wall') and \
                       (node['parent'] != 'Floor')
            object_nodes.sort(key = lambda node: int(is_second_tier(node)))

            # Make sure that all children come after their parents
            def cmp_parent_child(node1, node2):
                # Less than (negative): node1 is the parent of node2
                if node2['parent'] == node1['id']:
                    return -1
                # Greater than (postive): node2 is the parent of node1
                elif node1['parent'] == node2['id']:
                    return 1
                # Equal (zero): all other cases
                else:
                    return 0
            object_nodes.sort(key = cmp_to_key(cmp_parent_child))

        # #### TEST: make sure office_chair comes *after* desk   
        # def chair_vs_desk(node1, node2):
        #     catname1 = self.catnames[node1['category']]
        #     catname2 = self.catnames[node2['category']]
        #     if (catname1 == 'desk') and (catname2 == 'office_chair'):
        #         return -1
        #     elif (catname1 == 'office_chair') and (catname2 == 'desk'):
        #         return 1
        #     else:
        #         return 0
        # object_nodes.sort(key = cmp_to_key(chair_vs_desk))

        return object_nodes

    def get_scene(self, index, stop_prob=None):
        i = index+self.scene_indices[0]
        scene = RenderedScene(i, self.data_folder, self.data_root_dir)
        object_nodes = self.order_object_nodes(scene.object_nodes)

        # With some probability, sample a 'stop' (i.e. the end of the scene build sequence)
        if stop_prob is not None and random.random() < stop_prob:
            output_node = None
            input_nodes = object_nodes
        else:
            # Pick a random index at which to split into (a) existing objects and
            #    (b) objects yet-to-be-added.
            split_idx = random.randint(0, len(object_nodes)-1)
            # This object is the ouput node
            output_node = object_nodes[split_idx]
            # All object before this index are input nodes
            input_nodes = object_nodes[0:split_idx]

        return scene, input_nodes, output_node

    def get_scene_specific_category(self, cat_index_or_name, empty_room=False):
        if isinstance(cat_index_or_name, list):
            cat_index_or_name = random.choice(cat_index_or_name)
        if isinstance(cat_index_or_name, int):
            cat_index = cat_index_or_name
        else:
            cat_name = cat_index_or_name
            cat_index = self.cat_name2index[cat_name]

        # Pull out a scene (within scene_indices) that has an instance of this category
        scenes_for_cat = [idx for idx in self.cat_index2scenes[cat_index] if \
            (idx >= self.scene_indices[0] and idx < self.scene_indices[1])]
        scene_index = random.choice(scenes_for_cat)
        scene = RenderedScene(scene_index, self.data_folder, self.data_root_dir)
        object_nodes = self.order_object_nodes(scene.object_nodes)

        # Pick a random instance of the category
        cat_indices = [i for i in range(0, len(object_nodes)) if object_nodes[i]['category'] == cat_index]
        split_idx = random.choice(cat_indices)
        # This object is the ouput node
        output_node = object_nodes[split_idx]
        if empty_room:
            input_nodes = []   # No other objects in the scene
        else:
            input_nodes = object_nodes[0:split_idx]   # All object before this index are input nodes

        return scene, input_nodes, output_node

    def get_scene_same_category_batch(self, index):
        cat_index = self.same_category_batch_indices[index]
        return self.get_scene_specific_category(cat_index)

    # Balance training data so that we train equally often on all target categories
    def get_scene_uniform_category(self, stop_prob=None):
        if stop_prob is not None and random.random() < stop_prob:
            scene_index = random.randint(self.scene_indices[0], self.scene_indices[1]-1)
            scene = RenderedScene(scene_index, self.data_folder, self.data_root_dir)
            output_node = None
            input_nodes = self.order_object_nodes(scene.object_nodes)
            return scene, input_nodes, output_node
        else:
            cat_index = random.choice(self.cats_seen)
            return self.get_scene_specific_category(cat_index)

    def __getitem__(self,index):
        if self.seed:
            random.seed(self.seed)

        if self.use_same_category_batches:
            scene, input_nodes, output_node = self.get_scene_same_category_batch(index)
        elif self.cat_only:
            scene, input_nodes, output_node = self.get_scene(index, stop_prob=0.1)
        else:
            scene, input_nodes, output_node = self.get_scene(index)

        # Get the composite images
        if not self.do_rotation_augmentation:
            input_img = create_transformed_composite(scene, input_nodes, 0)
            if not self.cat_only:
                output_img = create_transformed_composite(scene, [output_node], 0)
        else:
            # Data augmentation: Get the composite images under a random cardinal rotation
            rot = random.choice([0, 90, 180, 270])
            input_img = create_transformed_composite(scene, input_nodes, rot)
            if not self.cat_only:
                output_img = create_transformed_composite(scene, [output_node], rot)

        # Get the category of the object
        # This is an integer index
        if output_node is None:
            cat = torch.LongTensor([self.n_categories])
        else:
            cat = torch.LongTensor([output_node["category"]])

        # Also get the count of all categories currently in the scene
        catcount = torch.zeros(self.n_categories)
        for node in input_nodes:
            catidx = node['category']
            catcount[catidx] = catcount[catidx] + 1

        # If the dataset is configured to only care about predicting the category, then we can go ahead
        #    and return now
        if self.cat_only:
            return input_img, cat, catcount

        # Select just the object mask channel from the output image
        output_img = output_img[2]
        # Put a singleton dimension back in for the channel dimension
        output_img = torch.unsqueeze(output_img, 0)
        # Make sure that it has value 1 everywhere (hack: multiply by huge number and clamp)
        output_img *= 1000
        torch.clamp(output_img, 0, 1, out=output_img)   # Clamp in place

        # Get the location of the object
        # Normalize the coordinates to [-1, 1], with (0,0) being the image center
        loc = output_node['location']
        x = loc[0]
        y = loc[1]
        w = output_img.size()[2]
        x_ = ((x / w) - 0.5) * 2
        y_ = ((y / w) - 0.5) * 2
        loc = torch.Tensor([x_, y_])

        # Get the orientation of the object
        # Here, we assume that there is no scale, and that the only rotation is about the up vector
        #  (so we can just read the cos, sin values directly out of the transformation matrix)
        xform = output_node["transform"]
        cos = xform[0]
        sin = xform[8]
        orient = torch.Tensor([cos, sin])

        # Get the object-space dimensions of the output object (in pixel space)
        # (Normalize to [0, 1])
        xsize, ysize = output_node['objspace_dims']
        xsize = xsize / w
        ysize = ysize / w
        # dims = torch.Tensor([xsize, ysize])
        dims = torch.Tensor([ysize, xsize])   # Not sure why this flip is necessary atm...

        return input_img, output_img, cat, loc, orient, dims, catcount
        # return input_img, output_img, cat, loc, orient, dims, catcount, torch.LongTensor([scene.index])





## TEST
if __name__ == "__main__":
    
    ##### Test that instances of the same category have consistent coord frames
    import torch.nn.functional as F
    import torchvision
    def inverse_xform_img(img, loc, orient, output_size):
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
        out_size = torch.Size((batch_size, img.shape[1], output_size, output_size))
        grid = F.affine_grid(matrices, out_size)
        return F.grid_sample(img, grid)

    catname = 'television'
    num_to_check = 30
    d = LatentDataset(data_folder="bedroom_graph_6x6")
    for i in range(num_to_check):
        scene, input_nodes, output_node = d.get_scene_specific_category(catname) 
        composite = scene.create_composite()
        composite.add_nodes(input_nodes)
        composite.add_node(output_node)
        img = composite.get_composite(num_extra_channels=0)
        # Get the location and orientation of the 'output' object
        xmin, _, ymin, _ = output_node["bbox_min"]
        xmax, _, ymax, _ = output_node["bbox_max"]
        x = (xmin+xmax)/2
        y = (ymin+ymax)/2
        w = img.shape[1]
        x_ = ((x / w) - 0.5) * 2
        y_ = ((y / w) - 0.5) * 2
        loc = torch.Tensor([x_, y_])
        xform = output_node["transform"]
        cos = xform[0]
        sin = xform[8]
        orient = torch.Tensor([cos, sin])
        # Inverse-transform the image accoring to the position and orientation
        img = inverse_xform_img(img.unsqueeze(0).cuda(), loc.unsqueeze(0).cuda(), orient.unsqueeze(0).cuda(), w).squeeze().cpu()
        depth_img = torchvision.transforms.ToPILImage()(img[3].unsqueeze(0))
        depth_img.save(f'coordframetest_{catname}_{i}_[{cos:.2f},{sin:.2f}].png')
