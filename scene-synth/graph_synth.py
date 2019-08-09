# So that we can refer to stuff from ../render
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'render'))
#import visualizations

from data import *
import random
import scipy.misc as m
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from models import *
from torch.autograd import Variable
from PIL import Image
import copy
from model_prior import *
from priors.observations import ObjectCollection
from utils import stdout_redirected
from models.graph import GraphNetConfig, GraphNet
from dims import Model as DimsModel
from orient import Model as OrientModel
from loc import Model as LocationModel
from math_utils.OBB import OBB
from math_utils import Transform
from filters.global_category_filter import *
import time
import json

model_root_dir = '.'

model_dir = "train"

model_dir = f'{model_root_dir}/{model_dir}'

room_type = "bedroom"

graph_dir = f""
config_dir = f""
loc_dir = f""
orient_dir = f""
dims_dir = f""
data_dir = ""
loc_gcn_dir = ""
loc_gcn_epoch = 6
scene_indices = range(5850,5910)
save_dir = ""

with open('data/'+data_dir+'/graph/stats_phase1.json', 'r') as f:
    stats = json.load(f)

with open('data/'+data_dir+'/graph/stats_phase2.json', 'r') as f:
    stats_phase_2 = json.load(f)

keep_edge_type = stats_phase_2["keep_edge_type"]
seed = 42

temperature_pixel = 0.8
temperature_cat = 1

save_vis = False
trials = 1

floor_top = 0.1209 #hard code as that seems to be a constant

def run_full_synth():
    a = SceneSynth(data_dir=data_dir)

    save_vis = False
    trials = 1

    a.synth(scene_indices, trials=trials, save_dir=save_dir, temperature_pixel=temperature_pixel, \
            seed=seed, save_visualizations=save_vis)

class SceneSynth():
    """
    Class that synthesizes scenes
    based on the trained models
    """
    def __init__(self, data_dir, data_root_dir=None, size=256):
        """
        Parameters
        ----------
        location_epoch, rotation_epoch, continue_epoch (int):
            the epoch number of the respective trained models to be loaded
        data_dir (string): location of the dataset relative to data_root_dir
        data_root_dir (string or None, optional): if not set, use the default data location,
            see utils.get_data_root_dir
        size (int): size of the input image
        """
        Node.warning = False
        self.data_dir_relative = data_dir #For use in RenderedScene
        if not data_root_dir:
            self.data_root_dir = utils.get_data_root_dir()
        self.data_dir = f"{self.data_root_dir}/{data_dir}"
        
        self.category_map = ObjectCategories()
        #Loads category and model information
        self.categories, self.cat_to_index, self.cat_importance = self._load_category_map()
        self.num_categories = len(self.categories)

        #Misc Handling
        self.pgen = ProjectionGenerator()

        self.possible_models = self._load_possible_models()
        self.model_symmetry_groups = self._group_models_by_symmetry()
        self.model_set_list = self._load_model_set_list()
        
        #Loads trained models and build up NNs
        self.model_graph = self._load_graph_model()
        self.model_location = self._load_location_model()
        self.model_orient = self._load_orient_model()
        self.model_dims = self._load_dims_model()

        self.softmax = nn.Softmax(dim=1)
        self.softmax.cuda()
        
        self.model_sampler = ModelPrior()
        self.model_sampler.load(self.data_dir)

        self.obj_data = ObjectData()
        self.object_collection = ObjectCollection()

    
    def _load_category_map(self):
        categories = self.category_map.all_non_arch_categories(self.data_root_dir, self.data_dir_relative)
        cat_to_index = {categories[i]:i for i in range(len(categories))}
        cat_importance = self.category_map.all_non_arch_category_importances(self.data_root_dir, self.data_dir_relative)
        return categories, cat_to_index, cat_importance

    def _load_graph_model(self, graph_dir=graph_dir):
        with open(config_dir, 'rb') as f:
            config = pickle.load(f)
        for (key, value) in config.items():
            setattr(GraphNetConfig, key, value)
        GraphNetConfig.compute_derived_attributes()
        model_graph = GraphNet()
        model_graph.load_state_dict(torch.load(graph_dir))
        model_graph.eval()

        return model_graph
    
    def _load_location_model(self, loc_dir=loc_dir):
        model_location = LocationModel(num_classes=self.num_categories+1, num_input_channels=self.num_categories+9)
        model_location.load_state_dict(torch.load(loc_dir))
        model_location.eval()
        model_location.cuda()

        return model_location
    
    def _load_orient_model(self, orient_dir=orient_dir):
        model_orient = OrientModel(10, 40, self.num_categories+8)
        model_orient.load(orient_dir)
        model_orient.eval()
        model_orient.snapping = True
        model_orient.testing = True
        model_orient.cuda()
        return model_orient

    def _load_dims_model(self, dims_dir=dims_dir):
        model_dims = DimsModel(10, 40, self.num_categories+8)
        model_dims.load(dims_dir)
        model_dims.eval()
        model_dims.cuda()
        return model_dims

    def _load_possible_models(self, model_freq_threshold=0):
        #model_freq_threshold: discards models with frequency less than the threshold
        possible_models = [[] for i in range(self.num_categories)]
        with open(f"{self.data_dir}/model_frequency") as f:
            models = f.readlines()

        with open(f"{self.data_dir}/model_dims.pkl", 'rb') as f:
            dims = pickle.load(f)

        models = [l[:-1].split(" ") for l in models]
        models = [(l[0], int(l[1])) for l in models]
        self.models = models
        for model in models:
            category = self.category_map.get_final_category(model[0])
            if not self.category_map.is_arch(category):
                possible_models[self.cat_to_index[category]].append(model)

        for i in range(self.num_categories):
            total_freq = sum([a[1] for a in possible_models[i]])
            possible_models[i] = [(a[0], dims[a[0]]) for a in possible_models[i] if a[1]/total_freq > model_freq_threshold]

        return possible_models

    def _group_models_by_symmetry(self): #Whatever
        fs = [self.category_map.is_not_symmetric,
              self.category_map.is_radially_symmetric,
              self.category_map.is_front_back_reflect_symmetric,
              self.category_map.is_left_right_reflect_symmetric,
              self.category_map.is_two_way_rotate_symmetric,
              self.category_map.is_four_way_rotate_symmetric,
              self.category_map.is_corner_1_symmetric,
              self.category_map.is_corner_2_symmetric]
        model_symmetry_groups = [[] for i in range(8)]
        for (modelId, _) in self.models:
            sym_classes = set(self.category_map.get_symmetry_class(modelId))
            for (i, f) in enumerate(fs):
                if f(sym_classes):
                    if i in [6,7] and "_mirror" in modelId:
                        model_symmetry_groups[13-i].append(modelId)
                    else:
                        model_symmetry_groups[i].append(modelId)

        return model_symmetry_groups
    
    def _load_model_set_list(self):
        possible_models = self.possible_models
        obj_data = ObjectData()
        model_set_list = [None for i in range(self.num_categories)]
        for category in range(self.num_categories):
            tmp_dict = {}
            for model in possible_models[category]:
                setIds = [a for a in obj_data.get_setIds(model[0]) if a != '']
                for setId in setIds:
                    if setId in tmp_dict:
                        tmp_dict[setId].append(model[0])
                    else:
                        tmp_dict[setId] = [model[0]]
            model_set_list[category] = \
                [value for key,value in tmp_dict.items() if len(value) > 1]
        
        return model_set_list

    def get_relevant_models(self, category, modelId):
        """
        Given a category and a modelId, return all models that are relevant to it
        Which is: the mirrored version of the model,
        plus all the models that belong to the same model set
        that appear more than model_freq_threshold (set to 0.01)
        See _load_possible_models and _load_model_set_list

        Parameters
        ----------
        category (int): category of the object
        modelId (String): modelId of the object

        Return
        ------
        set[String]: set of all relevant modelIds
        """
        relevant = set()
        if "_mirror" in modelId:
            mirrored = modelId.replace("_mirror", "")
        else:
            mirrored = modelId + "_mirror"
        if mirrored in self.possible_models[category]:
            relevant.add(mirrored)

        for model_set in self.model_set_list[category]:
            if modelId in model_set:
                relevant |= set(model_set)
        
        return relevant

    def is_second_tier(self, category):
        return self.categories[category] in self.second_tiers

    def synth(self, room_ids, trials=1, size=256, samples=64, save_dir=".", \
              temperature_cat=0.25, temperature_pixel=0.4, min_p=0.5, max_collision=-0.1, seed=seed, \
              save_visualizations=False):
        """
        Synthesizes the rooms!

        Parameters
        ----------
        room_ids (list[int]): indices of the room to be synthesized, loads their
            room arthicture, plus doors and windows, and synthesize the rest
        trials (int): number of layouts to synthesize per room
        size (int): size of the top-down image
        samples (int): size of the sample grid (for location and category)
        save_dir (str): location where the synthesized rooms are saved
        temperature_cat, temperature_pixel (float): temperature for tempering,
            refer to the paper for more details
        min_p (float): minimum probability where a model instance + orientation can be accepted
        max_collision (float): max number of collision penetration, in meters, that are allowed to occur
            This is not the only collision criteria, two more are hard coded, see SynthRoom._get_collisions
        """
        t0 = time.time()
        for room_id in room_ids:
            if seed:
                random.seed(seed+room_id)
            for trial in range(trials):
                if seed:
                    cur_seed = random.randint(0,10000)
                else:
                    cur_seed = None
                self.synth_room(room_id, trial, size, samples, save_dir, temperature_cat, temperature_pixel, min_p, max_collision, cur_seed, save_visualizations)

        t1 = time.time()
        print(t1-t0)

    def synth_room(self, room_id, trial, size, samples, \
                   save_dir, temperature_cat, temperature_pixel, \
                   min_p, max_collision, seed=None, save_visualizations=False):
        """
        Synthesize a single room, see synth for explanation of some most paramters
        """
        torch.manual_seed(seed)
        random.seed(seed)
        trial = 0
        while True:
            room = SynthRoom(room_id, trial, size, samples, self, temperature_cat, temperature_pixel, min_p, max_collision, seed, save_dir, save_visualizations)
            if room.synthesize():
                break
            trial += 1 

    # Do partial scene completion
    def complete_room(self, scene, save_dir, trials=1, size=256, samples=64, temperature_cat=0.25, temperature_pixel=0.4, \
                      min_p=0.5, max_collision=-0.1, seed=seed, save_visualizations=False):
        random.seed(seed + scene.index)
        for trial in range(trials):
            cur_seed = random.randint(0,10000)
            room = SynthRoom(scene, trial, size, samples, self, temperature_cat, temperature_pixel, min_p, max_collision, cur_seed, save_dir, save_visualizations, save_initial_state=True)
            # random.seed(random.randint(0,100000))
            # torch.manual_seed(random.randint(0,100000))
            room.synthesize()

    # Do next-object-suggestion
    def suggest_next_object(self, scene, save_dir, trials=1, size=256, samples=64, temperature_cat=0.25, temperature_pixel=0.4, \
                      min_p=0.5, max_collision=-0.1, seed=seed, save_visualizations=False):
        random.seed(seed + scene.index)
        for trial in range(trials):
            cur_seed = random.randint(0,10000)
            room = SynthRoom(scene, trial, size, samples, self, temperature_cat, temperature_pixel, min_p, max_collision, cur_seed, save_dir, save_visualizations, save_initial_state=True)
            room.synthesize(num_steps=1)


class SynthStep():
    """
    Handles everything happening for a single object insertion
    Multiple attempts can happen for the "step", so we know what stuff have been attemped
    after each backtrack. The step is erased (and we move to an earlier step)
    if it is involved in backtrack for too many times
    """
    def __init__(self, synth_room, composite, graph_node):
        """
        Parameters
        ----------
        synth_room: links back to the room
        composite: RenderedComposite describing the room state before, this should not change
            for the entire life cycle of the step
        graph_node: the node this step is trying to insert, fixed by definition
        """
        self.synth_room = synth_room
        self.graph_node = graph_node
        
        self.composite_before = composite
        #Get render to be used by NN
        self.current_room = self.composite_before.get_composite(num_extra_channels=0)
        
        #Create location map, parts of which will be zeroed out after attempts
        self.location_map = self._create_location_map()
        #Store existing collisions so we don't count them as violations
        self.existing_collisions = self.synth_room._get_collisions()
        self.num_samples = 0
        
        #Depending on the type of the node, we allow different number of 
        if graph_node.middle_of_chain:
            self.max_allowed_samples = 1
        else:
            #More constraining nodes deserves more attempts
            self.max_allowed_samples = math.floor(self.graph_node.how_constraining / 0.15) + 2
            if graph_node.is_chain:
                #chains are hard UwU
                self.max_allowed_samples += 4

        if len(self.synth_room.synth_steps) > 0:
            if (not self.graph_node.is_chain) and self.synth_room.synth_steps[-1].graph_node.is_chain:
                #as it's hard to get chains to work UwU we try not to backtrack into chains
                self.max_allowed_samples + 4
        self.total_error = 0 #need to init this because we don't verify 2nd tier
    
    def sample_next(self):
        """
        Make an attempt at sample a configuration
        """
        #If we have tried for the max allowed number of times, return, failed.
        if self.num_samples >= self.max_allowed_samples:
            self.graph_node.synth_node = None
            return False, None
        success, node_or_conflicts = self.standard_sample()
        if success:
            #Suceess, update node info, update composite for next step
            synth_node = node_or_conflicts
            self.graph_node.synth_node = synth_node
            composite_new = copy.deepcopy(self.composite_before)
            category, sin, cos = synth_node.category, synth_node.sin, synth_node.cos
            composite_new.add_height_map(synth_node.get_render(), category, sin, cos)
            self.num_samples += 1
            return True, composite_new
        else:
            #Failure, return, specifying the source conflicts so the main loop knows where to backtrack to
            self.graph_node.synth_node = None
            conflict_ids = node_or_conflicts
            return False, conflict_ids

    def contains_node_ids(self, node_ids):
        if self.graph_node.id in node_ids:
            return True
        else:
            return False

    def standard_sample(self):
        """
        Sample attributes for instantiation
        """
        #Name is "standard"_sample because we tried other stuff that don't work, obviously

        if self.num_samples > 0:
            #calculate the size of the region we want to zero out in the location map
            #increasingly aggressive as the number of attempts increases
            #not really radius because it's a square
            r = int(10*(1.2**self.num_samples))
            if self.graph_node.is_chain:
                r *= 2 #chain is hard
            x, y = int(self.graph_node.synth_node.x), int(self.graph_node.synth_node.y)
            #Zero out around previous regions, more for chains
            self.location_map[x-r:x+r+1,y-r:y+r+1] = 0

        #number of attempts at sampling a valid configuration
        #different from num_samples, every sample is valid, but might conflict with later steps
        #thus needing a backtrack
        self.num_trials = 0
        #hack in more trials for harder steps
        if len(self.synth_room.synth_steps) > 0:
            if (not self.graph_node.is_chain) and self.synth_room.synth_steps[-1].graph_node.is_chain:
                self.num_trials = -10
        else:
            self.num_trials -= 30 #Well if it's the first object we might as well try until we die

        graph_node = self.graph_node
        
        synthesizer = self.synth_room.synthesizer
        category = synthesizer.cat_to_index[graph_node.category_name]
        id = graph_node.id

        #Record all objects that causes a conflict that lead to a failed trial
        #Used for determine where to backtrack to
        conflict_ids = set()

        while True: #No infinite loop, I promise
            x,y = self._sample_location()
            if x is None:
                return False, list(conflict_ids)

            w = 256
            x_ = ((x / w) - 0.5) * 2
            y_ = ((y / w) - 0.5) * 2

            loc = torch.Tensor([x_, y_]).unsqueeze(0).cuda()
                
            orient = torch.Tensor([math.cos(0), math.sin(0)]).unsqueeze(0).cuda()
            input_img = self.current_room.unsqueeze(0).cuda()
            input_img_orient = inverse_xform_img(input_img, loc, orient, 64)
            noise = torch.randn(1, 10).cuda()
            orient = synthesizer.model_orient.generate(noise, input_img_orient, category)

            sin, cos = float(orient[0][1]), float(orient[0][0])
            
            #We don't need to predict dimension if either:
            #   is a spoke, and a spoke is already sampled, as all spokes share same modelId
            #   is a part of a chain, and a same category object exist in the chain,
            #    since all chain objects of same categories share modelId
            best_modelId, best_dims = self.get_modelId_from_peers()

            #If not the case, we predict dims and sample a model from dims
            if best_modelId is None:
                input_img_dims = inverse_xform_img(input_img, loc, orient, 64)
                noise = torch.randn(1, 10).cuda()
                dims = synthesizer.model_dims.generate(noise, input_img_dims, category)
                dims_numpy = dims.detach().cpu().numpy()[0,::-1]
            
                #A bunch of hacks
                modelIds = synthesizer.possible_models[category]
                sym_idx = self.graph_node.sym_idx
                symmetry_group = synthesizer.model_symmetry_groups[sym_idx]
                modelIds = [m for m in modelIds if m[0] in symmetry_group]
                if len(modelIds) == 0:
                    modelIds = synthesizer.possible_models[category] #hack
                
                scores = []

                for (modelId, dims_gt) in modelIds:
                    l2 = (dims_gt[0]-dims_numpy[0])**2 + (dims_gt[1]-dims_numpy[1])**2
                    scores.append((modelId, l2, dims_gt))
                important = []
                others = []

                for synth_node in self.synth_room.object_nodes:
                    if synth_node.category == category:
                        important.append(synth_node.modelId)
                    elif ((synth_node.x - x) ** 2 + (synth_node.y - y) ** 2) < 2500:
                        others.append(synth_node.modelId)

                models = synthesizer.model_sampler.get_models(category, important, others)
                set_augmented_models = set(models)
                for modelId in models:
                    set_augmented_models |= synthesizer.get_relevant_models(category, modelId)
                set_augmented_models = list(set_augmented_models)

                tolerated = (dims_numpy[0]**2 + dims_numpy[1]**2) * 0.01
                scores = sorted(scores, key=lambda x:x[1])
                possible = [s for s in scores if s[1] < tolerated and s[0] in set_augmented_models]
                if len(possible) > 0:
                    best_modelId, _, best_dims = possible[0]
                else:
                    best_modelId, _, best_dims = scores[0]

            #Determine height, thanks to gravity this is simple
            if self.is_second_tier:
                z = self.current_room[3][math.floor(x)][math.floor(y)]
            else:
                z = floor_top
            new_node = SynthNode(best_modelId, category, x, y, z, sin, cos, self.synth_room, id, best_dims)

            overhang = False
            #Check for overhang
            if self.is_second_tier:
                render = new_node.get_render()
                render[render>0] = 1
                render2 = render.clone()
                render2[self.current_room[3] < z-0.03] = 0
                if render2.sum() < render.sum() * (0.9 - max(self.num_trials,0)*0.03):
                    #Setting overhang to True skips the remaining part and restarts the loop
                    #I am aware this is horrible programming practice, but well there was a deadline
                    overhang = True 

            #break in this ugly if else hodgepodge indicates that we have succeeded in everything
            #and we can jump out of the infinite loop and report success
            if not overhang:
                collisions = self.synth_room._get_collisions([new_node])
                if (len(collisions) - len(self.existing_collisions)) <= 0:
                    if not self.is_second_tier:
                        #done checking collisions, verify graph relations
                        failed_anchors, snapped = self.verify_relations(graph_node, new_node)
                        if not len(failed_anchors) > 0:
                            if snapped:
                                #when an object is snapped due to adjacent edges
                                #we need to recheck collisions
                                collisions = self.synth_room._get_collisions([new_node])
                                if (len(collisions) - len(self.existing_collisions)) <= 0:
                                    break
                            else:
                                break
                        else:
                            #record anchor objects that cause failure
                            for failed_anchor in failed_anchors:
                                if "synth" in failed_anchor:
                                    conflict_ids.add(failed_anchor)
                    else:
                        #No relation to enforce for second tier objects, free pass
                        break
                else:
                    if self.is_second_tier:
                        collisions_count = len(collisions)

                    #record objects that cause collision
                    for collision in collisions:
                        idA = collision.idA
                        idB = collision.idB
                        seen = False
                        for e_collision in self.existing_collisions:
                            if e_collision.idA == idA and e_collision.idB == idB:
                                seen = True
                        if not seen: #ignore existing collisions
                            idA = idA[2:] #Works
                            idB = idB[2:]
                            if idA == self.graph_node.id:
                                conflict = idB
                            elif idB == self.graph_node.id:
                                conflict = idA
                            else:
                                print(idA, idB, self.graph_node.id) #probably a bug if this gets printed
                                continue
                            if self.is_second_tier:
                                #Ignore minor collisions with the supporting object
                                not_a_problem = False
                                for edge in self.graph_node.in_edges:
                                    if edge.start_node.id == conflict:
                                        collisions_count -= 1
                                        not_a_problem = True
                                if not_a_problem:
                                    continue
                            if "synth" in conflict:
                                conflict_ids.add(conflict)

                    if self.is_second_tier:
                        #if the only collision is with the supporting object, then we are good
                        if collisions_count == len(self.existing_collisions):
                            break
                        #else:
                            #print("second tier collisions")
            self.num_trials += 1

            #Simple heuristics that allow more trials when the number of backtracks is lower
            if self.num_trials > (15-min(7,self.num_samples)):
                return False, list(conflict_ids)

        return True, new_node

    def _sample_location(self):
        #This can happen for second tiers, where other locations are literally zeroed out
        #Instead of having low probability since logits
        if self.location_map.sum() == 0:
            self.location_map = self._create_location_map()
            if self.location_map.sum() == 0:
                self.location_map[128][128] = 0.1 #Whatever
        
        loc = int(torch.distributions.Categorical(probs=self.location_map.view(-1)).sample())
        x,y = loc//256,loc%256
        
        r = (3 + 2*max(self.num_trials-5, 1)) * (self.num_samples+1) #more aggressive with range each try
        assert r >= 3
        self.location_map[x-r:x+r+1,y-r:y+r+1] = 0
        #Aggressively zero out regions where sampled
        return x+0.5,y+0.5

    def get_modelId_from_peers(self):
        """
        check if existing nodes already enforce a single possible modelId
        """
        graph_node = self.graph_node
        if graph_node.is_spoke:
            for edge in graph_node.in_edges:
                if edge.is_spoke:
                    hub = edge.start_node
            for edge in hub.out_edges:
                if edge.is_spoke:
                    if edge.end_node.category_name == graph_node.category_name and \
                                        edge.end_node.synth_node is not None:
                        return edge.end_node.synth_node.modelId, \
                               edge.end_node.synth_node.dims
        elif graph_node.is_chain:
            if graph_node.middle_of_chain:
                for edge in graph_node.chain_part_of: #Don't need to check last node in chain
                    if edge.start_node.category_name == graph_node.category_name and \
                                        edge.start_node.synth_node is not None:
                        if edge.start_node.sym_idx == graph_node.sym_idx:
                            return edge.start_node.synth_node.modelId, \
                                   edge.start_node.synth_node.dims
        return None, None

    @property
    def anchor_edges(self):
        """
        Get all relations to this node
        """
        #chains are hacky, Nth time
        if self.graph_node.is_chain and self.graph_node.middle_of_chain:
            chain_edges = [e for e in self.graph_node.in_edges if e.is_chain]

            chain_origin = self.graph_node.chain_part_of[0].start_node
            chain_end = self.graph_node.chain_part_of[-1].end_node

            #Ignore out of chain relations
            #that are not shared with the start/end of chain
            #this should have been done at the graph cleanup stage
            #but doing it here just to make sure
            for e in self.graph_node.in_edges:
                for e1 in (chain_origin.in_edges + chain_end.in_edges):
                    if e1.start_node == e.start_node:
                        chain_edges.append(e)

            #Out of chain relations shared by the start and end of the chain
            #are automatically added to all chain members (basically walls lol)
            for e1 in chain_origin.in_edges:
                for e2 in chain_end.in_edges:
                    if e1.start_node == e2.start_node and e1.edge_type == e2.edge_type and \
                        not [e1.start_node == e.start_node for e in chain_edges]:
                            chain_edges.append(e1) #Hack but works

            return chain_edges
        else:
            return self.graph_node.in_edges

    def _create_location_map(self):
        """
        Create a heatmap of possible locations
        """
        graph_node = self.graph_node
        synthesizer = self.synth_room.synthesizer
        num_categories = synthesizer.num_categories
        size = self.synth_room.size
        synth_room = self.synth_room

        room = self.current_room
        mask = torch.zeros_like(room[0:1])
        room = torch.cat([room, mask], 0)

        category = synthesizer.cat_to_index[graph_node.category_name]
        location_map = torch.ones((256,256))
        second_tier = False
        
        #In case of ambiguous fronts we do some extra rotation
        #check RelationshipGraph.Node.sym_idx for correspondance
        #numbers indicate the number of 90 degree rotation applied to the object
        #so [0,2] means the first front is 0 degree, the second is 180 degree
        #[] is for radial symmetry, we process it separately later
        fronts_dict = [[0], [], [0, 2], [0], [0, 2], [0,1,1,1], [0], [0]]
        
        for (i, edge) in enumerate(self.anchor_edges):
            edge_label = torch.LongTensor([edge.edge_type.index]).unsqueeze(0).cuda()
            anchor_id = edge.start_node.id
            anchor_loc = None
            #preprocess all info we needs to compute anchor-wise heatmaps
            if "wall" in edge.start_node.id:
                #Wall representation is slightly different
                segments = synth_room.scene.wall_segments
                anchor = [seg for seg in segments if seg["id"] == anchor_id]
                assert len(anchor) == 1
                anchor = anchor[0]
                mask = compute_object_mask(anchor)

                anchor_seg = anchor["points"]
                anchor_normal = anchor["normal"]
                # Compute anchor location
                anchor_loc = torch.FloatTensor(anchor_seg[0] + anchor_seg[1])/2
                anchor_loc = (anchor_loc/256 - 0.5) * 2
                # Compute anchor orientation
                anchor_orient = torch.FloatTensor(anchor_normal)
                fronts = [0] #Always single front for walls
                # Compute anchor mask'
            else:
                #print(anchor_id)
                synth_node = edge.start_node.synth_node
                if synth_node is None:
                    print("Warning: synth node is None")
                    continue
                anchor_loc = torch.Tensor([synth_node.x, synth_node.y])
                anchor_loc = (anchor_loc/256 - 0.5) * 2
                anchor_orient = torch.FloatTensor([-synth_node.sin, synth_node.cos])
                object_render = synth_node.get_render()
                mask = torch.zeros_like(object_render)
                mask[object_render>0] = 1
                fronts = fronts_dict[edge.start_node.sym_idx]
                if edge.start_node.is_chain and not (edge.start_node.middle_of_chain):
                    fronts = [0,1,1,1] #For start of chain we always check 4 fronts
                                      #Just to be a bit more lenient

            #predict heatmap conditioned on the anchor
            #anchor_orient is used to deal with multiple semantic fronts
            def get_heatmap(anchor_orient):
                room[-1] = mask
                inputs = inverse_xform_img2(room, anchor_loc, anchor_orient, 2)

                with torch.no_grad():
                    inputs_cuda = inputs.unsqueeze(0).cuda()
                    outputs = synthesizer.model_location(inputs_cuda, edge_label)
                    outputs = synthesizer.softmax(outputs)
                    outputs = F.interpolate(outputs, mode='bilinear', scale_factor=2).squeeze().cpu()

                    outputs = forward_xform_img(outputs, anchor_loc, anchor_orient, 2)[category+1]
                
                return outputs    

            outputs = torch.zeros_like(location_map)
            anchor_orients = []
            if len(fronts) == 0:    #Radial symmetry, pick 16 fronts, which provides a good approximation in practice
                for rot in range(16):
                    ang = math.pi/8*rot
                    anchor_orients.append(torch.FloatTensor([math.cos(ang), math.sin(ang)]))
            else:
                for front in fronts:
                    for _ in range(front):
                        anchor_orient = torch.Tensor([anchor_orient[1], -anchor_orient[0]]) #90 degree rotation
                    anchor_orients.append(anchor_orient)
            
            outputs = torch.zeros((256,256))
            #assemble heatmap for all orientations
            #Simply add them together if we did multiple anchors
            for anchor_orient in anchor_orients:
                outputs += get_heatmap(anchor_orient)

            outputs = outputs / outputs.sum()
            #multiply the anchor-conditioned heatmap with the overall heatmap
            location_map *= outputs
            
            #visualizing anchors
            #dest_name = synth_room.curr_top_down_view_filename()[:-4] + f"_{graph_node.category_name}_anchor{i}_{edge.start_node.category_name}_{edge.edge_type.base_name}_{edge.edge_type.dist}_{len(fronts)}fronts.png"
            #save_img(outputs * 100 + room[3]*0.5 + mask, dest_name)

            #for support edges, zero out everything that's not part of the support surface
            if edge.edge_type.base_name == "support":
                if "wall" in edge.start_node.id:
                    print("Warning: wall support edges")
                support_mask = torch.zeros_like(location_map)
                support_mask[object_render==room[3]] = 1
                location_map *= support_mask
                second_tier = True

        #zero out out of room and wall regions
        location_map[self.current_room[0] == 0] = 0
        location_map[self.current_room[1] > 0] = 0
        #normalize
        location_map = location_map / location_map.sum()
        dest_name = synth_room.curr_top_down_view_filename()[:-4] + f"_{graph_node.category_name}_anchor{i+1}_combined_heatmap.png"
        save_img(location_map * 100 + room[3], dest_name)
        self.is_second_tier = second_tier
        return location_map

    def verify_relations(self, graph_node, new_node):
        """
        Check if all relations specified by the graph is satisfied
        """
        synthesizer = self.synth_room.synthesizer
        num_categories = synthesizer.num_categories
        synth_room = self.synth_room

        size = synth_room.size
        room = self.current_room

        mask = torch.zeros_like(room[0])
        room = torch.stack([room[2], mask, mask], 0)
        category = synthesizer.cat_to_index[graph_node.category_name]
        #Number of rotations we need to perform to get all relations to the same coord frame (front)
        num_rotations = {"front":0, "right": 3, "back":2, "left": 1}
        #Following part similar to when we compute the heatmaps
        fronts_dict = [[0], [], [0, 2], [0], [0, 2], [0,1,1,1], [0], [0]]

        snapped = False
        failed_anchors = []
        total_error = 0
        for (i, edge) in enumerate(self.anchor_edges):
            room[1] = new_node.get_render()
            dist = edge.edge_type.dist
            edge_label = torch.LongTensor([edge.edge_type.index]).unsqueeze(0).cuda()
            anchor_id = edge.start_node.id
            if "wall" in anchor_id:
                segments = synth_room.scene.wall_segments
                anchor = [seg for seg in segments if seg["id"] == anchor_id]
                assert len(anchor) == 1
                anchor = anchor[0]
                mask = compute_object_mask(anchor)

                anchor_seg = anchor["points"]
                anchor_normal = anchor["normal"]
                # Compute anchor location
                anchor_loc = torch.Tensor(anchor_seg[0] + anchor_seg[1])/2
                anchor_loc = (anchor_loc/256 - 0.5) * 2
                # Compute anchor orientation
                anchor_orient = torch.Tensor([anchor_normal[1], -anchor_normal[0]])
                # Compute anchor mask'
                fronts = [0] #Always single front for walls
            else:
                synth_node = edge.start_node.synth_node
                if synth_node is None:
                    print("Warning: synth node is None")
                    continue
                anchor_loc = torch.Tensor([synth_node.x, synth_node.y])
                anchor_loc = (anchor_loc/256 - 0.5) * 2
                anchor_orient = torch.Tensor([synth_node.cos, synth_node.sin])
                for j in range(num_rotations[edge.edge_type.base_name]):
                    #print(j)
                    anchor_orient = torch.Tensor([anchor_orient[1], -anchor_orient[0]])
                object_render = synth_node.get_render()
                mask = torch.zeros_like(object_render)
                mask = object_render

                fronts = fronts_dict[edge.start_node.sym_idx]
                if edge.start_node.is_chain and not (edge.start_node.middle_of_chain):
                    fronts = [0,1,1,1] #For start of chain we always check 4 fronts
                                      #Just to be a bit more lenient
            room[-1] = mask

            #Check amount of error given a specified direction
            #A really hacky way to do raymarching in image space
            #I should have written a more rigorous relation check
            #But well deadline is a thing
            def get_error(anchor_orient):
                score = 1
                if dist == "adjacent": #Need more accuracy for adjacent
                    transformed = inverse_xform_img2(room, anchor_loc, anchor_orient, 1)
                else: #Need to be able to see everything for proximal and distant
                    transformed = inverse_xform_img2(room, anchor_loc, anchor_orient, 2)
                
                transformed[transformed>0.01] = 1 #In case noise

                _, anchor_xs = torch.max(transformed[2], 1)
                anchor_xs = anchor_xs.nonzero()

                _, anchor_ys = torch.max(transformed[2], 0)
                anchor_ys = anchor_ys.nonzero()

                transformed = transformed[:,anchor_xs,127:].squeeze(2) #Crop the negative part
                
                assert anchor_ys[-1] > 127 #Otherwise something must be disastrously wrong with centroids
                boundary = anchor_ys[-1] - 127
                #Boundary of anchor objects
                transformed[1,:,0:1] = 0 
                #Zero out first column to distinguish between "no hit"
                transformed[0,:,0:boundary+1] = 0 
                #Zero out existing objects within the boundary (To deal with "contains"
                _, target_xhits = torch.max(transformed[1], 1)
                _, existing_xhits = torch.max(transformed[0], 1)
                target_xhits[target_xhits==0] = 999 #Set no hit to "infinite"
                existing_xhits[existing_xhits==0] = 999

                visible_pixels = (target_xhits < existing_xhits).sum()
                total_pixels_estimate = new_node.img_space_shorter_dim_length
                if dist != "adjacent":
                    total_pixels_estimate /= 2
                
                if visible_pixels == 0:
                    return 1 #Always fail

                if float(visible_pixels) / total_pixels_estimate < 0.2: #Conservative estimate
                    score *= float(visible_pixels) / (total_pixels_estimate * 0.2)

                snap_distance = float(target_xhits.min() - boundary)

                #Score definition is pretty ad hoc
                if dist == "adjacent": 
                    if snap_distance > 10:
                        #Falls off until about 1.2m
                        score *= max(1 - ((snap_distance - 10)**1.2 * 0.01), 0)
                    elif snap_distance > 0:
                        #Snap along the normal
                        new_node.x -= float(anchor_orient[1]) * snap_distance
                        new_node.y -= float(anchor_orient[0]) * snap_distance
                        snapped = True
                elif dist == "proximal":
                    #Falls off until about 3m
                    if snap_distance > 15: #Approximately 0.7m, more lenient than that in the extraction process
                        score *= max(1 - ((snap_distance - 15)**1.2 * 0.01), 0)
                        
                elif dist == "distant":
                    if snap_distance < 8: #Approximately 0.38m, also more lenient
                        #We don't care as much for a distant relation being too close
                        score *= (1 - ((8-snap_distance) * 0.05))
                else:
                    raise NotImplementedError #WTF???
                return 1 - score 

            anchor_orients = []
            if len(fronts) == 0:    #Radial symmetry, pick 16 fronts
                for rot in range(16):
                    ang = math.pi/8*rot
                    anchor_orients.append(torch.FloatTensor([math.cos(ang), math.sin(ang)]))
            else:
                for front in fronts:
                    for _ in range(front):
                        anchor_orient = torch.Tensor([anchor_orient[1], -anchor_orient[0]])
                    anchor_orients.append(anchor_orient)
            
            #for anchor objects with multiple possible semantic fronts
            #we pick the interpretation that gives us the lowest errors
            errors = [get_error(anchor_orient) for anchor_orient in anchor_orients]
            min_error = min(errors)
            #If there is an error, record it
            if min_error != 0:
                failed_anchors.append(anchor_id)
            total_error += min_error

        #average the error over multiple anchor nodes
        total_error /= len(graph_node.in_edges)
        self.total_error = total_error
        #If error is lower than tolerance, success
        if total_error <= self.synth_room.error_tolerance:
            print(total_error, self.synth_room.error_tolerance)
            return [], snapped
        else:
            #other wise, record anchors that give the failure
            return failed_anchors, snapped


class SynthRoom():
    """
    Class that synthesize a single room and keeps its record
    """

    def __init__(self, room_id, trial, size, samples, synthesizer, temperature_cat,
                 temperature_pixel, min_p, max_collision, seed, save_dir, save_visualizations,
                 save_initial_state=False):
        """
        Refer to SceneSynth.synth for explanations for most parameters
        """
        self.__dict__.update(locals())
        del self.self #Of course I don't care about readability

        self.scene = RenderedScene(index = self.room_id, \
                                    data_dir = synthesizer.data_dir_relative, \
                                    data_root_dir = synthesizer.data_root_dir, \
                                    load_objects = False)

        self.door_window_nodes = self.scene.door_window_nodes
        self.empty_house_json = None
        self.failures = 0
        self.seed = seed
        self.save_dir = save_dir
        self.save_visualizations = save_visualizations

    def sample_graph(self):
        """
        Sample a relationship graph
        see model.graph
        """
        graph = RelationshipGraph()
        graph.load_from_file(f"{self.synthesizer.data_dir}/graph/{self.room_id}.pkl")
        _, graph = self.synthesizer.model_graph.sample(graph, return_relation_graph=True, keep_edge_type=keep_edge_type)
        return graph

    def synthesize(self, num_steps=100000000):
        #ignore num_steps, not used anymore basically
        self.steps = 0
        save_new = True #For now
        print("New Room")

        while True:
            graph = self.sample_graph()
            while graph.has_cycle() or (not graph.is_connected()[0]):
                if graph.has_cycle():
                    print("Cycles detected, resample graph...")
                else:
                    print("Graph not connected, resample graph...")
                graph = self.sample_graph()
            
            graph.save_to_file(f"{save_dir}/{self.room_id}_{self.trial}.pkl")

            #Cleanup graphs, make hub-spoke and chains consistent
            graph.make_hubs(stats)
            if graph.clean_up_chains() is not False:
                nodes_count = 0
                for node in graph.nodes:
                    if node.category_name in self.synthesizer.categories:
                        nodes_count += 1

                #constraint based ordering
                nodes_prechain = graph.get_nodes_in_instantiation_order(data_dir)
                nodes = []

                #add back the middle chain nodes
                for node in nodes_prechain:
                    node.middle_of_chain = False
                    node.synth_node = None
                    nodes.append(node)
                    if node.is_chain:
                        for chain in node.chains_starting_from:
                            for edge in chain:
                                edge.end_node.middle_of_chain = True
                                edge.end_node.synth_node = None
                                edge.end_node.chain_part_of = chain
                                nodes.append(edge.end_node)
                
                if nodes_count != len(nodes): #Corner cases that probably still aren't fixed
                    pass
                else:
                    break
        
        self.graph = graph
        
        #Stack of current successful steps
        self.synth_steps = []
        composite = self.scene.create_composite()
        self.composite = copy.deepcopy(composite)
        self.current_room = self.composite.get_composite(num_extra_channels=0)
        if save_new:
            self.save_top_down_view()
            self.save_json()

        node = nodes[0]
        #next step to execute, since the room is empty, it would be the step containing the first node to insert       
        next_step = SynthStep(self, composite, node)
        print("================================================")
        self.max_allowed_backtracks = 25 #Pretty random choice
        self.backtracks = 0

        #main loop, tries and backtrakcs until scene is complete or max allowed backtracks is reached
        while len(self.synth_steps) < len(nodes):
            if self.backtracks >= self.max_allowed_backtracks:
                print("Too much backtracking, wtf, give up already.") #Keeping my frustration print statement OwO
                return False
            print(f'*** Synth step {len(self.synth_steps)}')
            
            #who doesn't love python typing
            success, composite_or_conflicts = next_step.sample_next()
            
            #If the insertion succeeded:
            if success:
                #add the succeded step to the stack
                self.synth_steps.append(next_step)
                composite = composite_or_conflicts

                #update room composite
                self.current_room = composite.get_composite(num_extra_channels=0)
                if save_new:
                    self.save_top_down_view()
                    self.save_json()
                
                #if not done, add a new step, init with the updated composite
                if len(self.synth_steps) < len(nodes):
                    node = nodes[len(self.synth_steps)]
                    next_step = SynthStep(self, composite, node) #set next step to be the newly added one
            #If unsuccessful
            else:
                print("Failed to find a reasonable location, backtracking")
                #get the list of nodes that caused a conflict (collision or graph wise) during the failed attempt
                conflict_ids = composite_or_conflicts

                #Handle corner cases where the step we executed was the first step (i.e. stack of successfuls steps empty)
                if len(self.synth_steps) == 0:
                    next_step = SynthStep(self, copy.deepcopy(self.composite), nodes[0])
                    #If the first step is unsuccessful, there must be something really bad happening early
                    #so we fast forward so we can start ignore relationships
                    #again a hack, I forgot if this is necessary, so I keep it here
                    self.backtracks = max(self.backtracks, int(self.max_allowed_backtracks/2))
                else:
                    #if not, initialize the next step with the last step on the stack
                    #this might be changed by the following code
                    next_step = self.synth_steps.pop()

                #If the insertion is actually performed, there must be some conflicts
                if conflict_ids is not None:
                    #print(conflict_ids)
                    self.backtracks += 1
                    #Pop the stack until we find the first step that causes a conflict
                    #Done this way because it is futile to resample things that didn't really cause trouble with insertion
                    if len(conflict_ids) > 0:
                        while not (next_step.contains_node_ids(conflict_ids)):
                            next_step = self.synth_steps.pop()
                    assert(len(self.synth_steps) >= 0)  #Must be the case due to how conflict_ids are computed
                else:
                    #Otherwise, no insertion happened
                    #the attempt failed because the maximum number of tries is reached for a step
                    self.steps -=1 #Not actually a step, just need to backtrack further
            self.steps += 1

        #done yay
        self.current_room = composite.get_composite(num_extra_channels=0)
        self.save_top_down_view(final=True)
        self.save_json(final=True)

        return True

    @property
    def error_tolerance(self):
        """
        error tolerance, function of number of allowed backtracks
        starts flat, then increases polynomially
        """
        #name of function use to be san check
        if self.backtracks < self.max_allowed_backtracks/4:
            return 0 #We want ot be nice towards the start
        else:
            #No CoC reference, I promise
            max_san = self.max_allowed_backtracks * 3 / 4
            sanity_level = self.backtracks - self.max_allowed_backtracks / 4
            return (sanity_level/max_san) ** 1.5 #used to be 2.114(redacted)
    
    def _get_existing_categories(self):
        #Category count to be used by networks
        existing_categories = torch.zeros(self.synthesizer.num_categories)
        for node in self.object_nodes:
            existing_categories[node.category] += 1
        return existing_categories

    def _get_category_name(self, index):
        #Return name of the category, given index
        return self.synthesizer.categories[index]

    @property
    def synth_nodes(self):
        return [step.graph_node.synth_node for step in self.synth_steps]

    @property
    def object_nodes(self):
        return self.synth_nodes #Not confusing at all

    def _get_collisions(self, additional_nodes=None):
        with stdout_redirected():
            oc = self.synthesizer.object_collection
            oc.reset()
            oc.init_from_house(House(house_json=self.get_json(additional_nodes)))
            contacts = oc.get_collisions(include_collision_with_static=True)
            collisions = []

        for (_, contact_record) in contacts.items():
            #print(collision_pair)
            if contact_record.idA != contact_record.idB:
                #If contact with the room geometry, be more lenient and allow anything with 
                #less than 0.25m overlap
                if "0_0" in contact_record.idA or "0_0" in contact_record.idB:
                    if contact_record.distance < -0.01:
                        collisions.append(contact_record)
                else:
                    #Else, check if collision amount is more than max_collision, if, then it is a collision
                    if contact_record.distance < -0.01:
                        collisions.append(contact_record)
                    elif contact_record.distance < -0.005:
                        if abs(contact_record.contactNormalOnBInWS[1]) > 0.1:
                            collisions.append(contact_record)

        return collisions

    def save_top_down_view(self, final=False):
        """
        Save the top down view of the current room
        """
        img = m.toimage(self.current_room[3].numpy(), cmin=0, cmax=1)
        img.save(self.curr_top_down_view_filename(final))
    
    def _create_empty_house_json(self):
        #Redacted
        raise NotImplementedError

    def save_json(self, final=False):
        """
        Save the json file, see save_top_down_view
        """
        house = self.get_json()
        with open(self.curr_json_filename(final), 'w') as f:
            json.dump(house, f)

    def get_json(self, additional_nodes=None):
        """
        Get the json of the current room, plus additional_nodes
        """
        #Redacted
        raise NotImplementedError

    @property
    def total_error(self):
        return sum([s.total_error for s in self.synth_steps])

    def curr_top_down_view_filename(self, final=False):
        if final:
            return f"{self.save_dir}/{self.room_id}_{self.trial}_{self.steps}_{len(self.object_nodes)}_final_{self.backtracks}_{self.total_error:.5f}.png"
        else:
            return f"{self.save_dir}/{self.room_id}_{self.trial}_{self.steps}_{len(self.object_nodes)}.png"

    def curr_json_filename(self, final=False):
        if final:
            return f"{self.save_dir}/{self.room_id}_{self.trial}_{self.steps}_{len(self.object_nodes)}_final_{self.backtracks}_{self.total_error:.5f}.json"
        else:
            return f"{self.save_dir}/{self.room_id}_{self.trial}_{self.steps}_{len(self.object_nodes)}.json"

class SynthNode():
    """
    Representing a node in synthesis time
    """
    def __init__(self, modelId, category, x, y, z, sin, cos, room, id=None, dims=None):
        self.__dict__.update(locals())
        del self.self
        self.render = None
    
    def get_render(self):
        """
        Get the top-down render of the object
        """
        o = Obj(self.modelId)
        o.transform(self.get_transformation())
        render = torch.from_numpy(TopDownView.render_object_full_size(o, self.room.size))
        self.render = render
        return render

    @property
    def img_space_diag_length(self, img_size=256):
        assert self.dims is not None
        return (self.dims[0]**2 + self.dims[1]**2)**0.5*img_size

    @property
    def img_space_shorter_dim_length(self, img_size=256):
        assert self.dims is not None
        return (min(self.dims[0], self.dims[1])) * img_size

    def get_transformation(self):
        """
        Get the transformation matrix
        Used to render the object
        and to save in json files
        """
        x,y,z = self.x, self.y, self.z
        xscale = self.room.synthesizer.pgen.xscale
        yscale = self.room.synthesizer.pgen.yscale
        zscale = self.room.synthesizer.pgen.zscale
        zpad = self.room.synthesizer.pgen.zpad

        sin, cos = self.sin, self.cos

        t = np.asarray([[cos, 0, -sin, 0], \
                        [0, 1, 0, 0], \
                        [sin, 0, cos, 0], \
                        [0, 0, 0, 1]])
        t_scale = np.asarray([[xscale, 0, 0, 0], \
                              [0, zscale, 0, 0], \
                              [0, 0, xscale, 0], \
                              [0, 0, 0, 1]])
        t_shift = np.asarray([[1, 0, 0, 0], \
                              [0, 1, 0, 0], \
                              [0, 0, 1, 0], \
                              [x, z, y, 1]])
        
        return np.dot(np.dot(t,t_scale), t_shift)

def compute_object_mask(node, img_size=256):
    h = node["height_map"]
    xsize, ysize = h.size()
    # Clip the mask if bounding box is out of boundary
    xmin = math.floor(node["bbox_min"][0])
    ymin = math.floor(node["bbox_min"][2])
    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0
    assert (xmin >= 0 and ymin >= 0), f"{node['category']}"
    height = torch.zeros((img_size, img_size))
    height[xmin:xmin+xsize,ymin:ymin+ysize] = h
    mask = torch.zeros_like(height)
    mask[height>0] = 1.0
    return mask

#some bad abstractions that's too late to fix
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

def show_img(tensor):
    img = tensor
    img = m.toimage(img, cmin=0, cmax=1)
    img.show()

def save_img(tensor, dest):
    img = tensor
    img = m.toimage(img, cmin=0, cmax=1)
    img.save(dest)

#I forgot why two versions but whatever, no time to check...
def inverse_xform_img2(img, loc, orient, scale=1):
    matrices = torch.zeros(2, 3)
    cos, sin = orient
    matrices[0, 0] = cos
    matrices[1, 1] = cos
    matrices[0, 1] = -sin
    matrices[1, 0] = sin
    matrices[0, 2] = loc[1]
    matrices[1, 2] = loc[0]
    scale_matrices = torch.eye(3, 3)
    scale_matrices[0, 0] = scale
    scale_matrices[1, 1] = scale
    matrices = torch.matmul(matrices, scale_matrices).unsqueeze(0)
    img = img.unsqueeze(0).float()
    out_size = torch.Size(img.shape)
    grid = F.affine_grid(matrices, out_size)
    return F.grid_sample(img, grid).squeeze(0).squeeze(0)

def forward_xform_img(img, loc, orient, scale=1):
    # First, build the inverse rotation matrices
    inv_rot_matrices = torch.zeros(3, 3)
    cos = orient[0]
    sin = orient[1]
    inv_rot_matrices[0, 0] = cos
    inv_rot_matrices[1, 1] = cos
    inv_rot_matrices[0, 1] = sin
    inv_rot_matrices[1, 0] = -sin
    inv_rot_matrices[2, 2] = 1
    # Then, build the inverse translation matrices
    # (Apparently, x and y need to be swapped. I don't know why...)
    inv_trans_matrices = torch.eye(3, 3)
    inv_trans_matrices[0, 2] = -loc[1]
    inv_trans_matrices[1, 2] = -loc[0]
    # Build scaling transform matrices
    scale_matrices = torch.eye(3, 3)
    scale_matrices[0, 0] = 1/scale
    scale_matrices[1, 1] = 1/scale
    # Multiply them to get the full affine matrix
    inv_matrices = torch.matmul(scale_matrices, inv_rot_matrices)
    inv_matrices = torch.matmul(inv_matrices, inv_trans_matrices)
    # Discard the last row (affine_grid expects 2x3 matrices)
    inv_matrices = inv_matrices[0:2, :].unsqueeze(0)
    # Finalize
    img = img.unsqueeze(0).float()
    out_size = torch.Size(img.shape)
    grid = F.affine_grid(inv_matrices, out_size)
    return F.grid_sample(img, grid).squeeze(0).squeeze(0)

if __name__ == '__main__':
    utils.ensuredir(save_dir)

    run_full_synth()
