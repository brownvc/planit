import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from contextlib import contextmanager
from torch.distributions import *
from data.graph import RelationshipGraph
from data.object_data import ObjectCategories
import random

"""
Pretty inefficient implementation of MPNN
Runs faster on CPU...
"""

#There are lot of leftover ablation codes
#I should really remove them
#but this is nontrivial effort...
#just refer to graph_train.py for actual setting for final experiments

EdgeType = RelationshipGraph.EdgeType

class GroundTruthNode():
    """
    Minimalistic data class that only holds the part of info necessary for the model
    As opposed to the more verbose strucutre of data.graph
    """
    def __init__(self, category, adjacent, length=0, sym_idx=0, is_hub=False, is_spoke=False, is_chain=False, id=None):
        self.category = category
        self.adjacent = adjacent
        self.length = length
        self.sym_idx = sym_idx
        self.is_hub = is_hub
        self.is_spoke = is_spoke
        self.is_chain = is_chain
        self.id = id
    
#Different linear layers for ablation
class Linear1(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=None, dropout=None):
        super(Linear1, self).__init__()
        if dropout is not None:
            self.model = nn.Sequential(
                            nn.Dropout(p=GraphNetConfig.dropout_p),
                            nn.Linear(in_size, out_size),
                         )
        else:
            self.model = nn.Linear(in_size, out_size)

    def forward(self, x):
        return self.model(x)

class Linear2(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=None, dropout=None):
        super(Linear2, self).__init__()
        if GraphNetConfig.hidden_size is None:
            hidden_size = max(in_size, out_size)
        else:
            hidden_size = GraphNetConfig.hidden_size
        if dropout is not None:
            self.model = nn.Sequential(
                            nn.Linear(in_size, hidden_size),
                            nn.LeakyReLU(),
                            nn.Dropout(p=GraphNetConfig.dropout_p),
                            nn.Linear(hidden_size, out_size),
                         )
        else:
            self.model = nn.Sequential(
                            nn.Linear(in_size, hidden_size),
                            nn.LeakyReLU(),
                            nn.Linear(hidden_size, out_size),
                         )

    def forward(self, x):
        return self.model(x)

class Linear3(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=None, dropout=None):
        super(Linear3, self).__init__()
        if GraphNetConfig.hidden_size is None:
            hidden_size = max(in_size, out_size)
        else:
            hidden_size = GraphNetConfig.hidden_size
            #print(hidden_size, in_size, out_size)
        #else:
            #print("??????????????????")
        if dropout is not None:
            self.model = nn.Sequential(
                            nn.Linear(in_size, hidden_size),
                            nn.LeakyReLU(),
                            nn.Linear(hidden_size, hidden_size),
                            nn.LeakyReLU(),
                            nn.Dropout(p=GraphNetConfig.dropout_p),
                            nn.Linear(hidden_size, out_size),
                         )
        else:
            self.model = nn.Sequential(
                            nn.Linear(in_size, hidden_size),
                            nn.LeakyReLU(),
                            nn.Linear(hidden_size, hidden_size),
                            nn.LeakyReLU(),
                            nn.Linear(hidden_size, out_size),
                         )

    def forward(self, x):
        return self.model(x)

layer_name_to_class = {
    "1": Linear1,
    "2": Linear2,
    "3": Linear3,
}

class GraphNetConfig():
    """
    Holding all configs as cls variables
    Doesn't really work in many cases
    Should be easy to restructure with json stuff but I guess it doesn't really matter
    """
    node_size = 64 #Size of node embedding
    num_edge_types = RelationshipGraph.EdgeType.NumTypes #number of edge types
    hG_multiplier = 2 #Size of graph embedding / size of node embedding

    shuffle_nodes = True #shuffle nodes when training
    shuffle_edges = True #shuffle edges when training
    include_walls = True #include wall nodes or not, probably won't work if turned off as too many stuff have changed
    
    propagation_layer = Linear1 #single layer doesn't really work, final version uses Linear3
    aggregation_layer = Linear1
    initializing_layer = Linear1
    cuda = False #GPU training is slower

    hidden_size = None #Size of hidden layers of linear modules, none means pick the layer of in_size & out_size

    decision_module_names = ["an", "ae", "node", "et"]
    #an: AddNode, ae: AddEdge, node: choose which node, et: determining edge type

    temp_dict = {}
    temp = 1 #amount of tempering for each module
    
    num_sym_classes = 8
    predict_sym_type = True

    decision_layer = Linear1
    decision_layer_dict = {}
    autoregressive = False #if true, resets node embedding after each decision
    restore_target = {}  #which version of node embedding to restore to
    #restore_target = {"ae": "an", "node": "an", "et": None}
    rounds_of_propagation = 3 #T
    rounds_of_propagation_dict = {} #This can be different for each module

    init_with_graph_representation = True #If true, graph embedding is used to initialize new nodes
    include_one_hot = False #If true, include one hot category encodings for message passing
    choose_node_graph_vector = False  #If true, include graph embedding when choosing node to connect edge to
    
    predict_edge_type_first = False 
    #If true, AddEdge predicts the edge type instead of just boolean indicating add or not

    separate_wall_edges_step = True
    #If true, Use different sets of weights for wall edges and handle them separately
    
    node_and_type_together = False #If true, predict which node to connect to and which edge to connect to together
    everything_together = False 
    #Implies the above one, if true, skip add edge step and use a fixed logit to represent "stop adding edge" instead
    no_edge_logit = math.log(10) #Used if everything_together, choose this value so logit of 0 is roughly prob of 0.1

    auxiliary_choose_node = False #Forgot what this option is
    per_node_prediction = False #Instead of softmaxing over all nodes, predciting node wise if there is an edge
    element_wise_loss_node = False #element wise BCE instead of cross entropy

    dropout_p = None

    @classmethod
    def compute_derived_attributes(cls):
        """
        Compute a bunch of attributes that is derived from everything else
        Could just use a lot of @property instead
        """
        cls.categories += ["Stop"] #Add "stop" category

        if cls.everything_together:
            cls.node_and_type_together = True #everything_together implies this

        if cls.predict_sym_type:
            cls.decision_module_names += ["sym"] #if predict symmetry type, also adding this as a module

        if cls.include_one_hot:
            cls.node_size = cls.node_size + cls.num_cat #if one hot is included, size of node embedding is larger
            if cls.predict_sym_type:
                cls.node_size += cls.num_sym_classes + 3 #If sym class is included, also need to include those as node embedding

        cls.hG_size = cls.node_size * cls.hG_multiplier #Size of graph embedding
        cls.edge_size = cls.num_edge_types + 1 #Add size of 1 to accomodate wall angles

        if cls.per_node_prediction: #Special case, special treatment. Well this is not used at all now
            cls.decision_module_names = ["an", "et"]
            cls.separate_wall_edges_step = False
        
        #Change layer type from strings to actual classes
        if isinstance(cls.propagation_layer, str):
            cls.propagation_layer = layer_name_to_class[cls.propagation_layer]
        if isinstance(cls.initializing_layer, str):
            cls.initializing_layer = layer_name_to_class[cls.initializing_layer]

        if isinstance(cls.aggregation_layer, str):
            cls.aggregation_layer = layer_name_to_class[cls.aggregation_layer]
        if isinstance(cls.decision_layer, str):
            cls.decision_layer = layer_name_to_class[cls.decision_layer]
        
        #Also add a restore point denoting the initial state of nodes
        if cls.autoregressive:
            cls.restore_target["start"] = "initial" 
        else:
            cls.restore_target["start"] = None

        for name in cls.decision_module_names:
            if not name in cls.temp_dict:
                cls.temp_dict[name] = cls.temp #set default temperature
            if not name in cls.rounds_of_propagation_dict:
                cls.rounds_of_propagation_dict[name] = cls.rounds_of_propagation #set deault T
            if not name in cls.decision_layer_dict:
                cls.decision_layer_dict[name] = cls.decision_layer #set default decision layer type
            elif isinstance(cls.decision_layer_dict[name], str):
                cls.decision_layer_dict[name] = layer_name_to_class[cls.decision_layer_dict[name]] #also update string to class names

            if not name in cls.restore_target:
                if cls.autoregressive:
                    cls.restore_target[name] = "initial" #set default restore target in the autoregressive case
                else:
                    cls.restore_target[name] = None
        
        backup_points = list(set(cls.restore_target.values()))
        cls.backup_needed = {} #If the state of the node embedding before a certain module is a restore target, back it up
        for name in cls.decision_module_names:
            if name in backup_points:
                cls.backup_needed[name] = True
            else:
                cls.backup_needed[name] = False

class Node():
    def __init__(self, category, sym_idx=0, is_hub=False, is_spoke=False, is_chain=False, is_arch=False):
        self.category = category
        self.is_arch = is_arch
        self.sym_idx = sym_idx
        self.is_hub = is_hub
        self.is_spoke = is_spoke
        self.is_chain = is_chain
        self.incoming = [] #Node is the end point of the edge
        self.outgoing = [] #Node is the starting point

    @property
    def category_verbose(self): #actual name instead of idx, used for debugging and logging
        if self.is_arch:
            return GraphNetConfig.arch_categories[self.category]
        else:
            return GraphNetConfig.categories[self.category]

class Propagator(nn.Module):
    """
    Propagate information across nodes
    """
    def __init__(self, rounds_of_propagation):
        super(Propagator, self).__init__()
        self.rounds_of_propagation = rounds_of_propagation

        node_size = GraphNetConfig.node_size
        edge_size = GraphNetConfig.edge_size
        message_size = node_size * 2 + edge_size

        # Models for propagating the state vector
        #Gathering message from an adjacent node
        #Separate weights for rounds_of_propagation rounds

        #Forward direction
        self.f_ef = nn.ModuleList([
            GraphNetConfig.propagation_layer(message_size, node_size * 2)
        for i in range(rounds_of_propagation)])
        
        #Reverse direction
        self.f_er = nn.ModuleList([
            GraphNetConfig.propagation_layer(message_size, node_size * 2)
        for i in range(rounds_of_propagation)])
        
        if GraphNetConfig.include_one_hot: #One hot part of node embedding is constant, so it need to be subtracted from output size
            output_size = node_size - GraphNetConfig.num_cat
            if GraphNetConfig.predict_sym_type:
                output_size -= GraphNetConfig.num_sym_classes + 3
        else:
            output_size = node_size

        #Mapping aggregated message vector to new node state vector
        self.f_n = nn.ModuleList([
            nn.GRUCell(node_size * 2, output_size)
        for i in range(rounds_of_propagation)])

    def forward(self, gn):
        if len(gn.nodes) == 0 or len(gn.edges) == 0:
            return
        for i in range(self.rounds_of_propagation):
            #compute messages
            messages_raw = torch.cat((gn.node_vectors[gn.u_indices],
                               gn.node_vectors[gn.v_indices],
                               gn.edge_vectors), 1)
            messages_forward = self.f_ef[i](messages_raw)
            messages_reverse = self.f_er[i](messages_raw)

            #aggregate messages           
            aggregated = None
            for node in gn.nodes: #this should be ideally done with scatter...
                if GraphNetConfig.cuda:
                    a_v = torch.zeros(1, GraphNetConfig.node_size*2).cuda()
                else:
                    a_v = torch.zeros(1, GraphNetConfig.node_size*2)
                if len(node.incoming) > 0:
                    a_v += messages_forward[node.incoming].sum(dim=0)
                if len(node.outgoing) > 0:
                    a_v += messages_reverse[node.outgoing].sum(dim=0)
                if aggregated is None:
                    aggregated = a_v
                else:
                    aggregated = torch.cat((aggregated, a_v), 0)
            
            #update node embedding
            gn.node_vectors = self.f_n[i](aggregated, gn._node_vectors)

class Aggregator(nn.Module):
    """
    Aggregates information across nodes to create a graph vector
    """
    def __init__(self):
        super(Aggregator, self).__init__()
        node_size = GraphNetConfig.node_size
        hG_size = GraphNetConfig.hG_size

        # Model for computing graph representation
        self.f_m = GraphNetConfig.aggregation_layer(node_size, hG_size)
        #Gated parameter when aggregating for graph representation
        self.g_m = nn.Sequential(
            GraphNetConfig.aggregation_layer(node_size, hG_size),
            nn.Sigmoid()
        )

    def forward(self, gn):
        #Default for empty graph
        if len(gn.nodes) == 0:
            if GraphNetConfig.cuda:
                h_G = torch.zeros(1, GraphNetConfig.hG_size).cuda()
            else:
                h_G = torch.zeros(1, GraphNetConfig.hG_size)
        else:
            h_G = (self.f_m(gn.node_vectors) * self.g_m(gn.node_vectors)).sum(dim=0).unsqueeze(0)

        return h_G

class AddNode(nn.Module):
    """
    Decide whether to add, and what type of nodes
    """
    def __init__(self):
        super(AddNode, self).__init__()
        num_cat = GraphNetConfig.num_cat
        hG_size = GraphNetConfig.hG_size
        
        if GraphNetConfig.rounds_of_propagation_dict["an"] > 0:
            self.prop = Propagator(GraphNetConfig.rounds_of_propagation_dict["an"])
        self.aggre = Aggregator()
            
        self.f_add_node = GraphNetConfig.decision_layer_dict["an"](hG_size, num_cat+1, dropout=GraphNetConfig.dropout_p)
    
    def forward(self, gn):
        if GraphNetConfig.rounds_of_propagation_dict["an"] > 0:
            self.prop(gn)
        return self.f_add_node(self.aggre(gn))

class SymmetryType(nn.Module):
    """
    Decide symmetrytype of the node
    """
    def __init__(self):
        super(SymmetryType, self).__init__()
        hG_size = GraphNetConfig.hG_size

        self.aggre = Aggregator()
        num_sym_classes = GraphNetConfig.num_sym_classes

        #cartesian product of sym_class * superstructure type
        self.f_sym_type = GraphNetConfig.decision_layer_dict["sym"](hG_size, num_sym_classes*5, dropout=GraphNetConfig.dropout_p)
    
    def forward(self, gn):
        aggre = self.aggre(gn)
        return self.f_sym_type(aggre)

class AddEdge(nn.Module):
    """
    Decide whether to add an edge or not
    """
    def __init__(self):
        super(AddEdge, self).__init__()
        node_size = GraphNetConfig.node_size
        hG_size = GraphNetConfig.hG_size

        if GraphNetConfig.rounds_of_propagation_dict["ae"] > 0:
            self.prop = Propagator(GraphNetConfig.rounds_of_propagation_dict["ae"])
        self.aggre = Aggregator()
        
        if GraphNetConfig.predict_edge_type_first:
            out_size = GraphNetConfig.num_edge_types + 1
        else:
            out_size = 1

        self.f_add_edge = nn.Sequential(
            GraphNetConfig.decision_layer_dict["ae"](hG_size + node_size, out_size, dropout=GraphNetConfig.dropout_p)
        )

    def forward(self, gn, target_idx):
        if GraphNetConfig.rounds_of_propagation_dict["ae"] > 0:
            self.prop(gn)
        return self.f_add_edge(torch.cat((self.aggre(gn), gn.node_vectors[[target_idx]]), 1))

class ChooseNode(nn.Module):
    """
    Decide which node to add an edge to
    """
    def __init__(self):
        super(ChooseNode, self).__init__()
        node_size = GraphNetConfig.node_size
        
        if GraphNetConfig.choose_node_graph_vector: #if uses graph embedding, need to include this to compute graph embedding
            self.aggre = Aggregator()

        if GraphNetConfig.rounds_of_propagation_dict["node"] > 0:
            self.prop = Propagator(GraphNetConfig.rounds_of_propagation_dict["node"])
        
        if GraphNetConfig.choose_node_graph_vector:
            in_size = node_size * 2 + GraphNetConfig.hG_size
        else:
            in_size = node_size * 2
        
        out_size = 2
        if GraphNetConfig.node_and_type_together:
            out_size = out_size * GraphNetConfig.num_edge_types
        
        self.f_s = GraphNetConfig.decision_layer_dict["node"](in_size, out_size, dropout=GraphNetConfig.dropout_p)
    
    def forward(self, gn, target_idx, wall_mode=-1):
        if GraphNetConfig.rounds_of_propagation_dict["node"] > 0:
            self.prop(gn)

        #compute the logits for all pairs
        if wall_mode == 1:
            concat = torch.cat((gn.node_vectors[:gn.num_walls], gn.node_vectors[target_idx].repeat(gn.num_walls,1)), 1)
        elif wall_mode == 0:
            if len(gn.nodes)-gn.num_walls-1 == 0:
                return None #Deal with everything together sampling phase
            concat = torch.cat((gn.node_vectors[gn.num_walls:-1], gn.node_vectors[target_idx].repeat(len(gn.nodes)-gn.num_walls-1,1)), 1)
        else:
            concat = torch.cat((gn.node_vectors[:-1], gn.node_vectors[target_idx].repeat(len(gn.nodes)-1,1)), 1)

        if GraphNetConfig.choose_node_graph_vector:
            h_G = self.aggre(gn)
            concat = torch.cat((concat, h_G.repeat(concat.size()[0],1)), 1)

        return self.f_s(concat).view(1,-1)

class ChooseEdgeType(nn.Module):
    """
    Decide which type the edge is
    """
    #not used in final ver
    def __init__(self):
        super(ChooseEdgeType, self).__init__()
        
        node_size = GraphNetConfig.node_size
        num_edge_types = GraphNetConfig.num_edge_types
        in_size = node_size * 2
        if GraphNetConfig.per_node_prediction:
            num_edge_types += 1 #Stop
            
            #I think otherwise edge type is pretty local
            if GraphNetConfig.choose_node_graph_vector:
                self.aggre = Aggregator()
                in_size += GraphNetConfig.hG_size

        if GraphNetConfig.rounds_of_propagation_dict["et"] > 0:
            self.prop = Propagator(GraphNetConfig.rounds_of_propagation_dict["et"])
        
        #minus additional edge vectors
        self.h_G = None

        self.f_s = GraphNetConfig.decision_layer_dict["et"](in_size, num_edge_types, dropout=GraphNetConfig.dropout_p)
    
    def forward(self, gn, source_idx, target_idx, repropagate=True):
        if GraphNetConfig.rounds_of_propagation_dict["et"] > 0 and repropagate:
            self.prop(gn)
        concat = torch.cat((gn.node_vectors[[source_idx]], gn.node_vectors[[target_idx]]) ,1)

        if GraphNetConfig.per_node_prediction and GraphNetConfig.choose_node_graph_vector:
            if repropagate:
                self.h_G = self.aggre(gn)
            concat = torch.cat((concat, self.h_G), 1)
        return self.f_s(concat)


class Initializer(nn.Module):
    """
    init node embedding
    """
    def __init__(self):
        super(Initializer, self).__init__()
        
        node_size = GraphNetConfig.node_size
        num_cat = GraphNetConfig.num_cat
        if GraphNetConfig.include_one_hot:
            node_size -= num_cat
            if GraphNetConfig.predict_sym_type:
                node_size -= GraphNetConfig.num_sym_classes + 3

        # Models for initializing
        #For initializing state vector of a new node
        feature_size = num_cat
        if GraphNetConfig.predict_sym_type:
            feature_size += GraphNetConfig.num_sym_classes + 3

        if GraphNetConfig.init_with_graph_representation:
            hG_size = GraphNetConfig.hG_size
            self.f_init = GraphNetConfig.initializing_layer(feature_size + hG_size, node_size)
            self.aggre = Aggregator()
        else:
            self.f_init = GraphNetConfig.initializing_layer(feature_size, node_size)
        
        self.tanh = nn.Tanh()
    
    def forward(self, gn, e):
        #One hot for now
        if GraphNetConfig.init_with_graph_representation:
            h_G = self.aggre(gn)
            h_v = self.f_init(torch.cat((e, h_G), 1))
        else:
            h_v = self.f_init(e)
        return self.tanh(h_v)

class WallInitializer(nn.Module):
    """
    init wall node embedding
    """
    def __init__(self):
        super(WallInitializer, self).__init__()
        node_size = GraphNetConfig.node_size
        if GraphNetConfig.include_one_hot:
            node_size -= GraphNetConfig.num_cat
            if GraphNetConfig.predict_sym_type:
                node_size -= GraphNetConfig.num_sym_classes + 3

        in_size = GraphNetConfig.num_arch_cat + 1
        self.f_init = GraphNetConfig.initializing_layer(in_size, node_size)
        self.tanh = nn.Tanh()

    def forward(self, e):
        h_v = self.f_init(e)
        return self.tanh(h_v)

class GraphNet(nn.Module):
    """
    Main class
    """

    def __init__(self):
        super(GraphNet, self).__init__()
        
        self.f_add_node = AddNode()
        if GraphNetConfig.predict_sym_type:
            self.f_sym_type = nn.ModuleList([SymmetryType() for i in range(GraphNetConfig.num_cat)])

        if not GraphNetConfig.per_node_prediction:
            if not GraphNetConfig.everything_together:
                self.f_add_edge = AddEdge()
            if GraphNetConfig.predict_edge_type_first:
                self.f_choose_node = nn.ModuleList([ChooseNode() for i in range(GraphNetConfig.num_edge_types)])
            else:
                self.f_choose_node = ChooseNode()

        if GraphNetConfig.separate_wall_edges_step:
            self.f_add_edge_wall = AddEdge()
            if GraphNetConfig.predict_edge_type_first:
                self.f_choose_node_wall = nn.ModuleList([ChooseNode() for i in range(GraphNetConfig.num_edge_types)])
            else:
                self.f_choose_node_wall = ChooseNode()
        
        if not (GraphNetConfig.node_and_type_together or GraphNetConfig.predict_edge_type_first):
            self.f_edge_type = ChooseEdgeType()

        self.initializer = Initializer()
        self.wall_initializer = WallInitializer()

        self.clear()

    def clear(self):
        """
        Set initial state
        """
        self.nodes = []
        self.edges = []
        self.edge_types = []
        self._node_vectors = None
        self._node_one_hot = None
        self.edge_vectors = None
        self.u_indices = []
        self.v_indices = []
        self._node_vectors_backup = {}
        self.backup("initial") #backup initial state
    
    @property
    def node_vectors(self):
        if GraphNetConfig.include_one_hot:
            return torch.cat((self._node_vectors, self._node_one_hot), 1)
        else:
            return self._node_vectors
    
    @node_vectors.setter #useless abstraction that end up only used once.
    def node_vectors(self, new_nv):
        self._node_vectors = new_nv

    def backup(self, version):
        if version is not None:
            self._node_vectors_backup[version] = self._node_vectors

    def restore(self, version):
        if version is not None:
            self._node_vectors = self._node_vectors_backup[version]
    
    def new_node(self, category, sym_idx=0, is_hub=False, is_spoke=False, is_chain=False):
        new_node = Node(category, sym_idx, is_hub, is_spoke, is_chain)
        num_cat = GraphNetConfig.num_cat
        feature_size = num_cat
        if GraphNetConfig.predict_sym_type:
            feature_size += GraphNetConfig.num_sym_classes + 3 #2 for hub and chain
        if GraphNetConfig.cuda:
            e = torch.zeros(1, feature_size).cuda()
        else:
            e = torch.zeros(1, feature_size)
        e[0,category] = 1
        if GraphNetConfig.predict_sym_type:
            e[0, sym_idx + num_cat] = 1
            e[0, -3] = int(is_hub)
            e[0, -2] = int(is_spoke)
            e[0, -1] = int(is_chain)
        h_v = self.initializer(self, e)

        if GraphNetConfig.include_one_hot:
            if self._node_one_hot is None:
                self._node_one_hot = e
            else:
                self._node_one_hot = torch.cat((self._node_one_hot, e), 0)

        self._add_node(h_v)
        self.nodes.append(new_node)
    
    def new_wall_node(self, category, length):
        new_node = Node(category, is_arch=True)
        if GraphNetConfig.include_one_hot:
            feature_size = GraphNetConfig.num_cat
            if GraphNetConfig.predict_sym_type:
                feature_size += GraphNetConfig.num_sym_classes + 3 #2 for hub and chain
            if GraphNetConfig.cuda:
                e = torch.zeros(1, feature_size).cuda()
            else:
                e = torch.zeros(1, feature_size)
            if self._node_one_hot is None:
                self._node_one_hot = e
            else:
                self._node_one_hot = torch.cat((self._node_one_hot, e), 0)
        
        feature_size = GraphNetConfig.num_arch_cat + 1 #One hot plus length
        if GraphNetConfig.cuda:
            e = torch.zeros(1, feature_size).cuda()
        else:
            e = torch.zeros(1, feature_size)
        e[0,category] = 1
        if length is None:
            length = 0
        e[0, -1] = length
        h_v = self.wall_initializer(e)
        self._add_node(h_v)
        self.nodes.append(new_node)

    def _add_node(self, h_v):
        if self._node_vectors is None:
            self._node_vectors = h_v
            for key in self._node_vectors_backup.keys():
                self._node_vectors_backup[key] = h_v
        else:
            self._node_vectors = torch.cat((self._node_vectors, h_v), 0)
            for key in self._node_vectors_backup.keys():
                self._node_vectors_backup[key] = torch.cat((self._node_vectors_backup[key], h_v), 0)

    def new_edge(self, u_index, v_index, edge_type):
        assert edge_type is not None
        if GraphNetConfig.cuda:
            x_u_v = torch.zeros(1,GraphNetConfig.edge_size).cuda()
        else:
            x_u_v = torch.zeros(1,GraphNetConfig.edge_size)
        x_u_v[0,edge_type] = 1
        self._add_edge(u_index, v_index, x_u_v, edge_type)

    def new_wall_edge(self, u_index, v_index, angle, edge_type):
        if GraphNetConfig.cuda:
            x_u_v = torch.zeros(1,GraphNetConfig.edge_size).cuda()
        else:
            x_u_v = torch.zeros(1,GraphNetConfig.edge_size)
        x_u_v[0,edge_type] = 1
        if angle is not None:
            x_u_v[0,-1] = angle
        self._add_edge(u_index, v_index, x_u_v, edge_type)

    def _add_edge(self, u_index, v_index, x_u_v, edge_type):
        if self.edge_vectors is None:
            self.edge_vectors = x_u_v
        else:
            self.edge_vectors = torch.cat((self.edge_vectors, x_u_v),0)
        self.u_indices.append(u_index)
        self.v_indices.append(v_index)
        e_index = len(self.edges)
        self.edges.append((u_index, v_index))
        self.nodes[u_index].outgoing.append(e_index)
        self.nodes[v_index].incoming.append(e_index)
        self.edge_types.append(edge_type)

    def _get_logits_add_node(self):
        self.restore(GraphNetConfig.restore_target["an"])
        logits_add_node = self.f_add_node(self)
        if GraphNetConfig.backup_needed["an"]:
            self.backup("an")
        return logits_add_node

    def _get_logits_sym_type(self, cat):
        return self.f_sym_type[cat](self)

    def _get_logits_add_edge(self, wall_mode=-1):
        self.restore(GraphNetConfig.restore_target["ae"])
        if wall_mode == 1:
            f_add_edge = self.f_add_edge_wall
        else:
            f_add_edge = self.f_add_edge
        #print("Executing add edge")
        logits_add_edge = f_add_edge(self, -1)
        if GraphNetConfig.backup_needed["ae"]:
            self.backup("ae")
        return logits_add_edge

    def _get_logits_nodes(self, wall_mode=-1, edge_type=None):
        self.restore(GraphNetConfig.restore_target["node"])
        #print("Executing choose node")
        if wall_mode == 1:
            f_choose_node = self.f_choose_node_wall
        else:
            f_choose_node = self.f_choose_node

        if edge_type is not None:
            f_choose_node = f_choose_node[edge_type]

        logits_nodes = f_choose_node(self, -1, wall_mode)
        if GraphNetConfig.backup_needed["node"]:
            self.backup("node")

        return logits_nodes

    def _get_logits_edge_type(self, adj, direction, repropagate=True):
        self.restore(GraphNetConfig.restore_target["et"])
        #print("Executing choose edge type")
        if direction == 0:
            logits_edge_type = self.f_edge_type(self, adj, -1, repropagate)
        else:
            logits_edge_type = self.f_edge_type(self, -1, adj, repropagate)
        if GraphNetConfig.backup_needed["et"]:
            self.backup("et")
        return logits_edge_type

    def add_walls(self, wall_nodes):
        for (i, gt_node) in enumerate(wall_nodes):
            self.new_wall_node(gt_node.category, gt_node.length)
            for (gt_adj, gt_edge_type, gt_direction, gt_angle) in gt_node.adjacent:
                if gt_direction == 0:
                    self.new_wall_edge(gt_adj, i, gt_angle, gt_edge_type)
                else:
                    self.new_wall_edge(i, gt_adj, gt_angle, gt_edge_type)
        self.num_walls = len(self.nodes)

    def get_structure_index(self, is_hub, is_spoke, is_chain):
        # Convert bool properties to indices, to be used for combined prediction
        # 0: No 
        # 1: Hub
        # 2: Hub & Chain
        # 3: Spoke
        # 4: Chain
        #assert (is_spoke and (is_hub or is_chain)) == False
        if is_hub:
            if is_chain:
                return 2
            else:
                return 1
        elif is_chain:
            return 4
        elif is_spoke:
            return 3
        else:
            return 0
    
    def structure_index_to_bools(self, idx):
        if idx == 0:
            return False, False, False
        elif idx == 1:
            return True, False, False
        elif idx == 2:
            return True, False, True
        elif idx == 3:
            return False, True, False
        elif idx == 4:
            return False, False, True
        else:
            raise NotImplementedError

    def train_step(self, graph, debug=False):
        #Training code
        self.clear() #clear internal states
        losses = {}
        losses["an"] = torch.zeros(1) #add node
        losses["ae"] = torch.zeros(1) #add edge
        losses["node"] = torch.zeros(1) #choose node
        losses["et"] = torch.zeros(1) #edge type
        losses["sym"] = torch.zeros(1) #symmetry type

        if GraphNetConfig.cuda:
            for key in losses.keys():
                losses[key] = losses[key].cuda()

        if debug:
            print("====================================================")
            print("Creating New Graph..................................")

        gt_nodes, wall_nodes = self.load_from_relation_graph(graph)
        self.gt_nodes = gt_nodes

        if wall_nodes is not None:
            self.add_walls(wall_nodes) #initialize walls if training with walls

        for (i, gt_node) in enumerate(gt_nodes): #teacher force through all nodes in training data
            #First, train the add node module to predict the ground truth node category
            logits_add_node = self._get_logits_add_node()
            if GraphNetConfig.cuda: #should change to todevice...
                target = torch.zeros(1, dtype=torch.long).cuda()
            else:
                target = torch.zeros(1, dtype=torch.long)
            target[0] = gt_node.category
            losses["an"] += F.cross_entropy(logits_add_node, target)
            
            if debug:
                self._debug_add_node(logits_add_node, gt_node.category)
            
            #Second, if symmetry type is included, train the sym type module to predict the 
            #   ground truth symmetry and superstructure type
            if GraphNetConfig.predict_sym_type:
                logits_sym = self._get_logits_sym_type(gt_node.category)
                if GraphNetConfig.cuda:
                    target = torch.zeros(1, dtype=torch.long).cuda()
                else:
                    target = torch.zeros(1, dtype=torch.long)
                structure_idx = self.get_structure_index(gt_node.is_hub,\
                                                         gt_node.is_spoke, gt_node.is_chain)
                target[0] = gt_node.sym_idx * 5 + structure_idx #cartesian product of sym types and 5 structure types
                losses["sym"] += F.cross_entropy(logits_sym, target)

                if debug:
                    self._debug_sym_class(logits_sym, gt_node.sym_idx, structure_idx)

                #Done with training node prediction, add ground truth node to the graph
                self.new_node(gt_node.category, gt_node.sym_idx, gt_node.is_hub, gt_node.is_spoke, gt_node.is_chain)
            else:
                #If no sym type, just add node with category
                self.new_node(gt_node.category)
            
            def train_edges(adjacent, wall_mode):
                #Train edges
                #adjacent: all the edges connected to the node, to be trained
                #wall_mode: -1: all nodes together
                #            0: object nodes
                #            1: wall nodes

                if debug:
                    self._debug_start_adding_edges(wall_mode)

                offset = self.num_walls*2 if wall_mode == 0 else 0 #if object nodes only, calculate offset to skip wall nodes
                if GraphNetConfig.element_wise_loss_node: #prepare for different loss formulation, not used in final paper
                    remaining_adj = []
                    if GraphNetConfig.node_and_type_together:
                        for (gt_adj, gt_edge_type, gt_direction) in adjacent:
                            remaining_adj.append((gt_adj*2+gt_direction-offset)*GraphNetConfig.num_edge_types + gt_edge_type)
                    else:
                        for (gt_adj, gt_edge_type, gt_direction) in adjacent:
                            remaining_adj.append(gt_adj*2 + gt_direction - offset)

                for (gt_adj, gt_edge_type, gt_direction) in adjacent: #iterate over all adjacent edges
                    if not GraphNetConfig.everything_together: #If we predict adjacent node and type separately, not used in final paper
                        logits_add_edge = self._get_logits_add_edge(wall_mode=wall_mode)
                        if GraphNetConfig.cuda:
                            target = torch.ones([1,1]).cuda()
                        else:
                            target = torch.ones([1,1])
                        if GraphNetConfig.predict_edge_type_first:
                            target[0] = gt_edge_type
                            target = target[0].long()
                            losses["ae"] += F.cross_entropy(logits_add_edge, target)
                        else:
                            losses["ae"] += F.binary_cross_entropy_with_logits(logits_add_edge, target)
                        
                        if debug:
                            if GraphNetConfig.predict_edge_type_first:
                                self._debug_add_edge(logits_add_edge, gt_edge_type)
                            else:
                                self._debug_add_edge(logits_add_edge, 1)

                    #Grab the logits for all edge types, for all possible nodes and directions
                    et = gt_edge_type if GraphNetConfig.predict_edge_type_first else None
                    logits_nodes = self._get_logits_nodes(wall_mode=wall_mode, edge_type=et)
                    if GraphNetConfig.everything_together: #If we combine all edge steps together, used in final paper
                        auxiliary = torch.ones(1,1) * GraphNetConfig.no_edge_logit #Just to ensure the logits are not too negative
                        if GraphNetConfig.cuda:
                            auxiliary = auxiliary.cuda()
                        logits_nodes = torch.cat((logits_nodes, auxiliary),1)
                    gt_idx = gt_adj * 2 + gt_direction - offset #take care of two possible directions
                    if GraphNetConfig.element_wise_loss_node: #different loss formulation, not used
                        logits_nodes_flat = logits_nodes.view(-1)
                        target = torch.zeros_like(logits_nodes_flat)
                        target[remaining_adj] = 1
                        losses["node"] += F.binary_cross_entropy_with_logits(logits_nodes_flat, target)
                        if GraphNetConfig.node_and_type_together:
                            remaining_adj.remove(gt_idx*GraphNetConfig.num_edge_types + gt_edge_type)
                        else:
                            remaining_adj.remove(gt_idx)
                    else: #loss formulation used in paper (just cross entropy over everything)
                        if GraphNetConfig.cuda:
                            target = torch.zeros(1, dtype=torch.long).cuda()
                        else:
                            target = torch.zeros(1, dtype=torch.long)

                        #compute target based on what we are predicting
                        if GraphNetConfig.node_and_type_together:
                            target[0] = gt_idx * GraphNetConfig.num_edge_types + gt_edge_type
                        else:
                            target[0] = gt_idx
                        losses["node"] += F.cross_entropy(logits_nodes, target)
                    
                    if debug:
                        if GraphNetConfig.node_and_type_together:
                            self._debug_choose_node_and_type(logits_nodes, gt_adj, gt_direction, gt_edge_type, wall_mode)
                        else:
                            self._debug_choose_node(logits_nodes, gt_adj, gt_direction, wall_mode)
                    
                    #If we predict edge type separately, do this step (not used in paper)
                    if not (GraphNetConfig.predict_edge_type_first or GraphNetConfig.node_and_type_together):
                        logits_edge_type = self._get_logits_edge_type(gt_adj, gt_direction)
                        if GraphNetConfig.cuda:
                            target = torch.zeros(1, dtype=torch.long).cuda()
                        else:
                            target = torch.zeros(1, dtype=torch.long)
                        target[0] = gt_edge_type
                        losses["et"] += F.cross_entropy(logits_edge_type, target)
                        
                        if debug:
                            self._debug_edge_type(logits_edge_type, gt_edge_type)

                    if gt_direction == 0: #Incoming to new node
                        self.new_edge(gt_adj, len(self.nodes)-1, gt_edge_type) #edge type ignored if not predicting anyways
                    else: #Outgoing from new node
                        self.new_edge(len(self.nodes)-1, gt_adj, gt_edge_type) #edge type ignored if not predicting anyways

                #All ground truth edges are processed, now train the network to learn to stop
                #If we don't skip the should we add edge step (i.e. not everything_together)
                #Train the add edge module to learn to stop
                if not GraphNetConfig.everything_together:
                    logits_add_edge = self._get_logits_add_edge(wall_mode=wall_mode)
                    if GraphNetConfig.cuda:
                        target = torch.zeros([1,1]).cuda()
                    else:
                        target = torch.zeros([1,1])
                    if GraphNetConfig.predict_edge_type_first:
                        target = target[0].long()
                        target[0] = GraphNetConfig.num_edge_types
                        losses["ae"] += F.cross_entropy(logits_add_edge, target)
                    else:
                        losses["ae"] += F.binary_cross_entropy_with_logits(logits_add_edge, target)
                    
                    if debug:
                        self._debug_stop_adding_edge(logits_add_edge, wall_mode)
                else: #otherwise train the choose which node module to favor the fixed no edge logit
                    et = gt_edge_type if GraphNetConfig.predict_edge_type_first else None
                    logits_nodes = self._get_logits_nodes(wall_mode=wall_mode, edge_type=et)
                    if logits_nodes is not None:
                        auxiliary = torch.ones(1,1) * GraphNetConfig.no_edge_logit #Just to ensure the logits are not too negatifve
                        if GraphNetConfig.cuda:
                            auxiliary = auxiliary.cuda()
                        if GraphNetConfig.cuda:
                            target = torch.zeros(1, dtype=torch.long).cuda()
                        else:
                            target = torch.zeros(1, dtype=torch.long)
                        target[0] = logits_nodes.size()[1]
                        logits_nodes = torch.cat((logits_nodes, auxiliary),1)
                        losses["node"] += F.cross_entropy(logits_nodes, target)
                        if debug:
                            self._debug_choose_node_and_type(logits_nodes, 0, 0, 0, wall_mode, stop=True)
                    else:
                        if debug:
                            print("No nodes to connect to, skip this step")

                if len(adjacent) > 0 and GraphNetConfig.auxiliary_choose_node: #some random experiments, ignore it
                    logits_nodes = self._get_logits_nodes(wall_mode=wall_mode)
                    if GraphNetConfig.cuda:
                        target = torch.zeros(1, dtype=torch.long).cuda()
                    else:
                        target = torch.zeros(1, dtype=torch.long)
                    target[0] = logits_nodes.size()[1]
                    logits_nodes = torch.cat((logits_nodes, torch.ones(1,1)),1)
                    losses["node"] += F.cross_entropy(logits_nodes, target) * 0.1

            # end subfunction definition
            # --------------------------
            if GraphNetConfig.per_node_prediction: #different factorization than one used in paper, ignore
                n_nodes = len(self.nodes) - 1
                NOEDGE = GraphNetConfig.num_edge_types
                gt_edges = [NOEDGE for _ in range(n_nodes * 2)]
                adjacent = gt_node.adjacent[0] + gt_node.adjacent[1]
                for (gt_adj, gt_edge_type, gt_direction) in adjacent:
                    gt_edges[gt_adj*2+gt_direction] = gt_edge_type
                
                gt_edges = list(enumerate(gt_edges))
                random.shuffle(gt_edges)
                training_seq = []
                for (i, gt_edge_type) in gt_edges:
                    training_seq.append((i, gt_edge_type))
                    if gt_edge_type != NOEDGE:
                        training_seq.append((i, NOEDGE))
                
                repropagate = True
                for (i, gt_edge_type) in training_seq:
                    gt_adj = i//2
                    gt_direction = i%2
                    logits_edge_type = self._get_logits_edge_type(gt_adj, gt_direction, repropagate)
                    if GraphNetConfig.cuda:
                        target = torch.zeros(1, dtype=torch.long).cuda()
                    else:
                        target = torch.zeros(1, dtype=torch.long)
                    target[0] = gt_edge_type
                    losses["et"] += F.cross_entropy(logits_edge_type, target)
                    
                    if gt_edge_type != NOEDGE:
                        if gt_direction == 0: #Incoming to new node
                            self.new_edge(gt_adj, len(self.nodes)-1, gt_edge_type)
                        else: #Outgoing from new node
                            self.new_edge(len(self.nodes)-1, gt_adj, gt_edge_type)
                        repropagate = True
                    else:
                        repropagate = False
                #print(Categorical(logits=p_add_node).sample())
            else: #standard one, choosing 1 or 2 steps depend on whether we process wall nodes separately
                if GraphNetConfig.separate_wall_edges_step:
                    train_edges(gt_node.adjacent[0], wall_mode=0) #object nodes first
                    train_edges(gt_node.adjacent[1], wall_mode=1) #then wall nodes
                else:
                    assert(len(gt_node.adjacent[1]) == 0)
                    train_edges(gt_node.adjacent[0], wall_mode=-1)
        
        #finished iterating over all nodes
        #finally, train the add node module to learn to stop
        logits_add_node = self._get_logits_add_node()
        if GraphNetConfig.cuda:
            target = torch.zeros(1, dtype=torch.long).cuda()
        else:
            target = torch.zeros(1, dtype=torch.long)
        target[0] = GraphNetConfig.num_cat
        losses["an"] += F.cross_entropy(logits_add_node, target)
        
        if debug:
            self._debug_stop_adding_node(logits_add_node)

        return losses

    def sample(self, graph, debug=False, return_relation_graph=False, keep_edge_type=None):
        #see train for details, basically the same structure, but sampling instead of getting losses
        _, wall_nodes = self.load_from_relation_graph(graph)

        graph = []
        self.clear()
        if debug:
            print("====================================================")
            print("Creating New Graph..................................")
        
        if wall_nodes is not None:
            self.add_walls(wall_nodes)
        
        temp = GraphNetConfig.temp_dict
        logits_add_node = self._get_logits_add_node()
        cat_sample = Categorical(logits=logits_add_node*temp["an"]).sample()
        while cat_sample != GraphNetConfig.num_cat:
            if debug:
                self._debug_add_node(logits_add_node, cat_sample)

            if GraphNetConfig.predict_sym_type:
                logits_sym = self._get_logits_sym_type(cat_sample)
                
                sym_sample = Categorical(logits=logits_sym*temp["sym"]).sample()
                if keep_edge_type is not None: #Ugly but works whatever
                    p_debug = list(F.softmax(logits_sym*temp["sym"], dim=1)[0].cpu().data.numpy())
                    while p_debug[sym_sample] < 0.03:
                        sym_sample = Categorical(logits=logits_sym*temp["sym"]).sample()
                structure_sample = sym_sample % 5
                sym_sample = sym_sample // 5
                if debug:
                    self._debug_sym_class(logits_sym, sym_sample, structure_sample)

                hub_sample, spoke_sample, chain_sample = self.structure_index_to_bools(structure_sample)

                self.new_node(cat_sample, sym_sample, hub_sample, spoke_sample, chain_sample)
                output_node = GroundTruthNode(cat_sample[0].cpu().numpy(), [], sym_idx=sym_sample, is_hub=hub_sample, is_spoke=spoke_sample, is_chain=chain_sample)
            else:
                output_node = GroundTruthNode(cat_sample[0].cpu().numpy(), [])
                self.new_node(cat_sample)

            def sample_edges(wall_mode):
                if GraphNetConfig.predict_edge_type_first:
                    STOP = GraphNetConfig.num_edge_types
                else:
                    STOP = 0
                if debug:
                    self._debug_start_adding_edges(wall_mode)
                
                if not GraphNetConfig.everything_together:
                    logits_add_edge = self._get_logits_add_edge(wall_mode=wall_mode)
                    if len(self.nodes) == 1 or ((len(self.nodes) == self.num_walls+1) and (wall_mode==0)):
                        add_sample = STOP #Take care of 1 node situation
                    else:
                        if GraphNetConfig.predict_edge_type_first:
                            add_sample = Categorical(logits=logits_add_edge*temp["ae"]).sample()
                        else:
                            add_sample = Bernoulli(logits=logits_add_edge*temp["ae"]).sample()
                else:
                    logits_nodes = self._get_logits_nodes(wall_mode=wall_mode, edge_type=None)
                    if logits_nodes is None:
                        add_sample = 0
                        STOP = 0
                    else:
                        auxiliary = torch.ones(1,1) * GraphNetConfig.no_edge_logit #Just to ensure the logits are not too negatifve
                        if GraphNetConfig.cuda:
                            auxiliary = auxiliary.cuda()
                        STOP = logits_nodes.size()[1]
                        logits_nodes = torch.cat((logits_nodes, auxiliary),1)

                        add_sample = Categorical(logits=logits_nodes*temp["node"]).sample()
                        if keep_edge_type is not None:
                            p_debug = list(F.softmax(logits_nodes*temp["node"], dim=1)[0].cpu().data.numpy())
                            def check_valid(sample): 
                                #follow the data processing step, reject edges that are always filtered out during graph pruning
                                if p_debug[sample] < 0.002:
                                    return False
                                if sample == STOP:
                                    return True
                                node_sample = sample[0].cpu().item()
                                edge_type_sample = node_sample % GraphNetConfig.num_edge_types
                                node_sample = node_sample // GraphNetConfig.num_edge_types
                                if wall_mode == 0:
                                    node_sample += self.num_walls * 2
                                adj_sample = node_sample // 2
                                direction_sample = node_sample % 2
                                base_name = EdgeType.BaseTypes[edge_type_sample//3]
                                distance = EdgeType.Distances[edge_type_sample%3]
                                if direction_sample == 0:
                                    start, end = adj_sample, -1
                                else:
                                    start, end = -1, adj_sample
                                if base_name == "support":
                                    return True
                                if self.nodes[start].is_hub and self.nodes[end].is_spoke:
                                    return True
                                if self.nodes[start].is_chain and self.nodes[end].is_chain:
                                    return True
                                if base_name == "front" and self.nodes[start].sym_idx in [0,3]:
                                    return True
                                if distance == "adjacent":
                                    cls = RelationshipGraph
                                    coarse_cat_1 = cls.cat_final_to_coarse_force(self.nodes[start].category_verbose)
                                    coarse_cat_2 = cls.cat_final_to_coarse_force(self.nodes[end].category_verbose)
                                    if coarse_cat_1 == coarse_cat_2:
                                        return True
                                edge_name = f"{self.nodes[start].category_verbose},{self.nodes[end].category_verbose},{base_name},{distance}"
                                if edge_name in keep_edge_type:
                                    if keep_edge_type[edge_name]:
                                        return True
                                print(edge_name)
                                return False

                            add_sample = Categorical(logits=logits_nodes*temp["node"]).sample()
                
                while add_sample != STOP and len(output_node.adjacent) < 10:
                    if debug and not GraphNetConfig.everything_together:
                        self._debug_add_edge(logits_add_edge, add_sample)
                    et = int(add_sample) if GraphNetConfig.predict_edge_type_first else None

                    if GraphNetConfig.everything_together:
                        node_sample = add_sample[0].cpu().item()
                    else:
                        logits_nodes = self._get_logits_nodes(wall_mode=wall_mode, edge_type=et)
                        node_sample = Categorical(logits=logits_nodes*temp["node"]).sample()[0].cpu().item()
                    if GraphNetConfig.node_and_type_together:
                        edge_type_sample = node_sample % GraphNetConfig.num_edge_types
                        node_sample = node_sample // GraphNetConfig.num_edge_types

                    if wall_mode == 0:
                        node_sample += self.num_walls * 2
                    adj_sample = node_sample // 2
                    direction_sample = node_sample % 2

                    if debug:
                        if GraphNetConfig.node_and_type_together:
                            self._debug_choose_node_and_type(logits_nodes, adj_sample, direction_sample, edge_type_sample, wall_mode)
                        else:
                            self._debug_choose_node(logits_nodes, adj_sample, direction_sample, wall_mode)
                    if GraphNetConfig.predict_edge_type_first:
                        edge_type_sample = add_sample[0].cpu().item()
                    elif GraphNetConfig.node_and_type_together:
                        pass
                    else:
                        logits_edge_type = self._get_logits_edge_type(adj_sample, direction_sample)
                        edge_type_sample = Categorical(logits=logits_edge_type*temp["et"]).sample()[0].cpu().item()
                        if debug:
                            self._debug_edge_type(logits_edge_type, edge_type_sample)

                    output_node.adjacent.append((adj_sample, edge_type_sample, direction_sample))
                    
                    if direction_sample == 0:
                        self.new_edge(adj_sample, len(self.nodes)-1, edge_type_sample)
                    else:
                        self.new_edge(len(self.nodes)-1, adj_sample, edge_type_sample)
                    
                    if GraphNetConfig.everything_together:
                        logits_nodes = self._get_logits_nodes(wall_mode=wall_mode, edge_type=None)
                        auxiliary = torch.ones(1,1) * GraphNetConfig.no_edge_logit #Just to ensure the logits are not too negatifve
                        if GraphNetConfig.cuda:
                            auxiliary = auxiliary.cuda()
                        STOP = logits_nodes.size()[1]
                        logits_nodes = torch.cat((logits_nodes, auxiliary),1)
                        add_sample = Categorical(logits=logits_nodes*temp["node"]).sample()
                    else:
                        logits_add_edge = self._get_logits_add_edge(wall_mode=wall_mode)
                        if GraphNetConfig.predict_edge_type_first:
                            add_sample = Categorical(logits=logits_add_edge*temp["ae"]).sample()
                        else:
                            add_sample = Bernoulli(logits=logits_add_edge*temp["ae"]).sample()

                if debug and not GraphNetConfig.everything_together:
                    if GraphNetConfig.everything_together:
                        if logits_nodes is None:
                            print("No nodes to connect to, skip this step")
                        else:
                            self._debug_choose_node_and_type(logits_nodes, 0, 0, 0, wall_mode, stop=True)
                    else:
                        self._debug_stop_adding_edge(logits_add_edge, wall_mode)
            
            if GraphNetConfig.per_node_prediction:
                NOEDGE = GraphNetConfig.num_edge_types
                if debug:
                    print("-------------------")
                    print("Checking edges to connect to...")
                n_nodes = len(self.nodes) - 1
                check_order = list(range(n_nodes * 2))
                random.shuffle(check_order)
                repropagate = True
                for i in check_order:
                    check_adj = i//2
                    check_direction = i%2
                    logits_edge_type = self._get_logits_edge_type(check_adj, check_direction, repropagate)

                    edge_type_sample = Categorical(logits=logits_edge_type*temp["et"]).sample()[0].cpu().item()
                    if edge_type_sample != NOEDGE:
                        if check_direction == 0: #Incoming to new node
                            self.new_edge(check_adj, len(self.nodes)-1, edge_type_sample)
                        else: #Outgoing from new node
                            self.new_edge(len(self.nodes)-1, check_adj, edge_type_sample)
                        repropagate = True
                        output_node.adjacent.append((check_adj, edge_type_sample, check_direction))
                    else:
                        repropagate = False

                    if debug:
                        print("-------------------")
                        print("Graph Status Recap:")
                        print("Nodes:")
                        self._print_current_state()
                        direction = "Incoming" if check_direction==0 else "Outgoing"
                        print(f"Checking node {check_adj}, {self.nodes[check_adj].category_verbose}, {direction}...")
                        p_debug = list(F.softmax(logits_edge_type*temp["et"], dim=1)[0].cpu().data.numpy())
                        logits_debug = list(logits_edge_type[0].cpu().data.numpy())
                        for i in range(len(p_debug)):
                            base_name = EdgeType.BaseTypes[i//3]
                            distance = EdgeType.Distances[i%3]
                            if i == NOEDGE:
                                base_name = "No Edge"
                            print(f"Relation {base_name}, {distance}, {logits_debug[i]}, {p_debug[i]}")

                        base_name = EdgeType.BaseTypes[edge_type_sample//3]
                        if edge_type_sample == NOEDGE:
                            base_name = "no edge"
                            distance = "distant"
                        distance = EdgeType.Distances[edge_type_sample%3]
                        print(f"Choosing edge type {base_name}, {distance}")
                        _ = input()
                #print(Categorical(logits=p_add_node).sample())
            else:
                if GraphNetConfig.separate_wall_edges_step:
                    sample_edges(wall_mode=0)
                    sample_edges(wall_mode=1)
                else:
                    sample_edges(wall_mode=-1)
            graph.append(output_node)
            logits_add_node = self._get_logits_add_node()
            cat_sample = Categorical(logits=logits_add_node*temp["an"]).sample()

        if debug:
            self._debug_stop_adding_node(logits_add_node)
        
        if return_relation_graph:
            return graph, self.convert_to_relation_graph(wall_nodes, graph)
        else:
            return graph
    
    def load_from_relation_graph(self, graph):
        #convert data into training format
        #####################
        #Warning: does not preserve input graph
        #####################
        shuffle_nodes = GraphNetConfig.shuffle_nodes
        shuffle_edges = GraphNetConfig.shuffle_edges
        include_walls = GraphNetConfig.include_walls
        categories = GraphNetConfig.categories
        arch_categories = GraphNetConfig.arch_categories
        cat_to_index = {categories[i]:i for i in range(len(categories))}
        arch_cat_to_index = {arch_categories[i]:i for i in range(len(arch_categories))}

        nodes = []
        ids = []
        walls = []
        wall_ids = []
        for node in graph.nodes:
            if node.category_name in categories:
                pass
            else:
                walls.append(node)

        if shuffle_nodes:
            random.shuffle(walls)
        
        wall_ids = [node.id for node in walls]
        #nodes = sorted(nodes, key=lambda x:cat_to_index[parse_cat(x)])
        #print(nodes)
        node_to_index = {walls[i].id:i for i in range(len(walls))}
        #print(node_to_index)
        
        edge_count = 0
        graph_gt=[]
        graph_gt_walls = []

        for (i, node) in enumerate(walls):
            if node.category_name in arch_categories:
                previous_ids = wall_ids[0:i]
                adj = []
                for edge in node.in_edges:
                    #print(edge.edge_type.base_name)
                    if edge.start_node.id in previous_ids:
                        adj.append((node_to_index[edge.start_node.id],edge.edge_type.index, 0, edge.wall_ang))
                for edge in node.out_edges:
                    #print(edge.edge_type.base_name)
                    if edge.end_node.id in previous_ids:
                        adj.append((node_to_index[edge.end_node.id],edge.edge_type.index, 1, edge.wall_ang))

                if shuffle_edges:
                    random.shuffle(adj)
                #edge_count += len(adj)
                graph_gt_walls.append(GroundTruthNode(arch_cat_to_index[node.category_name], adj, node.wall_length, id=node.id))

        graph.remove_non_wall_arch_nodes()
        for node in graph.nodes:
            if node.category_name in categories:
                nodes.append(node)
            else:
                pass

        if shuffle_nodes:
            random.shuffle(nodes)
        else:
            nodes_imp = sorted(nodes, key=lambda x:cat_to_index[x.category_name])
            l = len(nodes_imp)
            nl = [node for node in nodes_imp]
            nodes = []
            spokes = {node:[] for node in nodes_imp}

            for chain in graph.get_all_chains():
                st = chain[0].start_node
                if st in nodes_imp:
                    for chain_edge in chain:
                        if chain_edge.end_node in nodes_imp:
                            spokes[st].append(chain_edge.end_node)
                            nodes_imp.remove(chain_edge.end_node)
                            for node in spokes[chain_edge.end_node]:
                                spokes[st].append(node)
                            spokes[chain_edge.end_node] = []

            for node in [node for node in nodes_imp]:
                if node.is_hub:
                    for e in node.out_edges:
                        if e.is_spoke:
                            if not e.end_node in nodes_imp:
                                print('Found a spoke that was also a chain')
                            else:
                                spokes[node].append(e.end_node)
                                nodes_imp.remove(e.end_node)
            for node in nodes_imp:
                nodes.append(node)
                for node in spokes[node]:
                    nodes.append(node)
            if len(nodes) != l:
                nodes = nl

        ids = [node.id for node in nodes]
        if include_walls:
            ids = wall_ids + ids
            nodes = walls + nodes
        else:
            wall_ids = []
        
        node_to_index = {nodes[i].id:i for i in range(len(nodes))}

        for (i, node) in enumerate(nodes):
            if node.category_name in categories:
                cat_index = cat_to_index[node.category_name]
                previous_ids = ids[0:i]
                adj = []
                wall_adj = []
                for edge in node.in_edges:
                    if edge.start_node.id in wall_ids:
                        wall_adj.append((node_to_index[edge.start_node.id],edge.edge_type.index, 0))
                    elif edge.start_node.id in previous_ids:
                        adj.append((node_to_index[edge.start_node.id],edge.edge_type.index, 0))
                for edge in node.out_edges:
                    if edge.end_node.id in wall_ids:
                        wall_adj.append((node_to_index[edge.end_node.id],edge.edge_type.index, 1))
                    elif edge.end_node.id in previous_ids:
                        adj.append((node_to_index[edge.end_node.id],edge.edge_type.index, 1))

                if not GraphNetConfig.separate_wall_edges_step:
                    adj = adj + wall_adj
                    wall_adj = []
                if shuffle_edges:
                    random.shuffle(wall_adj)
                    random.shuffle(adj)

                graph_gt.append(GroundTruthNode(cat_index, (adj, wall_adj), sym_idx=node.sym_idx, is_hub=node.is_hub, is_spoke=node.is_spoke, is_chain=node.is_chain, id=node.id))
        return graph_gt, graph_gt_walls
        

    def convert_to_relation_graph(self, walls, nodes):
        #convert the graph back to the relation graph used for other tasks
        sym_classes = ["", "__SYM_ROTATE_UP_INF", "__SYM_REFLECT_FB", "__SYM_REFLECT_LR",
                       "__SYM_ROTATE_UP_2", "__SYM_ROTATE_UP_4", "__SYM_REFLECT_C1", 
                       "__SYM_REFLECT_C2"]
        graph = RelationshipGraph()
        idx = 0

        id_map = {}
        for node in walls:
            graph.add_node(RelationshipGraph.Node(
                node.id, GraphNetConfig.arch_categories[node.category], \
                [sym_classes[node.sym_idx]], wall_length=node.length, graph=graph
            ))

            id_map[idx] = node.id

            for (adj, edge_type, direction, angle) in node.adjacent:
                base_name = EdgeType.BaseTypes[edge_type//3]
                if base_name == 'support':
                    distance = None
                else:
                    distance = EdgeType.Distances[edge_type%3]
                et = RelationshipGraph.EdgeType(base_name, distance)
                if direction == 0:
                    s = id_map[adj]
                    t = id_map[idx]
                else:
                    s = id_map[idx]
                    t = id_map[adj]
                graph.add_edge(RelationshipGraph.Edge(str(s), str(t), et, wall_ang=angle, graph=graph))

            idx += 1

        for node in nodes:
            graph.add_node(RelationshipGraph.Node(
                f"synth_{idx}", GraphNetConfig.categories[node.category], \
                [sym_classes[node.sym_idx]], is_hub=node.is_hub, is_spoke=node.is_spoke, is_chain=node.is_chain, graph=graph
            ))
            id_map[idx] = f"synth_{idx}"

            for (adj, edge_type, direction) in node.adjacent:
                base_name = EdgeType.BaseTypes[edge_type//3]
                if base_name == 'support':
                    distance = None
                else:
                    distance = EdgeType.Distances[edge_type%3]
                et = RelationshipGraph.EdgeType(base_name, distance)
                if direction == 0:
                    s = id_map[adj]
                    t = id_map[idx]
                else:
                    s = id_map[idx]
                    t = id_map[adj]
                graph.add_edge(RelationshipGraph.Edge(str(s), str(t), et, graph=graph), ignore_duplicate=True)
            idx += 1
        graph._RelationshipGraph__sort_edges()
        return graph

#-------------------------------
#various functions for debugging
#-------------------------------
    def _print_current_state(self):
        def get_edge_type(i):
            base_name = EdgeType.BaseTypes[i//3]
            distance = EdgeType.Distances[i%3]
            return f"{base_name}, {distance}"

        for i in range(len(self.nodes)):
            node = self.nodes[i]
            print(f"Node: {i}, Category: {node.category_verbose}, Outgoing Edges to: ({') ('.join([', '.join([str(self.edges[adj][1]), self.nodes[self.edges[adj][1]].category_verbose, get_edge_type(self.edge_types[adj])]) for adj in node.outgoing])}), Incoming Edges from: ({') ('.join([', '.join([str(self.edges[adj][0]), self.nodes[self.edges[adj][0]].category_verbose, get_edge_type(self.edge_types[adj])]) for adj in node.incoming])})")

    def _debug_add_node(self, logits_add_node, cat_sample):
        temp = GraphNetConfig.temp_dict["an"]
        print("----------------------------------------------------")
        print("Current Graph Status:")
        print("Edges:")
        print(self.u_indices)
        print(self.v_indices)
        print(self.edges)
        print("Nodes:")
        self._print_current_state()
        _ = input()

        print("-------------------")
        print("Probability of a new node:")
        p_debug = list(F.softmax(logits_add_node*temp, dim=1)[0].cpu().data.numpy())
        logits_debug = list(logits_add_node[0].cpu().data.numpy())
        results = []
        for i in range(len(p_debug)):
            results.append((i, logits_debug[i], p_debug[i]))
        results = [results[-1]] + sorted(results[:-1], key=lambda x:-x[2])
        for (i, logit, p) in results:
            print(f"{i}, {GraphNetConfig.categories[i]}, {logit}, {p}")

        print(f"Choosing category {GraphNetConfig.categories[cat_sample]}")
        _ = input()

    def _debug_sym_class(self, logits_sym, sym_sample, structure_sample):
        sym_class_names = ["No", "RA", "FB", "LR", "R2", "R4", "C1", "C2"]
        temp = GraphNetConfig.temp_dict["sym"]
        print("-------------------")
        print("Probability of each symmetry class:")
        p_debug = list(F.softmax(logits_sym*temp, dim=1)[0].cpu().data.numpy())
        logits_debug = list(logits_sym[0].cpu().data.numpy())
        results = []
        for j in range(len(p_debug)):
            sym = j // 5
            structure = j % 5
            results.append((logits_debug[j], p_debug[j], sym, structure))

        results = [r for r in results if r[1] > 0.001]
        results = sorted(results, key=lambda x:-x[1])
        if len(results) > 10:
            results = results[:10]
        print("Most likely sym types:")
        for (logit, p, sym, structure) in results:
            hub, spoke, chain = self.structure_index_to_bools(structure)
            is_hub = "Hub" if hub else "   "
            is_spoke = "Spoke" if spoke else "     "
            is_chain = "Chain" if chain else "     "
            print(f"{sym_class_names[sym]}, {is_hub}, {is_spoke}, {is_chain}, {logit}, {p}")

        hub, spoke, chain = self.structure_index_to_bools(structure_sample)
        is_hub = "Hub" if hub else "   "
        is_spoke = "Spoke" if spoke else "     "
        is_chain = "Chain" if chain else "     "
        print(f"Choosing symmetry class {sym_class_names[sym_sample]}, {is_hub}, {is_spoke}, {is_chain}")
        _ = input()

    def _debug_add_edge(self, logits_add_edge, add_sample):
        temp = GraphNetConfig.temp_dict["ae"]
        if GraphNetConfig.predict_edge_type_first:
            NOEDGE = GraphNetConfig.num_edge_types
            print("-------------------")
            print("Pre Choosing edge type...")
            p_debug = list(F.softmax(logits_add_edge*temp, dim=1)[0].cpu().data.numpy())
            logits_debug = list(logits_add_edge[0].cpu().data.numpy())
            for i in range(len(p_debug)):
                base_name = EdgeType.BaseTypes[i//3]
                distance = EdgeType.Distances[i%3]
                if i == NOEDGE:
                    base_name = "no edge"
                    distance = "distant"
                print(f"Relation {base_name}, {distance}, {logits_debug[i]}, {p_debug[i]}")
            base_name = EdgeType.BaseTypes[add_sample//3]
            distance = EdgeType.Distances[add_sample%3]
            if add_sample == NOEDGE:
                base_name = "no edge"
                distance = "distant"
            print(f"Choosing edge type {base_name}, {distance}")
        else:
            print("-------------------")
            print("Probability of a new edge:")
            p_debug = list(torch.sigmoid(logits_add_edge*temp)[0].cpu().data.numpy())
            logits_debug = list(logits_add_edge[0].cpu().data.numpy())
            print(logits_debug, p_debug)
            print("Adding an Edge...")
        _ = input()

    def _debug_choose_node(self, logits_nodes, adj_sample, direction_sample, wall_mode):
        temp = GraphNetConfig.temp_dict["node"]
        print("-------------------")
        print("Choosing which node to connect to...")
        print("Graph Status Recap:")
        print("Nodes:")
        self._print_current_state()
        print("Probabily of connecting to nodes:")
        p_debug = list(F.softmax(logits_nodes*temp, dim=1)[0].cpu().data.numpy())
        logits_debug = list(logits_nodes[0].cpu().data.numpy())
        offset = self.num_walls if wall_mode==0 else 0
        for i in range(len(p_debug)):
            direction = "Incoming" if i%2==0 else "Outgoing"
            print(f"Node {i//2+offset}, {self.nodes[i//2+offset].category_verbose}, {direction}, {logits_debug[i]}, {p_debug[i]}")
        direction = "Incoming" if direction_sample==0 else "Outgoing"
        print(f"Choosing node {adj_sample}, {direction}")
        _ = input()

    def _debug_choose_node_and_type(self, logits_nodes, adj_sample, direction_sample, edge_type_sample, wall_mode, stop=False):
        temp = GraphNetConfig.temp_dict["node"]
        print("-------------------")
        print("Choosing which node to connect to...")
        print("Graph Status Recap:")
        print("Nodes:")
        self._print_current_state()
        print("Probabily of connecting to nodes:")
        p_debug = list(F.softmax(logits_nodes*temp, dim=1)[0].cpu().data.numpy())
        logits_debug = list(logits_nodes[0].cpu().data.numpy())
        offset = self.num_walls if wall_mode==0 else 0
        results = []
        for j in range(len(p_debug)):
            if GraphNetConfig.everything_together and j == len(p_debug) - 1:
                results.append((logits_debug[j], p_debug[j], "STOP", 0,0,0))
            else:
                i = j // GraphNetConfig.num_edge_types
                edge_type = j % GraphNetConfig.num_edge_types
                direction = "Incoming" if i%2==0 else "Outgoing"
                base_name = EdgeType.BaseTypes[edge_type//3]
                distance = EdgeType.Distances[edge_type%3]
                results.append((logits_debug[j], p_debug[j], i//2+offset, direction, base_name, distance))
        results = [r for r in results if r[1] > 0.001]
        results = sorted(results, key=lambda x:-x[1])
        if len(results) > 10:
            results = results[:10]
        print("Most likely edges:")
        for (logit, p, idx, direction, base_name, distance) in results:
            if idx == "STOP":
                print(f"Node s, stop_st, Stopping, stop, stopstop, {logit:.15f}, {p}")
            else:
                print(f"Node {idx}, {self.nodes[idx].category_verbose}, {direction}, {base_name}, {distance}, {logit}, {p}")
        direction = "Incoming" if direction_sample==0 else "Outgoing"
        
        if stop:
            print(f"STOP adding more edges")
        else:
            base_name = EdgeType.BaseTypes[edge_type_sample//3]
            distance = EdgeType.Distances[edge_type_sample%3]
            print(f"Choosing node {adj_sample}, {direction}")
            print(f"Choosing edge type {base_name}, {distance}")
        _ = input()

    def _debug_edge_type(self, logits_edge_type, edge_type_sample):
        temp = GraphNetConfig.temp_dict["et"]
        print("-------------------")
        print("Choosing edge type...")
        p_debug = list(F.softmax(logits_edge_type*temp, dim=1)[0].cpu().data.numpy())
        logits_debug = list(logits_edge_type[0].cpu().data.numpy())
        for i in range(len(p_debug)):
            base_name = EdgeType.BaseTypes[i//3]
            distance = EdgeType.Distances[i%3]
            print(f"Relation {base_name}, {distance}, {logits_debug[i]}, {p_debug[i]}")

        base_name = EdgeType.BaseTypes[edge_type_sample//3]
        distance = EdgeType.Distances[edge_type_sample%3]
        print(f"Choosing edge type {base_name}, {distance}")
        _ = input()

    def _debug_stop_adding_edge(self, logits_add_edge, wall_mode):
        temp = GraphNetConfig.temp_dict["ae"]
        if GraphNetConfig.predict_edge_type_first:
            print("-------------------")
            print("Pre Choosing edge type...")
            p_debug = list(F.softmax(logits_add_edge*temp, dim=1)[0].cpu().data.numpy())
            logits_debug = list(logits_add_edge[0].cpu().data.numpy())
            for i in range(len(p_debug)):
                NOEDGE = GraphNetConfig.num_edge_types
                base_name = EdgeType.BaseTypes[i//3]
                distance = EdgeType.Distances[i%3]
                if i == NOEDGE:
                    base_name = "no edge"
                    distance = "distant"
                print(f"Relation {base_name}, {distance}, {logits_debug[i]}, {p_debug[i]}")
        else:
            print("-------------------")
            print("Probability of a new edge:")
            p_debug = list(torch.sigmoid(logits_add_edge*temp)[0].cpu().data.numpy())
            logits_debug = list(logits_add_edge[0].cpu().data.numpy())
            print(logits_debug, p_debug)
        if wall_mode == 1:
            print("Stop adding wall edges...")
        else:
            print("Stop adding edges...")
        _ = input()
    
    def _debug_stop_adding_node(self, logits_add_node):
        temp = GraphNetConfig.temp_dict["an"]
        print("----------------------------------------------------")
        print("Current Graph Status:")
        print("Edges:")
        print(self.u_indices)
        print(self.v_indices)
        print(self.edges)
        print("Nodes:")
        self._print_current_state()
        _ = input()

        print("-------------------")
        print("Probability of a new node:")
        p_debug = list(F.softmax(logits_add_node*temp, dim=1)[0].cpu().data.numpy())
        logits_debug = list(logits_add_node[0].cpu().data.numpy())
        for i in range(len(p_debug)):
            print(f"{i}, {GraphNetConfig.categories[i]}, {logits_debug[i]}, {p_debug[i]}")

        print(f"Choosing to stop")
        _ = input()

    def _debug_start_adding_edges(self, wall_mode):
        print("-------------------")
        if wall_mode == 1:
            print("Adding Wall Edges...")
        else:
            print("Adding Edges...")
            
