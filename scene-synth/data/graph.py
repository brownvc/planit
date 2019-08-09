from bisect import *
from collections import Counter
from dataclasses import dataclass, field
from functools import cmp_to_key, reduce
import json
import math
import numpy as np
import torch
import os
import pickle
from queue import PriorityQueue
import random
import sys
from typing import Any

if __name__ == '__main__':
    # Allow python to see modules from the root directory
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(dir_path + '/..')

from data import ObjectCategories, Obj
from data.house import House, Wall
from math_utils.geometry_helpers import *
import utils

class RelationshipGraph:
    '''
    Class that represents a scene relationship graph
    '''
    category_map = ObjectCategories()

    # For some parts of the graph extraction pipeline, we use coarse categories.
    # However, there are some coarse categories that I don't think are quite right
    # For those, we still use fine categories.
    # This corrects for that
    bad_coarse_cats = set(['indoor_lamp', 'kitchen_cabinet', 'kitchen_appliance', 'kitchenware', 'music'])
    final_cats_for_bad_coarse_cats = []
    for cc in bad_coarse_cats:
        final_cats_for_bad_coarse_cats.extend(category_map.get_final_categories_for_coarse_category(cc))
    final_cats_for_bad_coarse_cats = set(final_cats_for_bad_coarse_cats)
    @classmethod
    def cat_final_to_coarse(cls, final_cat):
        if final_cat in cls.final_cats_for_bad_coarse_cats:
            return final_cat
        return cls.category_map.get_coarse_category_from_final_category(final_cat)
    @classmethod
    def cat_coarse_to_finals(cls, coarse_cat):
        if coarse_cat in cls.final_cats_for_bad_coarse_cats:
            return [coarse_cat]
        return cls.category_map.get_final_categories_for_coarse_category(coarse_cat)
    # Instead of going through the bad_coarse_cats pathway above, force retrieval of the coarse
    #    category
    # We also combine kitchen_cabinet and kitchen_appliance, b/c I think those should have the same
    #    coarse category.
    @classmethod
    def cat_final_to_coarse_force(cls, final_cat):
        catmap = cls.category_map
        coarse_cat = catmap.get_coarse_category_from_final_category(final_cat)
        if coarse_cat == 'kitchen_cabinet' or coarse_cat == 'kitchen_appliance':
            coarse_cat = 'kichen_cabinet_or_appliance'
        return coarse_cat

    # -----------------------------------------------------------------------------------

    ######## Hyperparameters ########
    # TODO: Tune these...

    # How 'visible' an object has to be from another object to consider them related
    min_occlusion_percentage = 0.3

    # How close two objects have to be to be considered 'adjacent'
    adj_rel_threshold = 0.05        # 5% of the larger bbox diag...
    adj_abs_threshold = 0.05        # ...or ~2 inches (whichever is smaller)

    # How close two objects have to be to be considered 'proximal'
    prox_rel_threshold = 0.1        # 10% of the larger bbox diag...
    prox_abs_threshold = 0.45       # ...or ~1.5 feet (whichever is larger)

    # Used by postprocessing to determine proximal thresholds from the data
    prox_postprocess_count_thresh = 5
    prox_postprocess_percent_thresh = 0.05

    # How often times must an edge type occur, relative to how often *both* of its
    #    endpoint categories occur together, for us not to filter it out?
    adjacent_filter_thresh = 0.03
    proximal_filter_thresh = 0.08
    distant_filter_thresh =  0.30

    # -----------------------------------------------------------------------------------

    class Node:
        '''
        Class that represents a single graph node
        '''
        def __init__(self, node_id, category_name, sym_types,\
                     out_edge_indices=[], in_edge_indices=[],\
                     wall_length=None, diag_length=None, pos=None,\
                     is_hub=False, is_spoke=False, is_chain=False,\
                     modelID=None, graph=None):
            self.id = node_id
            self.category_name = category_name
            self.sym_types = set(sym_types)
            self.__out_edge_indices = out_edge_indices
            self.__in_edge_indices = in_edge_indices
            self.wall_length = wall_length
            self.__diag_length = diag_length
            self.__pos = pos
            self.__is_hub = is_hub
            self.__is_spoke = is_spoke
            self.__is_chain = is_chain
            self.__modelID = modelID
            self.__graph = graph

        def __repr__(self):
            rep = f'{self.category_name} ({self.id}) : {self.sym_types}'
            if self.is_hub:
                rep += ' [HUB]'
            if self.is_spoke:
                rep += ' [SPOKE]'
            if self.is_chain:
                rep += ' [CHAIN]'
            return rep

        def clear_edges(self):
            self.__out_edge_indices = []
            self.__in_edge_indices = []
        def add_out_edge(self, edge_idx):
            self.__out_edge_indices.append(edge_idx)
        def add_in_edge(self, edge_idx):
            self.__in_edge_indices.append(edge_idx)

        def with_graph(self, graph):
            if self.__graph == graph:
                return self
            return RelationshipGraph.Node(self.id, self.category_name, self.sym_types, \
                self.__out_edge_indices, self.__in_edge_indices, self.wall_length, \
                self.diag_length, self.pos, self.is_hub, self.is_spoke, self.is_chain, self.modelID, graph)
        def without_graph(self):
            return RelationshipGraph.Node(self.id, self.category_name, self.sym_types, \
                self.__out_edge_indices, self.__in_edge_indices, self.wall_length, \
                self.diag_length, self.pos, self.is_hub, self.is_spoke, self.is_chain, self.modelID)

        @property
        def out_edges(self):
            return [self.__graph.edges[i] for i in self.__out_edge_indices]
        @property
        def in_edges(self):
            return [self.__graph.edges[i] for i in self.__in_edge_indices]
        @property
        def all_edges(self):
            return self.in_edges + self.out_edges

        @property
        def out_neighbors(self):
            return [e.end_node for e in self.out_edges]
        @property
        def in_neighbors(self):
            return [e.start_node for e in self.in_edges]
        @property
        def all_neighbors(self):
            return list(set(self.in_neighbors + self.out_neighbors))

        @property
        def is_arch_connected(self):
            return len([n for n in self.in_neighbors if n.is_arch]) > 0

        @property
        def is_wall(self):
            return self.category_name == 'wall'
        @property
        def is_window(self):
            return RelationshipGraph.category_map.is_window(self.category_name)
        @property
        def is_door(self):
            return RelationshipGraph.category_map.is_door(self.category_name)
        @property
        def is_arch(self):
            return self.is_wall or self.is_window or self.is_door
        @property
        def is_non_wall_arch(self):
            return self.is_window or self.is_door

        @property
        def is_second_tier(self):
            return len([e for e in self.in_edges if e.edge_type.is_support]) > 0

        @property
        def diag_length(self):
            if not hasattr(self, '_Node__diag_length'):
                setattr(self, '_Node__diag_length', None)
            return self.__diag_length
        def set_diag_length(self, diag):
            self.__diag_length = diag

        @property
        def pos(self):
            if not hasattr(self, '_Node__pos'):
                setattr(self, '_Node__pos', None)
            return self.__pos
        def set_pos(self, pos):
            self.__pos = pos

        @property
        def is_hub(self):
            if not hasattr(self, '_Node__is_hub'):
                setattr(self, '_Node__is_hub', False)
            return self.__is_hub
        def make_hub(self):
            self.__is_hub = True
        def make_not_hub(self):
            self.__is_hub = False

        @property
        def is_spoke(self):
            if not hasattr(self, '_Node__is_spoke'):
                setattr(self, '_Node__is_spoke', False)
            return self.__is_spoke
        def make_spoke(self):
            self.__is_spoke = True
        def make_not_spoke(self):
            self.__is_spoke = False

        @property
        def is_chain(self):
            if not hasattr(self, '_Node__is_chain'):
                setattr(self, '_Node__is_chain', False)
            return self.__is_chain
        def make_chain(self):
            self.__is_chain = True
        def make_not_chain(self):
            self.__is_chain = False

        @property
        def is_adjacent_chain(self):
            if not self.is_chain:
                return False
            for e in self.all_edges:
                if e.edge_type.is_adjacent and e.is_chain:
                    return True
            return False

        @property
        def is_chain_start(self):
            if not self.is_chain:
                return False
            chains = self.__graph.get_all_chains()
            for chain in chains:
                if chain[0].start_node.id == self.id:
                    return True
            return False

        @property
        def is_adjacent_chain_start(self):
            if not self.is_chain:
                return False
            chains = self.__graph.get_all_chains()
            for chain in chains:
                e = chain[0]
                if e.edge_type.is_adjacent and e.start_node.id == self.id:
                    return True
            return False

        @property
        def is_superstructure(self):
            return self.is_hub or self.is_spoke or self.is_chain

        @property
        def modelID(self):
            if not hasattr(self, '_Node__modelID'):
                setattr(self, '_Node__modelID', None)
            return self.__modelID
        def set_modelID(self, mid):
            self.__modelID = mid

        @property
        def is_not_symmetric(self):
            return RelationshipGraph.category_map.is_not_symmetric(self.sym_types)
        @property
        def is_radially_symmetric(self):
            return RelationshipGraph.category_map.is_radially_symmetric(self.sym_types)
        @property
        def is_front_back_reflect_symmetric(self):
            return RelationshipGraph.category_map.is_front_back_reflect_symmetric(self.sym_types)
        @property
        def is_left_right_reflect_symmetric(self):
            return RelationshipGraph.category_map.is_left_right_reflect_symmetric(self.sym_types)
        @property
        def is_two_way_rotate_symmetric(self):
            return RelationshipGraph.category_map.is_two_way_rotate_symmetric(self.sym_types)
        @property
        def is_four_way_rotate_symmetric(self):
            return RelationshipGraph.category_map.is_four_way_rotate_symmetric(self.sym_types)
        @property
        def is_corner_1_symmetric(self):
            return RelationshipGraph.category_map.is_corner_1_symmetric(self.sym_types)
        @property
        def is_corner_2_symmetric(self):
            return RelationshipGraph.category_map.is_corner_2_symmetric(self.sym_types)

        @property
        def has_unique_front(self):
            return self.is_not_symmetric or self.is_left_right_reflect_symmetric

        @property
        def sym_idx(self):
            check = [self.is_not_symmetric,
                     self.is_radially_symmetric,
                     self.is_front_back_reflect_symmetric,
                     self.is_left_right_reflect_symmetric,
                     self.is_two_way_rotate_symmetric,
                     self.is_four_way_rotate_symmetric,
                     self.is_corner_1_symmetric,
                     self.is_corner_2_symmetric]
            if sum(check) > 1:
                # print("Warning: more than 1 sym class")
                # print(self.sym_types)
                return 0
            if sum(check) == 0:
                return 0
            else:
                return check.index(True)

    # -----------------------------------------------------------------------------------

    class EdgeType:

        SUPPORT = 'support'

        class Distance:
            ADJACENT = 'adjacent'
            PROXIMAL = 'proximal'
            DISTANT = 'distant'

        BaseTypes = Direction.all_directions() + [SUPPORT]
        Distances = [Distance.DISTANT, Distance.PROXIMAL, Distance.ADJACENT]
        NumTypes = (len(Distances) * len(Direction.all_directions())) + 1

        def __init__(self, base_name, dist=None):
            self.base_name = base_name
            self.dist = dist

        def __eq__(self, other):
            return isinstance(other, RelationshipGraph.EdgeType) and self.__dict__ == other.__dict__

        def __hash__(self):
            return hash(self.name)

        @property
        def is_support(self):
            return self.base_name == RelationshipGraph.EdgeType.SUPPORT
        @property
        def is_spatial(self):
            return not self.is_support
        @property
        def direction(self):
            assert(self.is_spatial)
            return self.base_name
        @property
        def has_dist(self):
            return self.dist is not None and self.dist != 'None'
        @property
        def name(self):
            name = self.base_name
            if self.has_dist:
                name += ('_' + self.dist)
            return name
        @property
        def is_adjacent(self):
            return self.is_spatial and (self.dist == RelationshipGraph.EdgeType.Distance.ADJACENT)
        @property
        def is_proximal(self):
            return self.is_spatial and (self.dist == RelationshipGraph.EdgeType.Distance.PROXIMAL)
        @property
        def is_distant(self):
            return self.is_spatial and (self.dist == RelationshipGraph.EdgeType.Distance.DISTANT)
        @property
        def index(self):
            base_index = RelationshipGraph.EdgeType.BaseTypes.index(self.base_name)
            # Under this scheme, spatial edges without a distance get the same index
            #    as 'distant' edges (b/c 'distant') has index 0  
            dist_index = RelationshipGraph.EdgeType.Distances.index(self.dist) if self.has_dist else 0
            return base_index * len(RelationshipGraph.EdgeType.Distances) + dist_index

    '''
    Class that represents a single graph edge
    '''
    class Edge:

        def __init__(self, start_id, end_id, edge_type, dist=None, wall_ang=None, \
                anchor_percentage_visible=1.0, target_percentage_visible=1.0, \
                is_hub_edge=False, is_spoke=False, is_chain=False, graph=None):
            self.__start_id = start_id
            self.__end_id = end_id
            self.edge_type = edge_type
            self.dist = dist
            self.target_percentage_visible = target_percentage_visible
            self.anchor_percentage_visible = anchor_percentage_visible
            self.wall_ang = wall_ang
            self.__is_hub_edge = is_hub_edge
            self.__is_spoke = is_spoke
            self.__is_chain = is_chain
            self.__graph = graph

        def __repr__(self):
            edge_name = self.edge_type.name
            if not (self.dist is None):
                edge_name = f'{edge_name} [{self.dist:.2f}m]'
                # edge_name = f'{edge_name} [{self.dist}m]'
            if self.is_hub_edge:
                edge_name += ' [HUB]'
            if self.is_spoke:
                edge_name += ' [SPOKE]'
            if self.is_chain:
                edge_name += ' [CHAIN]'
            return f'{self.start_node.category_name} ({self.start_node.id}) [% {self.anchor_percentage_visible:.2f}] ---- {edge_name} ---> {self.end_node.category_name} ({self.end_node.id}) [% {self.target_percentage_visible:.2f}]'

        # I'm not overloading __eq__ because I still want to be able to compare and hash by reference
        # But sometimes this is a useful operation to have
        def is_equivalent_to(self, other):
            return \
                self.start_node.id == other.start_node.id and \
                self.end_node.id == other.end_node.id and \
                self.edge_type == other.edge_type

        def with_graph(self, graph):
            if self.__graph == graph:
                return self
            return RelationshipGraph.Edge(self.__start_id, self.__end_id, self.edge_type, self.dist, \
                self.wall_ang, self.anchor_percentage_visible, self.target_percentage_visible, \
                self.is_hub_edge, self.is_spoke, self.is_chain, graph)
        def without_graph(self):
            return RelationshipGraph.Edge(self.__start_id, self.__end_id, self.edge_type, self.dist, \
                self.wall_ang, self.anchor_percentage_visible, self.target_percentage_visible, \
                self.is_hub_edge, self.is_spoke, self.is_chain)
        
        def with_direction(self, d):
            assert(self.edge_type.is_spatial)
            edge_type = RelationshipGraph.EdgeType(d, self.edge_type.dist)
            return RelationshipGraph.Edge(self.__start_id, self.__end_id, edge_type, \
                self.dist, self.wall_ang, self.anchor_percentage_visible, self.target_percentage_visible, \
                self.is_hub_edge, self.is_spoke, self.is_chain, self.__graph)

        def with_distance(self, d):
            assert(self.edge_type.is_spatial)
            edge_type = RelationshipGraph.EdgeType(self.edge_type.direction, d)
            return RelationshipGraph.Edge(self.__start_id, self.__end_id, edge_type, \
                self.dist, self.wall_ang, self.anchor_percentage_visible, self.target_percentage_visible, \
                self.is_hub_edge, self.is_spoke, self.is_chain, self.__graph)

        def with_hub_endpoint(self, end_node):
            end_id = end_node.id
            return RelationshipGraph.Edge(self.__start_id, end_id, self.edge_type, \
                self.dist, self.wall_ang, self.anchor_percentage_visible, self.target_percentage_visible, \
                True, self.is_spoke, self.is_chain, self.__graph)

        def with_start_point(self, start_node):
            start_id = start_node.id
            return RelationshipGraph.Edge(start_id, self.__end_id, self.edge_type, \
                self.dist, self.wall_ang, self.anchor_percentage_visible, self.target_percentage_visible, \
                self.is_hub_edge, self.is_spoke, self.is_chain, self.__graph)

        def flipped(self):
            assert(self.edge_type.is_spatial)
            start_id = self.__end_id
            end_id = self.__start_id
            et = self.edge_type
            if et.direction == Direction.LEFT:
                direction = Direction.RIGHT
            elif et.direction == Direction.RIGHT:
                direction = Direction.LEFT
            elif et.direction == Direction.FRONT:
                direction = Direction.BACK
            elif et.direction == Direction.BACK:
                direction = Direction.FRONT
            new_et = RelationshipGraph.EdgeType(direction, et.dist)
            return RelationshipGraph.Edge(start_id, end_id, new_et, \
                self.dist, self.wall_ang, self.anchor_percentage_visible, self.target_percentage_visible, \
                self.is_hub_edge, self.is_spoke, self.is_chain, self.__graph)


        @property
        def start_node(self):
            assert(self.__graph is not None)
            return self.__graph.get_node_by_id(self.__start_id)
        @property
        def end_node(self):
            assert(self.__graph is not None)
            return self.__graph.get_node_by_id(self.__end_id)
        @property
        def neighbors(self):
            return self.start_node, self.end_node

        # Is this an inherited inbound edge of a hub?
        @property
        def is_hub_edge(self):
            if not hasattr(self, '_Edge__is_hub_edge'):
                setattr(self, '_Edge__is_hub_edge', False)
            return self.__is_hub_edge
        # Is this a spoke coming out of a hub
        @property
        def is_spoke(self):
            if not hasattr(self, '_Edge__is_spoke'):
                setattr(self, '_Edge__is_spoke', False)
            return self.__is_spoke
        def make_spoke(self):
            self.__is_spoke = True

        @property
        def is_chain(self):
            if not hasattr(self, '_Edge__is_chain'):
                setattr(self, '_Edge__is_chain', False)
            return self.__is_chain
        def make_chain(self):
            self.__is_chain = True
        def make_not_chain(self):
            self.__is_chain = False

        @property
        def min_visible_percent(self):
            return min(self.anchor_percentage_visible, self.target_percentage_visible)
        @property
        def max_visible_percent(self):
            return max(self.anchor_percentage_visible, self.target_percentage_visible)

        def sym_equivalences(self):
            return [edge for edge,_,_ in self.sym_equivalences_with_image_transforms()]

        # Get a list of all edges (including this one) that are symmetrically-equivalent to this one
        # Along the the edges, also return two functions that will transform an image and a point 
        #    according to the symmetry group that takes this edge to the symmetrically equivalent edge
        # (We don't worry about radial symmetry, since those only ever have FRONT edges, anyway)
        def sym_equivalences_with_image_transforms(self):

            # In CCW Front, Left, Back, Right order, get the other directions starting
            #    after 'direction'
            def get_dirs_ccw_from(direction):
                dirs = [Direction.FRONT, Direction.LEFT, Direction.BACK, Direction.RIGHT]
                idx = dirs.index(direction)
                return [dirs[(idx+i)%4] for i in range(1, 4)]

            eqs = set([(self, img_identity, pnt_identity)])

            if self.edge_type.is_spatial:
                n = self.start_node
                d = self.edge_type.direction
                if n.is_front_back_reflect_symmetric:
                    img_transform = img_reflect_vert
                    pnt_transform = pnt_reflect_vert
                    if d == Direction.FRONT:
                        eqs.add((self.with_direction(Direction.BACK), img_transform, pnt_transform))
                    elif d == Direction.BACK:
                        eqs.add((self.with_direction(Direction.FRONT), img_transform, pnt_transform))
                elif n.is_left_right_reflect_symmetric:
                    img_transform = img_reflect_horiz
                    pnt_transform = pnt_reflect_horiz
                    if d == Direction.LEFT:
                        eqs.add((self.with_direction(Direction.RIGHT), img_transform, pnt_transform))
                    elif d == Direction.RIGHT:
                        eqs.add((self.with_direction(Direction.LEFT), img_transform, pnt_transform))
                elif n.is_corner_1_symmetric:
                    img_transform = img_reflect_c1
                    pnt_transform = pnt_reflect_c1
                    if d == Direction.FRONT:
                        eqs.add((self.with_direction(Direction.LEFT), img_transform, pnt_transform))
                    elif d == Direction.LEFT:
                        eqs.add((self.with_direction(Direction.FRONT), img_transform, pnt_transform))
                    elif d == Direction.BACK:
                        eqs.add((self.with_direction(Direction.RIGHT), img_transform, pnt_transform))
                    elif d == Direction.RIGHT:
                        eqs.add((self.with_direction(Direction.BACK), img_transform, pnt_transform))
                elif n.is_corner_2_symmetric:
                    img_transform = img_reflect_c2
                    pnt_transform = pnt_reflect_c2
                    if d == Direction.FRONT:
                        eqs.add((self.with_direction(Direction.RIGHT), img_transform, pnt_transform))
                    elif d == Direction.RIGHT:
                        eqs.add((self.with_direction(Direction.FRONT), img_transform, pnt_transform))
                    elif d == Direction.BACK:
                        eqs.add((self.with_direction(Direction.LEFT), img_transform, pnt_transform))
                    elif d == Direction.LEFT:
                        eqs.add((self.with_direction(Direction.BACK), img_transform, pnt_transform))
                elif n.is_two_way_rotate_symmetric:
                    img_transform = img_rot_180
                    pnt_transform = pnt_rot_180
                    if d == Direction.FRONT:
                        eqs.add((self.with_direction(Direction.BACK), img_transform, pnt_transform))
                    elif d == Direction.BACK:
                        eqs.add((self.with_direction(Direction.FRONT), img_transform, pnt_transform))
                    elif d == Direction.LEFT:
                        eqs.add((self.with_direction(Direction.RIGHT), img_transform, pnt_transform))
                    elif d == Direction.RIGHT:
                        eqs.add((self.with_direction(Direction.LEFT), img_transform, pnt_transform))
                elif n.is_four_way_rotate_symmetric:
                    img_rots_ccw = [img_rot_90, img_rot_180, img_rot_270]
                    pnt_rots_ccw = [pnt_rot_90, pnt_rot_180, pnt_rot_270]
                    edges = [self.with_direction(d_) for d_ in get_dirs_ccw_from(d)]
                    eqs.update(zip(edges, img_rots_ccw, pnt_rots_ccw))
            return eqs

        

    # -----------------------------------------------------------------------------------
    # (Main body for RelationshipGraph)
    def __init__(self, nodes=[], edges=[], id_=None):
        self.id = id_
        self.__nodes = {n.id:n.with_graph(self) for n in nodes}
        self.edges = [e.with_graph(self) for e in edges]
        self.__record_node_edges()

    @property
    def nodes(self):
        return list(self.__nodes.values())

    def __repr__(self):
        ret = 'Nodes: ' + str(len(self.nodes)) + '\n'
        for node in self.nodes:
            ret += ('\t' + repr(node) + '\n')

        ret += 'Edges: ' + str(len(self.edges)) + '\n'
        for edge in self.edges:
            ret += ('\t' + repr(edge) + '\n')

        ret += 'Anchors:\n'
        for node in self.nodes:
            ret += ('\t' + str(len(node.out_edges)) + f': {node.category_name} ({node.id})\n')

        return ret

    def copy(self):
        nodes = [n.without_graph() for n in self.nodes]
        edges = [e.without_graph() for e in self.edges]
        return RelationshipGraph(nodes, edges, self.id)

    def has_node_with_id(self, id_):
        return id_ in self.__nodes
    def get_node_by_id(self, id_):
        if not id_ in self.__nodes:
            print(f'Could not find node with id {id_} in room {self.id}!')
        return self.__nodes[id_]
    def room_node_to_graph_node(self, room_node):
        return self.get_node_by_id(room_node.id)

    def add_node(self, node):
        self.__nodes[node.id] = node.with_graph(self)

    def add_edge(self, edge, ignore_duplicate=False):
        if ignore_duplicate:
            if self.contains_edge(edge):
                return
        else:
            assert(not self.contains_edge(edge))
        self.edges.append(edge.with_graph(self))
        self.__record_node_edges()

    def remove_node(self, node):
        # Remove all edges that refer to this node
        output_edges = []
        for edge in self.edges:
            if not node in edge.neighbors:
                output_edges.append(edge)
        self.edges = output_edges
        # Remove the node itself
        del self.__nodes[node.id]
        # Have to also update the node edge lists
        self.__record_node_edges()

    def contains_edge(self, edge):
        matches = [e for e in self.edges if e.is_equivalent_to(edge)]
        return len(matches) > 0

    def contains_opposite_edge(self, edge):
        return self.contains_edge_between(edge.end_node, edge.start_node)

    def contains_edge_between(self, n1, n2):
        matches = [e for e in self.edges if e.start_node.id == n1.id and e.end_node.id == n2.id]
        return len(matches) > 0

    def get_edge_between(self, n1, n2):
        matches = [e for e in self.edges if e.start_node.id == n1.id and e.end_node.id == n2.id]
        assert(len(matches) > 0)
        # If we fail on this next assert, then we know there's a duplicate edge...
        assert(len(matches) == 1)
        return matches[0]

    def remove_edge(self, edge):
        # Do this based on value equality, not reference equality
        matches = [e for e in self.edges if e.is_equivalent_to(edge)]
        assert(len(matches) > 0)
        # If we fail on this next assert, then we know there's a duplicate edge...
        assert(len(matches) == 1)
        self.edges.remove(matches[0])
        self.__record_node_edges()

    def union_with(self, other):
        assert isinstance(other, RelationshipGraph)
        # Add nodes from other
        for node in other.nodes:
            if not node.id in self.__nodes:
                self.add_node(node)
        # Add edges from other
        for edge in other.edges:
            if not self.contains_edge(edge):
                self.add_edge(edge)
        return self        

    # For debugging purposes
    def __get_duplicate_edges(self):
        dups = []
        graph = self.copy()
        while len(graph.edges) > 0:
            edge = graph.edges[0]
            del graph.edges[0]
            if graph.contains_edge(edge):
                dups.append(edge)
        return dups
    def __has_duplicate_edges(self):
        return len(self.__get_duplicate_edges()) > 0
    
    # Fills in the edge lists for each node object
    # Used as a helper when constructing a graph
    def __record_node_edges(self):
        for node in self.nodes:
            node.clear_edges()
        for edge_idx in range(len(self.edges)):
            edge = self.edges[edge_idx]
            edge.start_node.add_out_edge(edge_idx)
            edge.end_node.add_in_edge(edge_idx)

    @classmethod
    def dirname(cls, data_dir, data_root_dir=None):
        if not data_root_dir:
            data_root_dir = utils.get_data_root_dir()
        return f'{data_root_dir}/{data_dir}/graph'

    @classmethod
    def filename(cls, index, data_dir, data_root_dir=None):
        return f'{RelationshipGraph.dirname(data_dir, data_root_dir)}/{index}.pkl'

    def load(self, index, data_dir, data_root_dir=None):
        return self.load_from_file(RelationshipGraph.filename(index, data_dir, data_root_dir))

    def load_from_file(self, filename):
        fname = os.path.split(filename)[1]
        self.id = int(os.path.splitext(fname)[0])
        with open(filename, 'rb')as f:
            nodes, edges = pickle.load(f)
        self.__nodes = {n.id:n.with_graph(self) for n in nodes}
        self.edges = [e.with_graph(self) for e in edges]
        self.__record_node_edges()
        return self

    def save(self, index, data_dir, data_root_dir=None):
        return self.save_to_file(RelationshipGraph.filename(index, data_dir, data_root_dir))
    
    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            nodes = [n.without_graph() for n in self.nodes]
            edges = [e.without_graph() for e in self.edges]
            pickle.dump((nodes, edges), f, pickle.HIGHEST_PROTOCOL)

    # -----------------------------------------------------------------------------------

    # Extract subgraph where filter_node is a function to filter node, and filter_edge is a function to filter the edges
    def extract_subgraph(self, filter_node = lambda n: True, filter_edge = lambda e: True, trim_isolated_nodes = False):
        nodes = [n for n in self.nodes if filter_node(n)]
        edges = [e for e in self.edges if filter_edge(e)]
        if trim_isolated_nodes:
            node_ids = set()
            for e in edges:
                node_ids.add(e.start_node.id)
                node_ids.add(e.end_node.id)
            nodes = [n for n in nodes if n.id in node_ids]
        new_nodes = [n.without_graph() for n in nodes]
        new_edges = [e.without_graph() for e in edges]
        g = RelationshipGraph(new_nodes, new_edges)
        return g


    # Extract an (unpruned) graph from a scene
    def extract_raw_from_room(self, room, index):

        self.__nodes.clear()
        self.edges.clear()
        self.id = index

        # Do some preprocessing/filtering of the room itself, first
        remove_colocated_duplicates(room)
        guarantee_ccw_wall_loop(room, index)

        room_nodes = room.get_nodes()

        # First, build the list of nodes
        # Object nodes first
        for i, anchor_room_node in enumerate(room_nodes):
            anchor_id = anchor_room_node.id
            anchor_category = RelationshipGraph.category_map.get_final_category(anchor_room_node.modelId)
            anchor_sym_types = RelationshipGraph.category_map.get_symmetry_class(anchor_room_node.modelId)
            n = anchor_room_node
            anchor_pos = np.array([(n.xmin+n.xmax)/2, (n.ymin+n.ymax)/2])
            self.add_node(RelationshipGraph.Node(
                anchor_id, anchor_category, anchor_sym_types, \
                    modelID=anchor_room_node.modelId, pos=anchor_pos, graph=self
            ))
        # Then wall nodes
        for i, wall in enumerate(room.walls):
            anchor_id = wall.id
            anchor_category = 'wall'
            anchor_sym_types = []
            length = wall.length
            self.add_node(RelationshipGraph.Node(
                anchor_id, anchor_category, anchor_sym_types, wall_length=length, graph=self
            ))

        nodes_to_objs = {}
        for room_node in room_nodes:
            obj = Obj(room_node.modelId)
            nodes_to_objs[room_node] = obj
            self.room_node_to_graph_node(room_node).set_diag_length(obj_to_2d_diag_len(obj))
        for wall in room.walls:
            # Inverse-transform wall Obj into object space
            obj = Obj(wall=wall).transform(np.linalg.inv(wall.transform))
            nodes_to_objs[wall] = obj
            self.room_node_to_graph_node(wall).set_diag_length(obj_to_2d_diag_len(obj))

        # Extract wall -> wall adjacency relationships
        for wall in room.walls:
            wall_graph = self.room_node_to_graph_node(wall)
            for adj_wall_i in wall.adjacent:
                adj_wall = room.walls[adj_wall_i]
                adj_wall_graph = self.room_node_to_graph_node(adj_wall)
                # We've guaranteed that walls are ordered counterclockwise around the room,
                #    meaning that in a coordinate frame where up faces room-inward,
                #    the start point is left and the end point is right
                ## CASE: adj_wall is left-adjacent to wall
                if approx_colocated(adj_wall.pts[1], wall.pts[0]):
                    v0 = normalize(adj_wall.pts[0] - adj_wall.pts[1])
                    v1 = normalize(wall.pts[1] - wall.pts[0])
                    ang = np.dot(v0, v1)
                    edge_type = RelationshipGraph.EdgeType(Direction.LEFT, \
                        dist=RelationshipGraph.EdgeType.Distance.ADJACENT)
                    self.edges.append(RelationshipGraph.Edge(
                        wall_graph.id, adj_wall_graph.id, edge_type, wall_ang=ang, graph=self
                    ))
                ## CASE: adj_wall is right-adjacent to wall
                elif approx_colocated(adj_wall.pts[0], wall.pts[1]):
                    v0 = normalize(adj_wall.pts[1] - adj_wall.pts[0])
                    v1 = normalize(wall.pts[0] - wall.pts[1])
                    ang = np.dot(v0, v1)
                    edge_type = RelationshipGraph.EdgeType(Direction.RIGHT, \
                        dist=RelationshipGraph.EdgeType.Distance.ADJACENT)
                    self.edges.append(RelationshipGraph.Edge(
                        wall_graph.id, adj_wall_graph.id, edge_type, wall_ang=ang, graph=self
                    ))
                ## This should be impossible
                else:
                    raise("Impossible")

        # Extract wall -> wall across the room relationships
        for wall in room.walls:
            self._extract_edges_from_anchor(wall, nodes_to_objs, room, room.walls) 

        # Extract all of the "-> object" spatial relationships
        all_nodes = room_nodes + room.walls
        for i, anchor_room_node in enumerate(all_nodes):

            # Check for support edges
            if hasattr(anchor_room_node, 'parent'):
                if anchor_room_node.parent:
                    if not anchor_room_node.parent in ["Wall", "Floor", "Ceiling"]:
                        parent_id = anchor_room_node.parent.id
                        # Support edge goes parent --> child
                        self.edges.append(RelationshipGraph.Edge(
                            parent_id, anchor_room_node.id, \
                            RelationshipGraph.EdgeType(RelationshipGraph.EdgeType.SUPPORT), \
                            graph=self
                        ))

            # Continue to check for spatial edges
            self._extract_edges_from_anchor(anchor_room_node, nodes_to_objs, room, room_nodes)

        # Since we originate rays from near the edge of the anchor object, it might miss relationships
        #    with objects are mostly or fully contained within the bbox of the anchor.
        # Here, we detect additional ADJACENT relationships due to intersections
        self.__check_for_extra_intersect_adjacencies(room, nodes_to_objs)

        # The last step above might miss some wall -> window or wall -> door relationships, if there
        #    are e.g. windows/doors placed directly above one another. That's because these objects
        #    become co-located when projected into 2D, so one may occlude the other.
        # To account for this, we have an extra step here where we add ADJACENT edges from any wall
        #    to any arch feature whose bbox overlaps with it (if no such edge already exists)
        self.__check_for_extra_wall_arch_adjacencies(room, nodes_to_objs)

        # Prune edges (but back up the original graph in case we need to restore any of
        #    these edges to make the graph connected)
        orig_graph = self.copy()
        self.__prune_edges_initial()

        # If there are unreachable door/window nodes, check if it's because they're
        #    co-located with / intersecting some other door/window node (in 3D space)
        # If that's the case, then we should be able to safely remove them
        self.__remove_unreachable_intersecting_arch_features(room, nodes_to_objs)

        # Ensure connectivity
        if not self.__ensure_connected(orig_graph):
            print(f'Graph {index} had to be reconnected after initial pruning.')

        self.__sort_edges()
        self.__record_node_edges()

        return self

    def __check_for_extra_intersect_adjacencies(self, room, nodes_to_objs):
        self.__record_node_edges()
        room_nodes = room.get_nodes()
        for anchor_room_node in room_nodes:
            anchor_graph_node = self.room_node_to_graph_node(anchor_room_node)
            anchor_obj = nodes_to_objs[anchor_room_node]
            root_xmin, root_xmax = anchor_obj.xmin(), anchor_obj.xmax()
            root_ymin, root_ymax = anchor_obj.ymin(), anchor_obj.ymax()
            root_bbox = BBox2D.from_min_max([root_xmin, root_ymin], [root_xmax, root_ymax])
            anchor_transform = np.asarray(anchor_room_node.transform).reshape((4,4))
            anchor_inv_xform = np.linalg.inv(anchor_transform)
            for target_room_node in room_nodes:
                target_graph_node = self.room_node_to_graph_node(target_room_node)
                # Can't relate to yourself
                if anchor_room_node == target_room_node:
                    continue
                # Can't relate to an arch node
                if target_graph_node.is_arch:
                    continue
                # Skip if there's already an edge from anchor --> target
                if target_graph_node in anchor_graph_node.out_neighbors:
                    continue
                # Skip if anchor is supported by target (this looks like an intersection
                #    but it isn't, e.g. anchor:book --> target:desk)
                if len([e for e in anchor_graph_node.in_edges if e.start_node == target_graph_node and \
                    e.edge_type.is_support]) > 0:
                    continue
                # Else, check for intersection
                # Transform the target into the coordinate frame of the anchor, check
                #    for bbox overlaps
                target_transform = np.asarray(target_room_node.transform).reshape((4,4))
                relative_transform = np.matmul(target_transform, anchor_inv_xform)
                target_obj = nodes_to_objs[target_room_node]
                t_corners_3d = [np.dot(point, relative_transform) for point in \
                    [target_obj.front_left, target_obj.front_right, target_obj.back_right, target_obj.back_left]]
                t_corners = list(map(lambda p: np.array((p[0], p[2])), t_corners_3d))
                t_bbox2d = BBox2D(t_corners)
                if root_bbox.intersects(t_bbox2d):
                    # The intersection must be the majority of the target (this filters out some
                    #    incidental minor intersections caused by the expansion that bboxes
                    #    undergo when transformed)
                    if root_bbox.intersection(t_bbox2d).area < 0.9*t_bbox2d.area:
                        continue
                    # Add an adjacent edge from the anchor to the target
                    # Which direction do we pick? Well, if relation was missed by ray casting,
                    #    then the target bbox must be contained within the anchor bbox, such
                    #    that none of the rays starting from just inside the anchor bbox could
                    #    hit it.
                    # So, this is really a "contains" relationship, though that kind of conveys
                    #   the wrong semantics (i.e. there's no physical containment happening, just
                    #   an overly-large anchor bbox)
                    # What I do: construct four bboxes for the left half, right half, top half, 
                    #   and bottom half of the anchor bbox. If one of those has significantly
                    #   more overlap with the target bbox than the others, then we pick that
                    #   direction. Otherwise, we just pick FRONT
                    max_isect = 0
                    rb = root_bbox
                    boxs = [rb.left_half(), rb.right_half(), rb.back_half(), rb.front_half()]
                    dirs = [Direction.LEFT, Direction.RIGHT, Direction.BACK, Direction.FRONT]
                    areas = np.asarray([b.intersection(t_bbox2d).area for b in boxs])
                    min_area = areas.min()
                    max_area = areas.max()
                    if max_area/min_area > 1.5:
                        direc = dirs[np.argmax(areas)]
                    else:
                        direc = Direction.FRONT
                    et = RelationshipGraph.EdgeType(direc, RelationshipGraph.EdgeType.Distance.ADJACENT)
                    edge = RelationshipGraph.Edge(
                        anchor_room_node.id, target_room_node.id, et, dist=None, graph=self
                    )
                    self.add_edge(edge)
            

    def __check_for_extra_wall_arch_adjacencies(self, room, nodes_to_objs):
        room_nodes = room.get_nodes()
        for node in room_nodes:
            graph_node = self.room_node_to_graph_node(node)
            if graph_node.is_arch and not graph_node.is_wall:
                # Intersect its bbox with all the walls and see what we get
                # We do this by transforming the wall into the local coordinate frame
                #    of the object (to make its bbox axis aligned), then treating
                #    the wall as a ray and doing ray-box intersection
                root_bbox = self.__objspace_BBox2D(node, room, nodes_to_objs)
                transform = np.asarray(node.transform).reshape((4,4))
                inv_xform = np.linalg.inv(transform)
                for wall in room.walls:
                    p0, p1 = self.__transform_wall_seg(wall, inv_xform)
                    ray = Ray2D(p0, p1 - p0)
                    d = ray.distance_to_bbox(root_bbox)
                    max_d = np.linalg.norm(p1 - p0)     # Since Ray2D normalizes directions...
                    if d is not None and d >= 0 and d <= max_d:
                        # The wall segment intersects this window/door
                        graph_wall_node = self.room_node_to_graph_node(wall)
                        # Only add an edge if there isn't already one between these two nodes
                        matching_edges = [e for e in self.edges if \
                            e.start_node == graph_wall_node and e.end_node == node and \
                            e.edge_type.dist == RelationshipGraph.EdgeType.Distance.ADJACENT]
                        if len(matching_edges) == 0:
                            self.edges.append(RelationshipGraph.Edge(
                                graph_wall_node.id, graph_node.id, RelationshipGraph.EdgeType(Direction.FRONT, 
                                RelationshipGraph.EdgeType.Distance.ADJACENT), dist=0.0, graph=self))

    def __objspace_BBox2D(self, room_or_graph_node, room, nodes_to_objs):
        if isinstance(room_or_graph_node, RelationshipGraph.Node):
            room_node = [n for n in room.nodes if n.id == room_or_graph_node.id][0]
        else:
            room_node = room_or_graph_node
        obj = nodes_to_objs[room_node]
        root_xmin, root_xmax = obj.xmin(), obj.xmax()
        root_ymin, root_ymax = obj.ymin(), obj.ymax()
        return BBox2D.from_min_max([root_xmin, root_ymin], [root_xmax, root_ymax])

    def __transform_wall_seg(self, wall, xform):
        p0, p1 = np.ones(4), np.ones(4)
        p0[0:3] = wall.pts[0]
        p1[0:3] = wall.pts[1]
        p0 = np.dot(p0, xform)
        p1 = np.dot(p1, xform)
        p0 = np.array([p0[0], p0[2]])
        p1 = np.array([p1[0], p1[2]])
        return p0, p1

    def __remove_unreachable_intersecting_arch_features(self, room, nodes_to_objs):
        is_connected, disconnected_nodes = self.is_connected()
        if not is_connected:
            discon_arch_nodes = [n for n in disconnected_nodes if n.is_arch]
            if len(discon_arch_nodes) == 0:
                return
            arch_nodes = [n for n in self.nodes if n.is_arch and not n.is_wall]
            for node1 in discon_arch_nodes:
                mins1, maxs1 = self.__transformed_bbox(node1, room, nodes_to_objs)
                # Check for intersections
                for node2 in arch_nodes:
                    if node1 != node2:
                        mins2, maxs2 = self.__transformed_bbox(node2, room, nodes_to_objs)
                        no_overlap_checks = (mins1 > maxs2) | (mins2 > maxs1)
                        if not np.any(no_overlap_checks):
                            # That's an intersection; remove node1
                            self.remove_node(node1)

    def __transformed_bbox(self, graph_node, room, nodes_to_objs):
        room_node = [n for n in room.nodes if n.id == graph_node.id][0]
        obj = nodes_to_objs[room_node]
        transform = np.asarray(room_node.transform).reshape((4,4))
        corners = np.asarray([np.dot(point, transform) for point in \
            [obj.front_left, obj.front_right, obj.back_right, obj.back_left]])
        mins = np.min(corners, axis=0)
        maxs = np.max(corners, axis=0)
        return mins, maxs

    # If an (anchor, target) pair of nodes has multiple edges (i.e. both a left and a front edge),
    #    keep only the one that is least occluded.
    def __guarantee_one_edge_per_node_pair(self):
        output_edges = []
        for ancchor_n in self.nodes:
            for target_n in self.nodes:
                pair_edges = [e for e in self.edges if e.start_node.id == ancchor_n.id and e.end_node.id == target_n.id]
                if len(pair_edges) == 1:
                    output_edges += pair_edges
                if len(pair_edges) > 1:
                    best_edge = pair_edges[0]
                    for edge in pair_edges:
                        if edge.min_visible_percent >  best_edge.min_visible_percent:
                            best_edge = edge
                    output_edges.append(best_edge)
        self.edges = output_edges


    # Do whatever pruning we can having access to only this graph (and not the whole dataset)
    def __prune_edges_initial(self):

        # Any wall --> wall across room edges should be DISTANT (i.e. no PROXIMAL)
        for e in self.edges:
            if e.start_node.is_wall and e.end_node.is_wall and \
                e.edge_type.dist != RelationshipGraph.EdgeType.Distance.ADJACENT:
                e.edge_type.dist = RelationshipGraph.EdgeType.Distance.DISTANT

        # If an anchor/target pair have relations in more than 1 direction, choose one
        self.__guarantee_one_edge_per_node_pair()

        # If an edge can see very little of either object
        self.edges = [e for e in self.edges \
            if e.max_visible_percent > RelationshipGraph.min_occlusion_percentage] 

        # After pruning all of these edges: we may be left with arch nodes (doors, specifically)
        #    that have no edges to them. This is because they aren't really part of this
        #    room, but they have a sliver of overlap with one or more of the walls, so they got
        #    included. In this case, we delete those nodes entirely (they just clutter the graph
        #    and would make it be disconnected)
        self.__remove_useless_arch_nodes()

        self.__record_node_edges()

    # Delete arch nodes that have no ingoing edges
    def __remove_useless_arch_nodes(self):
        self.__record_node_edges()
        nodes = self.nodes[:]
        for node in nodes:
            if node.is_arch and len(node.in_edges) == 0:
                self.remove_node(node)

    # If the graph is disconnected, make it connected
    # orig_graph is self, prior to a bunch of edges being removed
    def __ensure_connected(self, orig_graph, force_connect_absolutely_unreachables=False):
        self.__record_node_edges()
        orig_graph.__record_node_edges()
        self.__sort_edges()
        was_disconnected = False
        absolutely_unreachable_nodes = []
        is_connected, unreachable_nodes = self.is_connected()
        while not is_connected:
            was_disconnected = True

            # We look for a path from the walls to one of the unreachable nodes
            # We want the 'lowest cost' path, where cost is:
            #  * We prefer paths that don't create cycles to those that do
            #  * Otherwise, the cost of a path is the sum of its edge costs (see edge_cost below)
            best_path, best_path_creates_cyc = None, True
            for unode in unreachable_nodes:
                best_path_for_unode, cyc = self.__find_best_valid_path_from_walls_to(unode, orig_graph)
                # If there are no valid paths, then this node is 'absolutely unreachable'
                if not best_path_for_unode:
                    # If force_connect is False, then this is a dealbreaker and we die with an error
                    if not force_connect_absolutely_unreachables:
                        print('*********************************************************************')
                        print(f'Room {self.id}: Could not find a valid path to connect {unode}')
                        print('*********************************************************************')
                        print('Orig graph:')
                        print(orig_graph)
                        print('*********************************************************************')
                        self.assert_is_connected('Current graph:')
                    # Otherwise, we'll deal with it later, so move on to the next unreachable node
                    else:
                        continue
                # If we have no best path so far, then of course we take this one
                if best_path is None:
                    best_path = best_path_for_unode
                    best_path_creates_cyc = cyc
                # If the best path so far induces a cycle, then any lower-cost path will do
                elif best_path_creates_cyc:
                    if best_path_for_unode.cost < best_path.cost:
                        best_path = best_path_for_unode
                        best_path_creates_cyc = cyc
                # Otherwise, the path must be both lower-cost *and* also not induce a cycle
                else:
                    if not cyc and best_path_for_unode.cost < best_path.cost:
                        best_path = best_path_for_unode

            if best_path:
                # If we found a path, add the best overall path we found into the graph
                self.add_path(best_path.edges)
                # Re-evaluate the graph's connectivity
                is_connected, unreachable_nodes = self.is_connected()
            else:
                # All the reamining nodes in unreachable_nodes are 'absolutely unreachable'
                absolutely_unreachable_nodes.extend(unreachable_nodes)
                # We reconnect these nodes by adding distant nodes from every wall
                wall_nodes = [n for n in self.nodes if n.is_wall]
                # Well, not *every* wall. Every wall that has a straight line to the node
                #    along the wall's front-facing direction.
                # To do this, we find the shortest path in the original graph from each wall
                #    to the node. Here, "shortest" is simply the sum of lengths of the edges.
                #    If no such path exists, then there is no straight line from the wall to
                #    the node, so we don't connect it to that wall.
                is_connected = False
                while not is_connected:
                    # In most cases, the problem will be solved by finding the node(s) that
                    #    have zero inbound edges and connecting that. So we try those nodes first.
                    # I suppose it's theoretically possible that we could have a disconnected
                    #    cycle, though, so we have to allow for that possibility.
                    self.__record_node_edges()
                    unreachable_nodes.sort(key=lambda node: len(node.in_edges))
                    # Pick the first (lowest-in-degree) node, find the paths that connect it
                    #    to the walls
                    unode = unreachable_nodes[0]
                    found_some_wall_path = False
                    for wnode in wall_nodes:
                        unode_orig = orig_graph.get_node_by_id(unode.id)
                        wnode_orig = orig_graph.get_node_by_id(wnode.id)
                        path_cost = orig_graph.__shortest_path_cost(wnode_orig, unode_orig)
                        if math.isfinite(path_cost):
                            found_some_wall_path = True
                            edge_type = RelationshipGraph.EdgeType(Direction.FRONT, \
                                RelationshipGraph.EdgeType.Distance.DISTANT)
                            self.add_edge(RelationshipGraph.Edge(
                                wnode.id, unode.id, edge_type, dist=None, graph=self
                            ))
                    # If there's no path from any wall to this node: that should be literally
                    #    impossible, but let's have a check, just in case
                    if not found_some_wall_path:
                        print('*********************************************************************')
                        print(f'Room {self.id}: Could not find *any* path to connect {unode}')
                        print('*********************************************************************')
                        print('Orig graph:')
                        print(orig_graph)
                        print('*********************************************************************')
                        self.assert_is_connected('Current graph:') 
                    # Recheck for connectivity and repeat
                    is_connected, unreachable_nodes = self.is_connected()
                
                # We're done here; break out of the outer while loop and head to the return block below
                break
                    
            
        if was_disconnected:
            self.__record_node_edges()
            self.make_chains()    # Re-label any chains that might've gotten flipped
            self.__sort_edges()
        
        if force_connect_absolutely_unreachables:
            return not was_disconnected, absolutely_unreachable_nodes
        else:
            return not was_disconnected

    # Returns the path, plus a boolean indicating whether that path induces a cycle in self
    def __find_best_valid_path_from_walls_to(self, node, orig_graph):
        
        def edge_cost(edge):

            # If either of these nodes is missing from self, then Infinity (can't use this edge)
            if not self.has_node_with_id(edge.start_node.id) or \
                not self.has_node_with_id(edge.end_node.id):
                return math.inf

            start_node_in_self = self.get_node_by_id(edge.start_node.id)
            end_node_in_self = self.get_node_by_id(edge.end_node.id)

            edge_matches = [e for e in start_node_in_self.out_edges if e.end_node == end_node_in_self]
            edge_in_self = edge_matches[0] if len(edge_matches) > 0 else None
            opp_matches = [e for e in start_node_in_self.in_edges if e.start_node == start_node_in_self]
            opposite_edge_in_self = opp_matches[0] if len(opp_matches) > 0 else None

            # 0 if this edge is in self (we like using edges that are already in the graph!)
            if edge_in_self:
                return 0

            ##### The rest of these cases are edges that are not in the graph: we'd need to
            #####    swap out some edge that is in the graph for this one, in order to use it

            # Infinity if this edge's opposite edge is in self and is a spoke edge
            # (We don't allow this process to violate a hub-spoke group)
            if opposite_edge_in_self and opposite_edge_in_self.is_spoke:
                return math.inf
            # # We also don't allow this process to violate a chain
            ### NOTE: This turns out to be a bad idea, b/c you want to be able to reverse entire chains
            ###   sometimes to fix connectivity issues
            # if opposite_edge_in_self and opposite_edge_in_self.is_chain:
            #     return math.inf

            # If the edge goes from an object to a significantly bigger object, then Infinity
            # The reasoning here is that functionally speaking, it makes more sense for big objects
            #    to be the 'parents' of small objects
            # It is always OK for an edge to start from a wall or window (i.e. they are considered
            #    "infinitely big")
            # It is also always OK for an edge to go from two instances of the same category (as in
            #    a chain), no matter their relative size
            SIZE_FACTOR = 0.8
            cat1 = edge.start_node.category_name
            cat2 = edge.end_node.category_name
            cls = RelationshipGraph
            coarse_cat_1 = cls.cat_final_to_coarse_force(cat1)
            coarse_cat_2 = cls.cat_final_to_coarse_force(cat2)
            if not (edge.start_node.is_wall or edge.start_node.is_window) and \
                not (coarse_cat_1 == coarse_cat_2) and \
                (edge.start_node.diag_length < SIZE_FACTOR*edge.end_node.diag_length):
                return math.inf

            # 1 if this edge is support
            if edge.edge_type.is_support:
                return 1
            # 1 if this edge is adjacent
            if edge.edge_type.is_adjacent:
                return 1
            # 2 if this edge is proximal
            if edge.edge_type.is_proximal:
                return 2
            # 100 if this edge is distant (we'd really prefer not to use these edges unless
            #    absolutely necessary) + the distance of the edge (we prefer the shortest
            #    possible DISTANT edge)
            if edge.edge_type.is_distant:
                return 100 + edge.dist
            # I think this ought to cover all possible cases
            else:
                raise Exception('This should be impossible')

        def edge_cost_adjonly(edge):
            raw_cost = edge_cost(edge)
            if not math.isfinite(raw_cost):
                return math.inf
            if edge.edge_type.is_spatial and not edge.edge_type.is_adjacent:
                return math.inf
            else:
                return raw_cost

        def edge_cost_adjprox(edge):
            raw_cost = edge_cost(edge)
            if not math.isfinite(raw_cost):
                return math.inf
            if edge.edge_type.is_spatial and edge.edge_type.is_distant:
                return math.inf
            else:
                return raw_cost

        node_orig = orig_graph.get_node_by_id(node.id)

        # First, find valid paths that use only adjacent edges
        # We sort by cost, and we try to find one that doesn't induce a cycle
        # The reason we search for restricted paths like this is for optimization purposes:
        #    it drastically cuts down on the combinatorial search space, so if we can find
        #    such a path, we save ourselves the time of having to search for *all* possible paths
        adjonly_paths = orig_graph.__all_valid_paths_from_walls_to(node_orig, edge_cost_adjonly)
        adjonly_paths = [p for p in adjonly_paths if p.make_chain_consistent_with(self, orig_graph)]
        adjonly_paths.sort(key=lambda path: path.cost)
        for path in adjonly_paths:
            if not path.induces_cycle_in(self):
                return path, False
        # If we hit this point, then all the adjonly_paths induce a cycle
        # So next, let's look for valid paths that are also allowed to use
        #   proximal edges
        adjprox_paths = orig_graph.__all_valid_paths_from_walls_to(node_orig, edge_cost_adjprox)
        adjprox_paths = [p for p in adjprox_paths if p.make_chain_consistent_with(self, orig_graph)]
        adjprox_paths.sort(key=lambda path: path.cost)
        for path in adjprox_paths:
            if not path.induces_cycle_in(self):
                return path, False
        # If we hit this point, then all the adjprox_paths induce a cycle
        # So, let's look for *any* valid path that doesn't induce a cycle
        all_paths = orig_graph.__all_valid_paths_from_walls_to(node_orig, edge_cost)
        all_paths = [p for p in all_paths if p.make_chain_consistent_with(self, orig_graph)]
        all_paths.sort(key=lambda path: path.cost)
        for path in all_paths:
            if not path.induces_cycle_in(self):
                return path, False
        # If we hit this point, then all possible paths induce a cycle
        # So, let's just pick the lowest-cost one
        if len(all_paths) > 0:
            return all_paths[0], True
        # But, we might have an absolutely unreachable node, in which
        #    case we have no choice but to return None
        return None, False

    # Find all the paths from the walls to a target node, such that the path has a
    #   non-infinite cost according to 'edge_cost'
    def __all_valid_paths_from_walls_to(self, node, edge_cost):

        class Path(object):
            def __init__(self, edges):
                self.edges = edges
                self.cost = sum([edge_cost(e) for e in edges])
            def contains_node(self, node):
                for e in self.edges:
                    if e.start_node == node or e.end_node == node:
                        return True
                return False
            def extend(self, edge):
                newpath = Path(self.edges + [edge])
                newpath.cost += edge_cost(edge)
                return newpath
            def induces_cycle_in(self, graph):
                scratch_graph = graph.copy()
                scratch_graph.add_path(self.edges)
                return scratch_graph.has_cycle()
            def make_chain_consistent_with(self, graph, orig_graph):
                # If this path contains any edge between two nodes that are part of a chain
                #    in 'graph,' then it must be *consistent* with that chain: either it
                #    leaves the entire chain the same, or it flips all the edges.

                def edgetuple(e):
                    return (e.start_node.id, e.end_node.id)
                def edgetuple_reversed(e):
                    return (e.start_node.id, e.end_node.id)
                
                ### First, check if this path contains any edges that are chain edges
                ###    in graph (or whose opposite is).
                chain_edges = []
                reverse_chain_edges = []
                for e in self.edges:
                    if graph.contains_edge(e):
                        ge = graph.get_edge_between(e.start_node, e.end_node)
                        if ge.is_chain:
                            chain_edges.append(e)
                    elif graph.contains_opposite_edge(e):
                        ge = graph.get_edge_between(e.end_node, e.start_node)
                        if ge.is_chain:
                            reverse_chain_edges.append(e)
                ### If not, we can return True right away
                if len(chain_edges) == 0 and len(reverse_chain_edges) == 0:
                    return True

                ### Otherwise, we've got to check for chain consistency
                all_chains = graph.get_all_chains()
                edge2chainidx = {}
                for i, chain in enumerate(all_chains):
                    for e in chain:
                        edge2chainidx[edgetuple(e)] = i
                if len(chain_edges) > 0:
                    ### Find every chain that that this path overlaps with
                    chains_involved = [all_chains[i] for i in set([edge2chainidx[edgetuple(e)] \
                        for e in chain_edges])]
                    ### To be consistent, the combination of this path with 'graph' must contain all of these
                    ###    edges, and must contain none of their opposites.
                    ### We already know that 'graph' contains all of these edges (they are its chains)
                    ### All that remains is to check that this path contains none of their opposites
                    all_chain_edges = reduce(lambda c1,c2: c1+c2, chains_involved)
                    opposite_edges = set([(e.end_node.id, e.start_node.id) for e in all_chain_edges])
                    for e in self.edges:
                        if (e.start_node.id, e.end_node.id) in opposite_edges:
                            return False

                ### Also check for reverse chain consistency
                if len(reverse_chain_edges) > 0:
                    ### Find every chain that the reverse of some edge in this path overlaps with
                    chains_involved = [all_chains[i] for i in set([edge2chainidx[edgetuple_reversed(e)] \
                        for e in chain_edges])]
                    ### Verify that it is possible to reverse all of these chains (i.e. the reverse of 
                    ###    every edge in the chain exists in the original graph)
                    ### If there is some overlapped chain for which this is not true, then this path is
                    ###    not chain consistent and we can return False
                    for chain in chains_involved:
                        for e in chain:
                            if not orig_graph.contains_edge_between(e.end_node, e.start_node):
                                return False
                    ### OK, so it's possible to reverse all the overlapped chains.
                    ### Next, verify that this path doesn't require *both* forward and reverse
                    ###    edges from the same chain
                    path_edges = set([edgetuple(e) for e in self.edges])
                    for chain in chains_involved:
                        fwd = False
                        rvs = False
                        for e in chain:
                            if edgetuple(e) in path_edges:
                                fwd = True
                            elif edgetuple_reversed(e) in path_edges:
                                rvs = True
                        if fwd and rvs:
                            return False
                    ### OK, so this path only requires edges in the reverse direction of all overlapped
                    ###    chains.
                    ### Finally, we need to check that the path includes *all* reversed edges of all
                    ###    overlapped chains (i.e. that it completely flips the chain)
                    ### If it doesn't, we add the edges that it's missing
                    for chain in chains_involved:
                        for e in chain:
                            if not edgetuple_reversed(e) in path_edges:
                                e_rev = orig_graph.get_edge_between(e.end_node, e.start_node)
                                self.extend(e_rev)

                return True

        paths = []
        wall_nodes = [n for n in self.nodes if n.is_wall]
        for wn in wall_nodes:
            # Initialize the queue with all length-1 paths consisting of all the edges
            #    coming out of the wall node (except not the ones that connect to another
            #    wall node; that's useless)
            partial_paths = [Path([e]) for e in wn.out_edges if not e.end_node.is_wall]
            while len(partial_paths) > 0:
                path = partial_paths.pop()
                last_edge = path.edges[-1]
                last_node = last_edge.end_node
                # If we've reached the target node, then we're done tracing this path
                if last_node == node:
                    if math.isfinite(path.cost):
                        paths.append(path)
                # Otherwise, we need to keep pursuing all possible extensions of this path
                for e in last_node.out_edges:
                    # Don't follow any cyclic paths (i.e. paths with repeating nodes)
                    next_node = e.end_node
                    if path.contains_node(next_node):
                        continue
                    next_path = path.extend(e)
                    # Also, don't follow any paths that have developed an infinite cost
                    if not math.isfinite(next_path.cost):
                        continue
                    # Otherwise, push this new path onto the queue and continue
                    partial_paths.append(next_path)
        
        return paths

    def __shortest_path_cost(self, start_node, end_node):
        # Initialize all nodes to have distance infinity
        node2dist = {n:math.inf for n in self.nodes}
        node2dist[start_node] = 0
         # List of visited nodes
        visited = set([])

        # Init prio queue
        q = PriorityQueue()

        # How we add/remove nodes from the queue
        # https://docs.python.org/3/library/queue.html#queue.PriorityQueue
        @dataclass(order=True)
        class PrioritizedItem:
            priority: int
            item: Any=field(compare=False)
        def add_node_to_queue(node):
            q.put(PrioritizedItem(node2dist[node], node))
        def get_node_from_queue():
            return q.get().item

        def get_edge_connecting(u, v):
            for e in u.out_edges:
                if e.end_node == v:
                    return e
            for e in u.in_edges:
                if e.start_node == v:
                    return e

        # Run Djikstra
        add_node_to_queue(start_node)
        while not q.empty():
            u = get_node_from_queue()
            visited.add(u)
            for v in u.all_neighbors:
                if v not in visited:
                    e = get_edge_connecting(u, v)
                    edge_cost = (e.dist if e.dist else 0)
                    new_dist = node2dist[u] + edge_cost
                    if new_dist < node2dist[v]:
                        node2dist[v] = new_dist
                    # We stop at end_node
                    if v != end_node:
                        add_node_to_queue(v)

        return node2dist[end_node]


    # Add a path (a list of edges) to the graph
    # It re-uses any edges from the path that are already in the graph
    # If the opposite of any path edge occurs in the graph, that is removed
    def add_path(self, path):
        for edge in path:
            if not self.contains_edge(edge):
                if self.contains_edge_between(edge.end_node, edge.start_node):
                    self.remove_edge(self.get_edge_between(edge.end_node, edge.start_node))
                self.add_edge(edge)

    # Deletes all the doors and windows
    # If this disconnects the graph, have the adjacent wall inherit any necessary
    #    outbound edges
    def remove_non_wall_arch_nodes(self):
        self.__record_node_edges()
        arch_nodes = [n for n in self.nodes if n.is_non_wall_arch]
        # Record the possible outbound edges that walls could inherit
        possible_wall_edges = []
        for n in arch_nodes:
            for ie in n.in_edges:
                # if not ie.start_node.is_wall:
                #     print(n)
                #     print(ie)
                #     print(ie.start_node)
                assert(ie.start_node.is_wall)
                wall_node = ie.start_node
                for oe in n.out_edges:
                    # Only valid if the graph doesn't already a edge betwen the wall
                    #    and this end node
                    if not self.contains_edge_between(wall_node, oe.end_node):
                        possible_wall_edges.append(oe.with_start_point(wall_node))
        # Remove the nodes (these also removes any edges connected to them)
        for n in arch_nodes:
            self.remove_node(n)
        # If the graph is disconnected, add the shortest possible wall edge
        #    until the graph becomes connected
        possible_wall_edges.sort(key=lambda e: e.dist if e.dist is not None else 0)
        is_connected, disconnected_nodes = self.is_connected()
        while not is_connected:
            best_edge, best_num = None, 0
            for e in possible_wall_edges:
                # No point in checking an edge that we already have
                if self.contains_edge(e):
                    continue
                self.add_edge(e)
                is_conn, disconn_nodes = self.is_connected()
                num_disconn = len(disconn_nodes) if disconn_nodes else 0
                # Kee track of which edge reduces the number of disconnected
                #    nodes the most (greedy)
                num = len(disconnected_nodes) - num_disconn
                if num > best_num:
                    best_num = num
                    best_edge = e
                self.remove_edge(e)
                # If this edge connects the graph, then great, we're done -- no need
                #    to look at any other possible edges in our list
                if is_conn:
                    break
            # We must have found some edge that connects at least one node
            if not(best_edge):
                return len(arch_nodes) > 0
            assert(best_edge)
            # Add that to the graph and remove it from the list of potential edges
            self.add_edge(best_edge)
            possible_wall_edges.remove(best_edge)
            # Update our loop variable
            is_connected, disconnected_nodes = self.is_connected()
        self.__sort_edges()
        # Return true if any nodes were removed
        return len(arch_nodes) > 0
        
    def get_all_chains(self):
        chains = []
        chain_edges = [e for e in self.edges if e.is_chain]
        for e in chain_edges:
            # Look for a chain that this edge extends
            found_chain = False
            for chain in chains:
                if chain[-1].end_node.id == e.start_node.id:
                    found_chain = True
                    chain.append(e)
                    break
                elif chain[0].start_node.id == e.end_node.id:
                    found_chain = True
                    chain.insert(0, e)
                    break
            # If we don't find one, then start a new chain with this edge
            if not found_chain:
                chains.append([e])
        # The above process may result in fragmented chains.
        # Try to merge chains together.
        no_chains_to_merge = False
        while not no_chains_to_merge:
            indexpairs = [(i, j) for i in range(0, len(chains)-1) for j in range(i+1, len(chains))]
            for i,j in indexpairs:
                chain1 = chains[i]
                chain2 = chains[j]
                if chain1[-1].end_node.id == chain2[0].start_node.id:
                    chain1.extend(chain2)
                    del chains[j]
                    break
                elif chain2[-1].end_node.id == chain1[0].start_node.id:
                    chain2.extend(chain1)
                    del chains[i]
                    break 
            else:
                no_chains_to_merge = True
        return chains

    def get_chains_starting_from(self, node):
        assert node.is_chain
        chains = self.get_all_chains()
        out_chains = []
        for chain in chains:
            if chain[0].start_node == node:
                out_chains.append(chain)
        return out_chains

    def __sort_edges(self):

        def assert_same_edge_sets(edges1, edges2, msg):
            if not (len(edges1) == len(edges2) and set(edges1) == set(edges2)):
                es1 = set(edges1)
                es2 = set(edges2)
                diff12 = es1 - es2
                diff21 = es2 - es1
                print('')
                print(self)
                print(f'{msg}: Inconsistency in __sort_edges!!!')
                if len(diff12) > 0:
                    print('Edges in edges1 that are not in edges2:')
                    for e in diff12:
                        print('   ', e)
                if len(diff21) > 0:
                    print('Edges in edges2 that are not in edges1:')
                    for e in diff21:
                        print('   ', e)
                assert False

        # First, sort by start node id
        self.edges.sort(key=lambda edge: edge.start_node.id)

        # Separate the different types of edges
        object_edges = [e for e in self.edges if not e.start_node.is_arch]
        arch_edges = [e for e in self.edges if e.start_node.is_arch and not e.start_node.is_wall]
        wall_edges = [e for e in self.edges if e.start_node.is_wall and not e.end_node.is_wall]
        wall_wall_adj_edges = [e for e in self.edges if e.start_node.is_wall and e.end_node.is_wall \
            and e.edge_type.is_adjacent]
        wall_wall_across_edges = [e for e in self.edges if e.start_node.is_wall and e.end_node.is_wall \
            and e.edge_type.is_distant]

        # Within the object edges, chains come first
        chains = self.get_all_chains()
        chains.sort(key=lambda chain: chain[0].start_node.id)
        chain_edges = []
        for chain in chains:
            chain_edges.extend(chain)
        non_chain_edges = [e for e in object_edges if not e.is_chain]
        object_edges = chain_edges + non_chain_edges
        
        # Final ordering goes:
        #  * object -> object
        #  * arch -> object
        #  * wall -> object
        #  * wall -> wall adjacent
        #  * wall -> wall opposite
        all_edges = object_edges + arch_edges + wall_edges + wall_wall_adj_edges + wall_wall_across_edges
        # Verify that these are all the edges
        assert_same_edge_sets(self.edges, all_edges, 'all edges')

        self.edges = all_edges
        self.__record_node_edges()

        def keyfn(e):
            if not e.start_node.is_wall:
                return 0
            if not e.end_node.is_wall:
                return 1
            if e.edge_type.dist == RelationshipGraph.EdgeType.Distance.ADJACENT:
                return 2
            else:
                return 3
        self.edges.sort(key=keyfn)

        # Within the object-object edges: find chains and sort them in order
        


    def _extract_edges_from_anchor(self, anchor_room_node, nodes_to_objs, room, possible_targets):

        DEBUG_ID = None
        # DEBUG_ID = '1_16'

        anchor_graph_node = self.room_node_to_graph_node(anchor_room_node)
        anchor_transform = np.asarray(anchor_room_node.transform).reshape((4,4))
        anchor_obj = nodes_to_objs[anchor_room_node]

        root_xmin, root_xmax = anchor_obj.xmin(), anchor_obj.xmax()
        root_ymin, root_ymax = anchor_obj.ymin(), anchor_obj.ymax()
        root_bbox = BBox2D.from_min_max([root_xmin, root_ymin], [root_xmax, root_ymax])

        def _valid_target_candidate(room_node):
            graph_node = self.room_node_to_graph_node(room_node)

            # (1) Cannot relate to yourself
            if room_node == anchor_room_node:
                return False

            # (2) Cannot relate to an architectural feature, unless the anchor is a wall
            if not anchor_graph_node.is_wall and graph_node.is_arch:
                return False

            # (3) Must have the same parent
            if hasattr(anchor_room_node, 'parent') and anchor_room_node.parent and \
               hasattr(room_node, 'parent') and room_node.parent:
                if anchor_room_node.parent in ["Wall", "Floor", "Ceiling"]:
                    if room_node.parent not in ["Wall", "Floor", "Ceiling"]:
                        return False # A top level object can only relate to top level objects
                else:
                    if anchor_room_node.parent != room_node.parent:
                        return False # Non top level objects need the exact same parent
            # This includes if the anchor node is a wall and the other object is second-tier
            if hasattr(room_node, 'parent') and room_node.parent:
                if anchor_graph_node.is_wall and room_node.parent not in ["Wall", "Floor", "Ceiling"]:
                    return False

            # (4) If both anchor and target are walls, there must not already exist
            #    an edge between them
            if anchor_graph_node.is_wall and graph_node.is_wall:
                id1, id2 = anchor_graph_node.id, graph_node.id
                if len([e for e in self.edges if e.start_node.id == id1 and e.end_node.id == id2]) > 0:
                    return False


            return True

        def _valid_directions_for_anchor(anchor_graph_node):
            if anchor_graph_node.is_arch:
                # It's kind of weird to include both FRONT and BACK here, but it's needed because
                #    e.g. some door subcategories can functionally face both ways...
                return [d for d in Direction.all_directions() if not Direction.is_horizontal(d)]
            else:
                return Direction.all_directions()

        target_room_nodes = [node for node in possible_targets if _valid_target_candidate(node)]

        # if anchor_room_node.id == 'wall_1':
        #     target_room_nodes = [n for n in target_room_nodes if n.id != '0_69']

        # If the anchor is radially symmetric, then in theory we should check rays in an infinite
        #    number of directions. Since that's not possible in practice, we pick several of them.
        # The way we do is is by rotating the anchor transform slightly each time.
        # We just need to cover a 90 degree increment with a few steps, to make sure we don't miss
        #    anything.
        anchor_transforms = [anchor_transform]
        if anchor_graph_node.is_radially_symmetric:
            n_steps = 4
            rots = [math.pi/2 * ((i+1)/n_steps) for i in range(n_steps)]
            for ang in rots:
                cos = math.cos(ang)
                sin = math.sin(ang)
                rot_mat = np.array([[cos,  0, sin, 0],
                                    [0,    1, 0,   0],
                                    [-sin, 0, cos, 0],
                                    [0,    0, 0,   1]])
                rot_mat = np.transpose(rot_mat)
                a_xform = np.dot(rot_mat, anchor_transform)
                anchor_transforms.append(a_xform)

        # Check for relationships using every one of these anchor transforms
        for anc_xform in anchor_transforms:

            # Collect bboxes of target in objspace of anchor
            anchor_inv_xform = np.linalg.inv(anc_xform)
            targets_to_bboxes = {}

            for target_room_node in target_room_nodes:
                target_transform = np.asarray(target_room_node.transform).reshape((4,4))
                relative_transform = np.matmul(target_transform, anchor_inv_xform)

                target_obj = nodes_to_objs[target_room_node]
                t_corners_3d = [np.dot(point, relative_transform) for point in [target_obj.front_left, target_obj.front_right, target_obj.back_right, target_obj.back_left]]
                t_corners = list(map(lambda p: np.array((p[0], p[2])), t_corners_3d))
                t_bbox2d = BBox2D(t_corners)

                targets_to_bboxes[target_room_node] = t_bbox2d

            def _get_range_along_edge(target, direction):
                '''
                Given target bbox (in object-space of anchor)
                Returns range (min, max) along the edge of the anchor
                '''
                t_range = Range() # initialize as empty range
                target_bbox = targets_to_bboxes[target]

                for corner in target_bbox.corners:
                    # TODO improve this
                    if Direction.is_horizontal(direction):
                        candidate_point = corner[1]
                    else:
                        candidate_point = corner[0]

                    t_range.update(candidate_point)

                return t_range

            for direction in _valid_directions_for_anchor(anchor_graph_node):
                if DEBUG_ID and anchor_room_node.id == DEBUG_ID:
                    print('DIRECTION:', direction)
                # TODO improve this
                if Direction.is_horizontal(direction):
                    edge_range = Range(root_ymin, root_ymax)
                else: 
                    edge_range = Range(root_xmin, root_xmax)

                # target_node -> range (min, max)
                targets_to_ranges = {}
                targets_to_projected_length = {}
                for target_room_node in target_room_nodes:
                    t_range = _get_range_along_edge(target_room_node, direction)
                    # if DEBUG_ID and anchor_room_node.id == DEBUG_ID:
                    #     print('-----------------------')
                    #     print(direction)
                    #     print(edge_range)
                    #     print(t_range)
                    projected_length = t_range.length()
                    t_range.clip(edge_range)
                    # if DEBUG_ID and anchor_room_node.id == DEBUG_ID:
                    #     print(t_range)
                    if t_range.is_valid():
                        targets_to_projected_length[target_room_node] = projected_length
                        targets_to_ranges[target_room_node] = t_range

                split_points_set = set()
                for target, t_range in targets_to_ranges.items():
                    split_points_set.add(t_range.min)
                    split_points_set.add(t_range.max)
                
                split_points = sorted(split_points_set) # convert to list

                # Collect partitions which are range (min, max) between two 
                # subsequent split points along the edge
                partitions = []
                for i in range(len(split_points) - 1):
                    partitions.append(Range(split_points[i], split_points[i+1]))

                found_targets = {}

                for partition in partitions:
                    if DEBUG_ID and anchor_room_node.id == DEBUG_ID:
                        print('  PARTITION:', partition, 'length:', partition.length())
                    epsilon = 0.001
                    if Direction.is_horizontal(direction):
                        anchor_length = anchor_obj.xmax() - anchor_obj.xmin()
                        ray_offset = min(RelationshipGraph.prox_rel_threshold*anchor_length, \
                            RelationshipGraph.prox_abs_threshold)

                        if direction == Direction.LEFT:
                            x = anchor_obj.xmin() + ray_offset
                        else: # RIGHT
                            x = anchor_obj.xmax() - ray_offset

                        min_point, max_point = [x, partition.min + epsilon], [x, partition.max - epsilon]
                    else:
                        anchor_length = anchor_obj.ymax() - anchor_obj.ymin()
                        ray_offset = min(RelationshipGraph.prox_rel_threshold*anchor_length, \
                            RelationshipGraph.prox_abs_threshold)

                        if direction == Direction.FRONT:
                            y = anchor_obj.ymax() - ray_offset
                        else: # BACK
                            y = anchor_obj.ymin() + ray_offset
                        min_point, max_point = [partition.min + epsilon, y], [partition.max - epsilon, y]

                    direction_vector = Direction.to_vector(direction)
                    partition_rays = [Ray2D(min_point, direction_vector), Ray2D(max_point, direction_vector)]

                    candidate_targets = [t for t in targets_to_ranges if partition.overlaps(targets_to_ranges[t])]

                    closest_target = (None, None)

                    for target_room_node in candidate_targets:
                        target_bbox = targets_to_bboxes[target_room_node]
                        dist_to_target = target_bbox.distance_from_edge(partition_rays)

                        if DEBUG_ID and anchor_room_node.id == DEBUG_ID:
                            print('    ', self.room_node_to_graph_node(target_room_node))
                            print('    ', dist_to_target)

                        if dist_to_target is None or dist_to_target < 0:
                            continue

                        # If both this and the closest target have distance 0 (i.e. are intersecting),
                        #    then prefer the one that has more bbox overlap with the anchor in the
                        #    current direction
                        if closest_target[0] is not None and closest_target[0] == 0 and \
                            dist_to_target == 0:
                            closest_target_bbox = targets_to_bboxes[closest_target[1]]
                            closest_isect_box = root_bbox.intersection(closest_target_bbox)
                            target_isect_box = root_bbox.intersection(target_bbox)
                            if Direction.is_horizontal(direction):
                                if target_isect_box.width > closest_isect_box.width:
                                    closest_target = (dist_to_target, target_room_node)
                            else:
                                if target_isect_box.height > closest_isect_box.height:
                                    closest_target = (dist_to_target, target_room_node)
                        else:
                            closest_target = get_min(closest_target, dist_to_target, target_room_node)

                    if closest_target[0] is not None:
                        partition_length = partition.length()
                        distance = closest_target[0] - ray_offset
                        target_room_node = closest_target[1]

                        if target_room_node in found_targets:
                            found_targets[target_room_node].append((distance, partition_length))
                        else:
                            found_targets[target_room_node] = [(distance, partition_length)]

                if DEBUG_ID and anchor_room_node.id == DEBUG_ID:
                    print('  FOUND TARGETS')
                    for target_room_node,parts in found_targets.items():
                        print('    ', self.room_node_to_graph_node(target_room_node))
                        for distance, partition_length in parts:
                            print('      ', distance, partition_length)

                for target_room_node, partitions_found in found_targets.items():
                    min_distance = min(partitions_found, key=lambda x: abs(x[0]))[0] # Absolute value of distance
                    # Distance is in objectspace, not worldspace

                    # Lengths
                    occluded_length = sum(x[1] for x in partitions_found) # Sum lengths of partitions
                    target_projected_length = targets_to_projected_length[target_room_node]
                    anchor_length = edge_range.length()

                    target_percentage_visible = occluded_length / float(target_projected_length)
                    anchor_percentage_visible = occluded_length / float(anchor_length)

                    if target_percentage_visible > 1.1 or anchor_percentage_visible > 1.1:
                        print('!!! impossible visibility !!!')
                        print('anchor:', anchor_graph_node)
                        print('target:', self.room_node_to_graph_node(target_room_node))
                        print('anchor percent visible:', anchor_percentage_visible)
                        print('target percent visible:', target_percentage_visible)
                        exit()

                    # Compute the direction label for this edge
                    edge_dir = direction
                    # If the anchor object is radially symmetric, then all directions are FRONT
                    if anchor_graph_node.is_radially_symmetric:
                        edge_dir = Direction.FRONT
                    # If the anchor object is an architectural feature, then we should consider this
                    #   FRONT (even if it might technically be BACK due to local coord system quirks)
                    elif anchor_graph_node.is_arch:
                        edge_dir = Direction.FRONT
                    # If the min distance is <= 0, then there's an intersection. The direction here
                    #    can't always be trusted, so we compute it by looking for what side of the
                    #    anchor the target is protruding out of the most
                    elif min_distance <= 0:
                        anchor_obj = nodes_to_objs[anchor_room_node]
                        root_xmin, root_xmax = anchor_obj.xmin(), anchor_obj.xmax()
                        root_ymin, root_ymax = anchor_obj.ymin(), anchor_obj.ymax()
                        root_bbox = BBox2D.from_min_max([root_xmin, root_ymin], [root_xmax, root_ymax])
                        anchor_transform = np.asarray(anchor_room_node.transform).reshape((4,4))
                        anchor_inv_xform = np.linalg.inv(anchor_transform)
                        target_transform = np.asarray(target_room_node.transform).reshape((4,4))
                        relative_transform = np.matmul(target_transform, anchor_inv_xform)
                        target_obj = nodes_to_objs[target_room_node]
                        t_corners_3d = [np.dot(point, relative_transform) for point in \
                            [target_obj.front_left, target_obj.front_right, target_obj.back_right, target_obj.back_left]]
                        t_corners = list(map(lambda p: np.array((p[0], p[2])), t_corners_3d))
                        t_bbox2d = BBox2D(t_corners)
                        left_protrusion = root_bbox.mins[0] - t_bbox2d.mins[0]  # positive if target pokes out left
                        right_protrusion = t_bbox2d.maxs[0] - root_bbox.maxs[0] # positive if target pokes out right
                        back_protrusion = root_bbox.mins[1] - t_bbox2d.mins[1]  # positive if target pokes out back
                        front_protrusion = t_bbox2d.maxs[1] - root_bbox.maxs[1] # positive if target pokes out front
                        protrusions = [left_protrusion, right_protrusion, back_protrusion, front_protrusion]
                        dirs = [Direction.LEFT, Direction.RIGHT, Direction.BACK, Direction.FRONT]
                        edge_dir = dirs[np.argmax(protrusions)]

                    # Compute the distance label for this edge
                    anchor_diag = diag_length(anchor_room_node, nodes_to_objs)
                    target_diag = diag_length(target_room_node, nodes_to_objs)
                    max_diag = max(anchor_diag, target_diag)
                    adj_thresh = min(RelationshipGraph.adj_rel_threshold*max_diag,\
                        RelationshipGraph.adj_abs_threshold)
                    prox_thresh = max(RelationshipGraph.prox_rel_threshold*max_diag,\
                        RelationshipGraph.prox_abs_threshold)

                    ##### Don't need this case anymore, b/c we have more explicit handling
                    #####    for intersections now
                    # # CASE: bboxes intersect. This is adjacent, and we make sure the min_distance
                    # #    value is zero
                    # target_bbox = targets_to_bboxes[target_room_node]
                    # if root_bbox.intersects(target_bbox):
                    #     min_distance = 0
                    #     edge_dist = RelationshipGraph.EdgeType.Distance.ADJACENT

                    # CASE: adjacent (but not intersecting)
                    if min_distance <= adj_thresh:
                        edge_dist = RelationshipGraph.EdgeType.Distance.ADJACENT
                    # CASE: proximal
                    elif min_distance <= prox_thresh:
                        edge_dist = RelationshipGraph.EdgeType.Distance.PROXIMAL
                    # Everything else is 'distant'
                    else:
                        edge_dist = RelationshipGraph.EdgeType.Distance.DISTANT
                    
                    edge_type = RelationshipGraph.EdgeType(edge_dir, dist=edge_dist)

                    anchor_id = anchor_graph_node.id
                    target_id = target_room_node.id
                    self.edges.append(RelationshipGraph.Edge(
                        anchor_id, target_id, edge_type, dist=min_distance, \
                        anchor_percentage_visible=anchor_percentage_visible, \
                        target_percentage_visible=target_percentage_visible, \
                        graph=self
                    ))
                    if DEBUG_ID and anchor_room_node.id == DEBUG_ID:
                        print('  ADDING EDGE:', self.edges[-1])

# -----------------------------------------------------------------------------------

    # Post-process a whole directory of generated graphs
    @classmethod
    def postprocess(cls, data_root_dir, data_folder, from_orig_dir=False):
        # This proceeds in two phases.
        # The first phase does most things, including a data-driven way of
        #    finding a split between proximal and distant
        # Then, in the second phase (which computes stats based on this new
        #    proximal/distant split), we filter out 'unimportant' relationships

        dirname = RelationshipGraph.dirname(data_folder, data_root_dir)

        ###### We print to terminal *and* save this output to a file
        logfile = open(f'{dirname}/log.txt', 'w')
        stdout = sys.stdout
        sys.stdout = utils.Forwarder(sys.stdout, logfile)

        # Helper
        def print_connectivity_stats(num_disconnected, num_total):
            dis_freq = num_disconnected / num_total
            print(f'Num disconnected: {num_disconnected}/{num_total} ({100.0 * dis_freq:.2f}%)')

        print('========== Graph postprocessing! ==========')

        orig_dirname = f'{dirname}/../graph_orig'

        # First, load up all the graphs
        print('Loading graphs...')
        graphs = []
        load_dir = orig_dirname if from_orig_dir else dirname
        for fname in os.listdir(load_dir):
            if fname.endswith('.pkl'):
                filename = load_dir + '/' + fname
                graph = RelationshipGraph().load_from_file(filename)
                graphs.append(graph)
                sys.stdout.write(f' Loaded graph {fname}...                 \r')
        # Sort by numeric ID
        graphs.sort(key=lambda graph: graph.id)

        # Save a copy of the graphs we started with, just in case
        if not from_orig_dir:
            print('Saving copies of starting graphs...')
            utils.ensuredir(orig_dirname)
            for graph in graphs:
                sys.stdout.write(f' Saving graph {graph.id}...                 \r')
                fname = f'{orig_dirname}/{graph.id}.pkl'
                graph.save_to_file(fname)

        # Initialize postprocessing coroutines for all graphs
        coroutines = [graph.start_postprocess_coroutine() for graph in graphs]

        orig_graphs = []

        # ---------------------- Phase 0 ----------------------
        print('========== Fix errors from initial graph extraction ==========')
        num_total = 0
        num_disconnected = 0
        num_edges_2obj = 0
        num_edges_arch2obj = 0
        for i, graph in enumerate(graphs):
            num_total += 1
            sys.stdout.write(f' Processing graph {graph.id}...                 \r')
            orig_graph = next(coroutines[i])
            orig_graphs.append(orig_graph)
            num_edges_2obj += len([e for e in graph.edges if not e.end_node.is_arch])
            num_edges_arch2obj += len([e for e in graph.edges if \
                e.start_node.is_arch and not e.end_node.is_arch])
        print_connectivity_stats(num_disconnected, num_total)
        print('Percent of --> obj edges which are arch --> obj:', num_edges_arch2obj/num_edges_2obj)

        # ---------------------- Phase 1 ----------------------
        print('========== Graph postprocess pass 1 ==========')

        # Gather statistics (for everything except the edge importance filter)
        stats = RelationshipGraph.__gather_post_statistics(data_root_dir, data_folder, graphs, 1)

        # Process every generated graph
        print('Processing graphs...')
        num_total = 0
        num_disconnected = 0
        for i, graph in enumerate(graphs):
            num_total += 1
            sys.stdout.write(f' Processing graph {graph.id}...                 \r')

            # # Save pre-superstructure-extraction graphs for Angel's figures
            # if (data_folder == 'living_graph_6x6' and graph.id in [256, 257]) or \
            #     (data_folder == 'kitchen_graph_6x6' and graph.id in [231, 235, 243, 244]):
            #     pregraph = graph.copy()
            #     pregraph.__break_single_edge_cycles(stats)
            #     pregraph.__postprocess_phase1_part2(stats)
            #     pregraph.__ensure_connected(orig_graphs[i])
            #     utils.ensuredir(f'{data_folder}_nosuperstructs')
            #     pregraph.save_to_file(f'{data_folder}_nosuperstructs/{graph.id}.pkl')

            coroutines[i].send(stats)

        print_connectivity_stats(num_disconnected, num_total)
        print('Num hubs found:', stats["num_hubs"])
        print('Num chains found:', stats["num_chains"])
        print('Chain length hist:', stats["chain_lengths"])

        # ---------------------- Phase 2 ----------------------
        print('========== Graph postprocess pass 2 ==========')

        # Gather statistics (for just the edge importance filter)
        stats = RelationshipGraph.__gather_post_statistics(data_root_dir, data_folder, graphs, 2)
        absolute_unreachables = []
        num_cycles = 0
        cycles_found = []

        # Process every generated graph
        print('Processing graphs...')
        num_total = 0
        num_disconnected = 0
        for i, graph in enumerate(graphs):
            num_total += 1
            sys.stdout.write(f' Processing graph {graph.id}...                 \r')
            was_connected, absolutely_unreachable_nodes, cycles = coroutines[i].send(stats)
            if not was_connected:
                num_disconnected += 1
                if len(absolutely_unreachable_nodes) > 0:
                    absolute_unreachables.append((graph.id, absolutely_unreachable_nodes))
            if len(cycles) > 0:
                num_cycles += len(cycles)
                for cycle in cycles:
                    cycles_found.append((graph.id, cycle))
        print_connectivity_stats(num_disconnected, num_total)

        print(f'Absolutely unreachable nodes: {len(absolute_unreachables)}')
        for room,aunodes in absolute_unreachables:
            print('-----------------------------------------')
            print('In room:', room)
            for n in aunodes:
                print('   ', n)

        print(f'Cycles found: {num_cycles}')
        for room,cycle in cycles_found:
            print('-----------------------------------------')
            print('In room:', room)
            for edge in cycle:
                print('   ', edge)

        # ---------------------- Final checks ----------------------
        print('========== Final checks before saving ==========')
        print('Verify that remove_non_wall_arch_nodes works for all graphs...')
        for i, graph in enumerate(graphs):
            sys.stdout.write(f' Checking graph {graph.id}...                 \r')
            graph.copy().remove_non_wall_arch_nodes()
        print('\nFinal graph stats:')
        num_nodes = 0
        num_non_wall_nodes = 0
        num_non_arch_nodes = 0
        num_edges = 0
        num_non_wall_edges = 0
        num_non_arch_edges = 0
        num_non_wall_wall_edges = 0
        num_non_across_room_wall_edges = 0
        for i, graph in enumerate(graphs):
            num_nodes += len(graph.nodes)
            num_non_wall_nodes += len([n for n in graph.nodes if not n.is_wall])
            num_non_arch_nodes += len([n for n in graph.nodes if not n.is_arch])
            num_edges += len(graph.edges)
            num_non_wall_edges += len([e for e in graph.edges if not e.start_node.is_wall])
            num_non_arch_edges += len([e for e in graph.edges if not e.start_node.is_arch])
            num_non_wall_wall_edges += len([e for e in graph.edges if \
                not (e.start_node.is_wall and e.end_node.is_wall)])
            num_non_across_room_wall_edges += len([e for e in graph.edges if \
                not (e.start_node.is_wall and e.end_node.is_wall and e.edge_type.is_distant)])
        n = len(graphs)
        print('Average num nodes per graph:           ', num_nodes/n)
        print('Average num nodes per graph (non-wall):', num_non_wall_nodes/n)
        print('Average num nodes per graph (non-arch):', num_non_arch_nodes/n)
        print('Average num edges per graph:           ', num_edges/n)
        print('Average num edges per graph (non-wall):', num_non_wall_edges/n)
        print('Average num edges per graph (non-arch):', num_non_arch_edges/n)
        print('Average num edges per graph (non-wall-wall):', num_non_wall_wall_edges/n)
        print('Average num edges per graph (non-wall-wall-across):', num_non_across_room_wall_edges/n)

        # ---------------------- Final Saving ----------------------
        print('========== Saving final graphs ==========')
        for i, graph in enumerate(graphs):
            sys.stdout.write(f' Saving graph {graph.id}...                 \r')
            orig_graph = orig_graphs[i]
            filename = f'{dirname}/{graph.id}.pkl'
            orig_filename = f'{dirname}/{graph.id}.pkl.orig'
            graph.save_to_file(filename)
            orig_graph.save_to_file(orig_filename)

        ###### Restore the original standard out
        sys.stdout = stdout

    # This returns a coroutine that yields control back after every stage of postprocessing
    def start_postprocess_coroutine(self):
        # Phase 0
        self.__postprocess_phase0()
        orig_graph = self.copy()

        # Phase 1
        stats = (yield orig_graph)
        self.__postprocess_phase1_part1(stats)
        orig_graph.union_with(self)
        self.__postprocess_phase1_part2(stats)

        # Phase 2
        stats = (yield)
        self.__postprocess_phase2(stats)
        # Ensure connectivity
        was_connected, absolutely_unreachable_nodes = self.__ensure_connected(orig_graph, \
            force_connect_absolutely_unreachables=True)
        # # Very last step: get rid of any unnecessary inbound edges to chain nodes
        # self.__remove_extraneous_chain_edges()
        # Check for cycles
        cycles = self.find_cycles()
        yield (was_connected, absolutely_unreachable_nodes, cycles)
        
    # Just postprocess this one graph, using saved stats loaded from disk
    def postprocess_one(self, data_folder, data_root_dir=None, verbose=False):
        dirname = RelationshipGraph.dirname(data_folder, data_root_dir)
        with open(dirname+'/stats_phase1.json', 'r') as f:
            stats_phase1 = json.load(f)
        with open(dirname+'/stats_phase2.json', 'r') as f:
            stats_phase2 = json.load(f)
        coroutine = self.start_postprocess_coroutine()
        # Do phase 0
        next(coroutine)
        if verbose:
            print('****************************************************************************')
            print('**************************** ORIGINAL GRAPH ********************************')
            print('****************************************************************************')
            print(self)
        # Do phase 1
        coroutine.send(stats_phase1)
        if verbose:
            print('****************************************************************************')
            print('**************************** AFTER PHASE 1 *********************************')
            print('****************************************************************************')
            print(self)
        # Do phase 2
        _, absolutely_unreachable_nodes, cycles = coroutine.send(stats_phase2)
        if verbose:
            print('****************************************************************************')
            print('**************************** AFTER PHASE 2 *********************************')
            print('****************************************************************************')
            print(self)
        if len(absolutely_unreachable_nodes) > 0:
            print('Phase 2 - Absolutely unreachable nodes:')
            for n in absolutely_unreachable_nodes:
                print('  ', n)
        if len(cycles) > 0:
            print('Phase 2 - Cycles')
            for cyc in cycles:
                print('  Cycle:')
                for e in cyc:
                    print('    ', e)

    
    @classmethod
    def __gather_post_statistics(cls, data_root_dir, data_folder, graphs, phase, save=True):
        print('Gathering statistics...')

        stats = {}
        
        category_names = RelationshipGraph.category_map.all_non_arch_categories(
            data_root_dir, data_folder
        )
        archcat_names = RelationshipGraph.category_map.all_arch_categories()

        # Number of occurrences of each category (inc. arch)
        print(' Category occurrences...')
        category_counts = RelationshipGraph.category_map.all_non_arch_category_counts(
            data_root_dir, data_folder
        )
        archcat_counts = RelationshipGraph.category_map.all_arch_category_counts(
            data_root_dir, data_folder
        )
        cat_count_dict = {category_names[i]:count for i, count in enumerate(category_counts)}
        archcat_count_dict = {archcat_names[i]:count for i, count in enumerate(archcat_counts)}
        stats["category_occurrences"] = {**cat_count_dict, **archcat_count_dict}

        # Size and 'importance' of each category
        if phase == 1:
            print(' Category sizes/importances...')
            category_sizes = RelationshipGraph.category_map.all_non_arch_category_sizes(
                data_root_dir, data_folder
            )
            stats["category_sizes"] = {category_names[i]:s for i, s in enumerate(category_sizes)}
            category_importances = RelationshipGraph.category_map.all_non_arch_category_importances(
                data_root_dir, data_folder
            )
            stats["category_importances"] = {category_names[i]:imp for i, imp in enumerate(category_importances)}

        if phase == 2:
            # Number of co-occurrences of (cat1,cat2) (inc. arch)
            stats["cat_co-occurrences"] = Counter()
            stats["coarse_cat_co-occurrences"] = Counter()
            # Number of occurrences of each (cat1,cat2,edge_type) (inc. arch)
            stats["edge_occurrences"] = Counter()
            stats["coarse_edge_occurrences"] = Counter()
        if phase == 1:
            # List of distances for each (cat1,cat2,direction) (excluding adjacent edges)
            stats["edge_distances"] = {}

        for graph in graphs:
            sys.stdout.write(f' Per-graph stuff (doing graph {graph.id})...                 \r')
            if phase == 2:
                cls = RelationshipGraph
                # Record cat co-occurrences
                cats_in_graph = list(set([n.category_name for n in graph.nodes]))
                for cat1 in cats_in_graph:
                    coarse_cat_1 = cls.cat_final_to_coarse(cat1)
                    for cat2 in cats_in_graph:
                        coarse_cat_2 = cls.cat_final_to_coarse(cat2)
                        stats["cat_co-occurrences"][hash_cat_pair(cat1, cat2)] += 1
                        stats["cat_co-occurrences"][hash_cat_pair(coarse_cat_1, coarse_cat_2)] += 1
                # Record edge occurrences
                for edge in graph.edges:
                    # Count once for every edge symmetrically equivalent to this one
                    for e in edge.sym_equivalences():
                        stats["edge_occurrences"][hash_edge(e)] += 1
                        cc1 = cls.cat_final_to_coarse(e.start_node.category_name)
                        cc2 = cls.cat_final_to_coarse(e.end_node.category_name)
                        et = e.edge_type
                        stats["coarse_edge_occurrences"][hash_cat_pair_edgetype(cc1, cc2, et)] += 1
            # Record edge distances (excluding adjacent edges and wall edges)
            if phase == 1:
                for edge in graph.edges:
                    cat1 = edge.start_node.category_name
                    #    and not edge.start_node.is_wall \
                    if edge.edge_type.is_spatial \
                        and edge.edge_type.has_dist \
                        and not (edge.start_node.is_wall and edge.end_node.is_wall) \
                        and edge.edge_type.dist != RelationshipGraph.EdgeType.Distance.ADJACENT:
                        assert(edge.dist is not None)
                        # Record one entry for every edge symmetrically equivalent to this one
                        for e in edge.sym_equivalences():
                            hashstr = hash_cat_dir(cat1, e.edge_type.direction)
                            if hashstr not in stats["edge_distances"]:
                                stats["edge_distances"][hashstr] = []
                            # We do relative distances, unless its a wall, in which case it's
                            #   absolute
                            if e.start_node.is_wall or e.start_node.diag_length is None:
                                dist = e.dist
                            else:
                                dist = e.dist/e.start_node.diag_length
                            stats["edge_distances"][hashstr].append(dist)

        if phase == 1:
            # # Determine where the proximal/distant split should be
            # # (This should add extra fields to 'stats')
            # RelationshipGraph.__determine_proximal_distant_split(stats)
            # Determine which categories can be 'hubs' for which other categories
            # Use the new proximal / distant threshold for this
            RelationshipGraph.__find_hubs(graphs, stats)

        # Determine which edge types we should keep vs. filter
        # (This should add extra fields to 'stats')
        if phase == 2:
            RelationshipGraph.__determine_keep_vs_filter_cats(stats)

        # Also save these stats to a file
        if save:
            dirname = RelationshipGraph.dirname(data_folder, data_root_dir)
            with open(f'{dirname}/stats_phase{phase}.json', 'w') as f:
                json.dump(stats, f)

        return stats

    @classmethod
    def __determine_proximal_distant_split(cls, stats):
        # Should create stats["proximal_threshold"] for every (anchor_cat, direction)
        stats["proximal_threshold"] = {}
        # Also record some statistics about how we computed the threshold
        stats["proximal_threshold_stats"] = {}

        # The basic algorithm is to split the list of distances into two bins, such
        #    that variance of the data falling into the bins is minimized.
        # We do with a multi-pass approach: subdivide the [min, max] distance range
        #    into some number of discrete points, evaluate the 'cost' of each
        #    possible split, pick the best one. Then, shrink the search range (to
        #    to be centered around the split we found last) and repeat this process.

        ### Hyperparameters
        N_SPLITS = 5
        N_ITERS = 4
        REDUCE_FACTOR = 0.5     # By what factor do we reduce the search range after each iter?
        BIN_SIZE_WEIGHT = 1

        # How big are the two partitions created by this split, relatively?
        def split_sizes(split_point, min_dist, max_dist):
            s = (split_point - min_dist) / (max_dist - min_dist)
            return s, 1 - s

        def find_best_split(dists, search_lo, search_hi, min_dist, max_dist):
            splits = [search_lo + ((i+0.5)/N_SPLITS)*(search_hi-search_lo) for i in range(N_SPLITS)]
            costs = [cost_of_split(dists, s, min_dist, max_dist) for s in splits]
            best_split_index = np.argmin(np.asarray(costs))
            return splits[best_split_index], costs[best_split_index]

        def cost_of_split(dists, split_point, min_dist, max_dist):
            bin1 = np.asarray([d for d in dists if d <= split_point])
            bin2 = np.asarray([d for d in dists if d > split_point])
            assert(len(bin1) > 0 and len(bin2) > 0)
            # print('-----------------------------')
            # print(f'Candidate split point: {split_point}')
            # print(f'Size of left bin: {len(bin1)}')
            # print(f'Size of right bin: {len(bin2)}')
            var1 = np.var(bin1)
            var2 = np.var(bin2)
            # # The cost also takes into account (as sort of a tie-breaker) the size of the
            # #    PROXIMAL bin (i.e. bin1) -- we'd like it to be small
            # size = split_sizes(split_point, min_dist, max_dist)[0]
            # cost = (var1 + var2)/2 + BIN_SIZE_WEIGHT*size
            cost = (var1 + var2)/2
            # print(f'Split cost: {cost}')
            # print('-----------------------------')
            return cost

        # https://docs.python.org/2/library/bisect.html
        def find_lt(a, x):
            'Find rightmost value less than x'
            i = bisect_left(a, x)
            if i:
                return a[i-1]
            return None
        def find_gt(a, x):
            'Find leftmost value greater than x'
            i = bisect_right(a, x)
            if i != len(a):
                return a[i]
            return None

        # Check for duplicates up to some tolerance
        def unique_dists(dists, tol=1e-5):
            # First, remove exact duplicates
            dists = list(set(dists))
            # Then, check for approximate duplicates
            dists.sort()
            ret = []
            while len(dists) > 0:
                d = dists.pop()
                lt = find_lt(dists, d)
                if lt and (abs(d - lt) < tol):
                    continue
                gt = find_gt(dists, d)
                if gt and (abs(d - gt) < tol):
                    continue
                ret.append(d)
            return ret

        # Special values for thresholds that we use to signify special cases
        ONE_UNQIUE_DIST = -10.0
        BIN_TOO_SMALL_ABS = -20.0
        BIN_TOO_SMALL_REL = -30.0

        for hashkey, dists in stats["edge_distances"].items():
            # Compute basic stats
            mean = np.mean(np.asarray(dists))
            var = np.var(np.asarray(dists))
            stats["proximal_threshold_stats"][hashkey] = {
                "Overall #": len(dists),
                "Overall mean": mean,
                "Overall variance": var
            }
            # Handle the case where there is only one unique dist and therefore we can't
            #    split into bins
            if len(unique_dists(dists)) == 1:
                # Make this be considered DISTANT by putting the proximal threshold below it
                stats["proximal_threshold"][hashkey] = ONE_UNQIUE_DIST
            else:
                min_dist = min(dists)
                max_dist = max(dists)
                search_lo = min_dist
                search_hi = max_dist
                curr_best_split = None
                curr_best_cost = math.inf
                for i in range(N_ITERS):
                    # print(f'================== ITER {i} ==================')
                    # print(f'lo: {search_lo}, hi: {search_hi}')
                    best_split, best_cost = find_best_split(dists, search_lo, search_hi, min_dist, max_dist)
                    # print(f'best_split: {best_split}, best_cost: {best_cost}')
                    if best_cost < curr_best_cost:
                        curr_best_split = best_split
                        curr_best_cost = best_cost
                    search_range = (search_hi - search_lo) * REDUCE_FACTOR
                    search_lo = max(min_dist, best_split - search_range/2)
                    search_hi = min(max_dist, best_split + search_range/2)
                # print('==========================================')
                # print(f'OVERALL BEST SPLIT: {curr_best_split}, COST: {curr_best_cost}')
                # exit()
                bin1 = np.asarray([d for d in dists if d <= curr_best_split])
                bin2 = np.asarray([d for d in dists if d > curr_best_split])
                prox_stats = stats["proximal_threshold_stats"][hashkey]
                prox_stats["Cost of best split"] = curr_best_cost
                prox_stats['# (bin 1)'] = len(bin1)
                prox_stats['# (bin 2)'] = len(bin2)
                prox_stats["Min (bin 1)"] = np.min(bin1)
                prox_stats["Max (bin 1)"] = np.max(bin1)
                prox_stats["Mean (bin 1)"] = np.mean(bin1)
                prox_stats["Variance (bin 1)"] = np.var(bin1)
                prox_stats["Min (bin 2)"] = np.min(bin2)
                prox_stats["Max (bin 2)"] = np.max(bin2)
                prox_stats["Mean (bin 2)"] = np.mean(bin2)
                prox_stats["Variance (bin 2)"] = np.var(bin2)

                # Set all edges of this type to be DISTANT if:
                # * There are two few distances in either bin, in an absolute sense
                if prox_stats["# (bin 1)"] < RelationshipGraph.prox_postprocess_count_thresh or \
                   prox_stats["# (bin 2)"] < RelationshipGraph.prox_postprocess_count_thresh:
                   stats["proximal_threshold"][hashkey] = BIN_TOO_SMALL_ABS
                # * There are two few distances in either bin, as a percentage of the total distances
                elif prox_stats["# (bin 1)"]/len(dists) < RelationshipGraph.prox_postprocess_percent_thresh or \
                     prox_stats["# (bin 2)"]/len(dists) < RelationshipGraph.prox_postprocess_percent_thresh:
                   stats["proximal_threshold"][hashkey] = BIN_TOO_SMALL_REL
                else:
                    stats["proximal_threshold"][hashkey] = curr_best_split
        
        print('')
        print('Proximal thresholds:')
        name_thresh_pairs = list(stats["proximal_threshold"].items())
        prox_stats = stats["proximal_threshold_stats"]
        name_thresh_pairs.sort(key=lambda pair: -prox_stats[pair[0]]["Overall #"])
        n_types = len(name_thresh_pairs)
        def print_thresh_stats(pairs):
            for name,thresh in pairs:
                name_prox_stats = prox_stats[name]
                print('------------------------------------------')
                print(f'Edge Type: {name}')
                statstr = f'Thresh: {thresh}\n'
                for k,v in name_prox_stats.items():
                    statstr += f'{k}: {v}\n'
                print(statstr)
        print('= ONLY ONE UNIQUE DISTANCE =')
        pairs_filtered = [(n,t) for n,t in name_thresh_pairs if t == ONE_UNQIUE_DIST]
        print(f'Num: {len(pairs_filtered)} ({int(100.0 * len(pairs_filtered)/n_types)}%)')
        # print_thresh_stats(pairs_filtered)
        print('= BIN TOO SMALL (ABSOLUTE COUNT) =')
        pairs_filtered = [(n,t) for n,t in name_thresh_pairs if t == BIN_TOO_SMALL_ABS]
        print(f'Num: {len(pairs_filtered)} ({int(100.0 * len(pairs_filtered)/n_types)}%)')
        # print_thresh_stats(pairs_filtered)
        print('= BIN TOO SMALL (RELATIVE PERCENTAGE) =')
        pairs_filtered = [(n,t) for n,t in name_thresh_pairs if t == BIN_TOO_SMALL_REL]
        print(f'Num: {len(pairs_filtered)} ({100.0 * len(pairs_filtered)/n_types}%)')
        # print_thresh_stats(pairs_filtered)
        print('=EVERYTHING OK =')
        pairs_filtered = [(n,t) for n,t in name_thresh_pairs if t > 0]
        print(f'Num: {len(pairs_filtered)} ({int(100.0 * len(pairs_filtered)/n_types)}%)')
        # print_thresh_stats(pairs_filtered)

    # NOTE: This operates on coarse categories, then converts back to fine categories
    # This lets us find more valid hubs than if we just looked at fine categories
    @classmethod
    def __find_hubs(cls, graphs, stats):
        cls = RelationshipGraph

        # Is it proximal or adjacent, based on the new stats we just computed?
        def is_prox_or_adj(edge):
            if not edge.edge_type.is_spatial:
                return False
            if edge.edge_type.is_adjacent:
                return True
            cat = edge.start_node.category_name
            d = edge.edge_type.direction
            # If "proximal_threshold" exists in the stats, then we must be using
            #    data-driven thresholding, so we should use that (b/c the graphs
            #    on disk haven't had their edge labels updated according to it yet)
            if "proximal_threshold" in stats:
                prox_thresh = stats["proximal_threshold"][hash_cat_dir(cat, d)]
                is_proximal = (edge.dist < prox_thresh)
            else:
                is_proximal = edge.edge_type.is_proximal
            if is_proximal:
                return True
            return False

        print('Finding hubs...                                                ')
        # This will record, for each hub category, what its possible spoke categories are
        stats["hub_categories"] = {}
        for graph in graphs:
            # Find all nodes that have multiple *different direction* ADJACENT/PROXIMAL edges
            #    to the same category (as long as that category isn't the same as the hub category)
            for n in graph.nodes:
                if n.is_arch:
                    continue
                hub_cat = n.category_name
                # Gather all the ADJACENT and PROXIMAL neighbors
                out_edges = [e for e in n.out_edges if is_prox_or_adj(e) and not e.end_node.is_arch]
                spoke_cat_dirs = {}
                spoke_nodes_seen = set([])
                for e in out_edges:
                    s = e.end_node
                    if s in spoke_nodes_seen:
                        continue
                    spoke_nodes_seen.add(s)
                    spoke_cat = s.category_name
                    # Can't be the same as the hub category
                    if spoke_cat != hub_cat:
                        if not spoke_cat in spoke_cat_dirs:
                            spoke_cat_dirs[spoke_cat] = []
                        spoke_cat_dirs[spoke_cat].append(e.edge_type.direction)
                for cat,dirs in spoke_cat_dirs.items():
                    is_hub = False

                    # We only consider it a hub if it's bigger than its spokes
                    if stats["category_sizes"][hub_cat] < stats["category_sizes"][cat]:
                        continue

                    dirset = set(dirs)    
                    # If the hub node is radially symmetric, then all we need is that this
                    #    category occurs more than once
                    if n.is_radially_symmetric:
                        is_hub = (len(dirs) > 1)
                    # If the hub node is left-right reflectionally symmetric, then we need
                    #    that both LEFT and RIGHT occur
                    elif n.is_left_right_reflect_symmetric:
                        is_hub = (Direction.LEFT in dirset) and (Direction.RIGHT in dirset)
                    # If the hub node is front-back reflectionally symmetric, then we need
                    #    that both FRONT and BACK occur
                    elif n.is_front_back_reflect_symmetric:
                        is_hub = (Direction.FRONT in dirset) and (Direction.BACK in dirset)
                    # If the hub node is two-way rotationally symmetric, then we need that
                    #    *EITHER* both FRONT and BACK occur *OR* LEFT and RIGHT occur
                    elif n.is_two_way_rotate_symmetric:
                        is_hub = \
                            (Direction.LEFT in dirset) and (Direction.RIGHT in dirset) or \
                            (Direction.FRONT in dirset) and (Direction.BACK in dirset)
                    # If the hub node is four-way rotationally symmetric, then we need that
                    #    *any* two distinct directions occur
                    elif n.is_four_way_rotate_symmetric:
                        is_hub = (len(dirset) > 1)
                    # If the hub node is corner 1 symmetric, then we need that *EITHER* 
                    #   FRONT AND LEFT occur *OR* BACK and RIGHT occur
                    elif n.is_corner_1_symmetric:
                        is_hub = \
                            (Direction.FRONT in dirset) and (Direction.LEFT in dirset) or \
                            (Direction.BACK in dirset) and (Direction.RIGHT in dirset)
                    # If the hub node is corner 2 symmetric, then we need that *EITHER* 
                    #   FRONT AND RIGHT occur *OR* BACK and LEFT occur
                    elif n.is_corner_2_symmetric:
                        is_hub = \
                            (Direction.FRONT in dirset) and (Direction.RIGHT in dirset) or \
                            (Direction.BACK in dirset) and (Direction.LEFT in dirset)
                    # We do not consider it a hub if the object has no symmetries

                    if is_hub:
                        # print('------------------------------------------')
                        # print('hub cat:', hub_cat)
                        # print('spoke cat:', cat)
                        # print('Hub Edges:')
                        # for e in (out_edges):
                        #     print(e)
                        coarse_hub_cat = cls.cat_final_to_coarse(hub_cat)
                        if not coarse_hub_cat in stats["hub_categories"]:
                            stats["hub_categories"][coarse_hub_cat] = []
                        stats["hub_categories"][coarse_hub_cat].append(cat)

        # Go through and check that for each possible hub --> spoke, we've seen that pattern
        #    enough times (in both an absolute and a relative sense)
        filtered_hub_categories = {}
        for hub_cat,spoke_cats in stats["hub_categories"].items():
            # Count the number of occurrences of each spoke cat
            spoke_cat_counts = Counter()
            for spoke_cat in spoke_cats:
                spoke_cat_counts[spoke_cat] += 1
            # Keep it if it's occurred frequently enough
            for spoke_cat,count in spoke_cat_counts.most_common():
                # TODO: Factor this out into a hyperparameter
                MIN_COUNT = 10
                if count > MIN_COUNT:
                    if hub_cat not in filtered_hub_categories:
                        filtered_hub_categories[hub_cat] = []
                    filtered_hub_categories[hub_cat].append(spoke_cat)    

        # For every hub -> spoke we found, also add hub -> eq_spoke for
        #   all eq_spoke that have the same coarse category as spoke (provided that
        #   eq_spoke actually occurs in this dataset)
        hub_categories_finalspokes = {}
        for hub_cat,spoke_cats in filtered_hub_categories.items():
            coarse_cats = [cls.cat_final_to_coarse(c) for c in spoke_cats]
            all_final_cats = []
            for c in coarse_cats:
                final_cats = [fc for fc in cls.cat_coarse_to_finals(c) if \
                    fc in stats["category_occurrences"]]
                all_final_cats.extend(final_cats)
            all_final_cats = list(set(all_final_cats))      # Deduplicate
            hub_categories_finalspokes[hub_cat] = all_final_cats

        # Finally, turn the hub categories back from coarse to final
        final_hub_categories = {}
        for hub_cat,spoke_cats in hub_categories_finalspokes.items():
            final_hub_cats = [c for c in cls.cat_coarse_to_finals(hub_cat) if \
                c in stats["category_occurrences"]]
            for hc in final_hub_cats:
                final_hub_categories[hc] = spoke_cats

        stats["hub_categories"] = final_hub_categories
        
        print('Possible hub/spoke relationships:')
        for hub,spokes in stats["hub_categories"].items():
            for spoke in spokes:
                print(f'{hub} --> {spoke}')


    # This phase is for fixing stuff from the initial graph extraction
    # We do this *before* saving the original graph for phase 1
    def __postprocess_phase0(self):
        # Delete any object -> arch edges (there was a brief bug where this was possible
        #    for certain ADJACENT edges)
        self.edges = [e for e in self.edges if not (e.end_node.is_arch and not e.start_node.is_arch)]
        self.__record_node_edges()

        self.__resplit_distances()

        # Delete any wall --> second-tier edges (this is now handled by initial graph
        #    construction; this is here just to help with old graphs)
        self.edges = [e for e in self.edges if not (e.start_node.is_wall and e.end_node.is_second_tier)]
        self.__record_node_edges()

        # Delete any non-support edges between first-tier and second-tier objects (in either direction)
        # This should be impossible (according to the initial graph extraction), but I'm seeing some
        #    instances of this popping up in the final graphs
        #### First, get rid of any edge between a second-tier and a first-tier object
        self.edges = [e for e in self.edges if not \
            (e.start_node.is_second_tier and not e.end_node.is_second_tier)]
        self.__record_node_edges()
        #### Then, get rid of any non-support edge between a first-tier and a second-tier object
        self.edges = [e for e in self.edges if not \
            (not e.start_node.is_second_tier and e.end_node.is_second_tier and not e.edge_type.is_support)]
        self.__record_node_edges()

        assert(not self.__has_duplicate_edges())

    # Postprocess phase 1 is split into two parts.
    # The first part does everything up to (and including) adding hubs and chains
    # Then we take a break, at which point we record all of the edges
    # Then we further remove DISTANT wall edges
    # We need to do this so that we have a record of *all* the hub edges that were added, in
    #    case we need to re-insert them later to restore graph connectivity
    def __postprocess_phase1_part1(self, stats):
        self.make_chains(stats)
        self.__break_single_edge_cycles(stats)
        self.make_hubs(stats)
        self.__record_node_edges()
        assert(not self.__has_duplicate_edges())
    def __postprocess_phase1_part2(self, stats):
        # Delete any wall --> non-wall edges that are not ADJACENT
        # NOTE: This needs to happen *after* make_hubs, so that hubs can inherit wall distant nodes
        #   from their spokes (in case these edges are needed later to restore connectivity)
        self.edges = [e for e in self.edges if not (e.start_node.is_wall and not e.end_node.is_wall \
            and not e.edge_type.is_adjacent)]
        # Do this after every time we remove edges
        self.__remove_useless_arch_nodes()
        self.__record_node_edges()
        assert(not self.__has_duplicate_edges())

    # The simpler, non-data-driven version
    def __resplit_distances(self):
        for edge in self.edges:
            et = edge.edge_type
            if et.has_dist \
                and not (edge.start_node.is_wall and edge.end_node.is_wall):
                # If the edge is labeled ADJACENT and has no dist, leave it that way
                if et.is_adjacent and edge.dist is None:
                    continue
                # Just repeat the same logic from the initial graph construction
                # (Repeated in case we have an old graph version)
                max_diag = max(edge.start_node.diag_length, edge.end_node.diag_length)
                adj_thresh = min(RelationshipGraph.adj_rel_threshold*max_diag,\
                        RelationshipGraph.adj_abs_threshold)
                prox_thresh = max(RelationshipGraph.prox_rel_threshold*max_diag,\
                    RelationshipGraph.prox_abs_threshold)
                if edge.dist < adj_thresh:
                    et.dist = RelationshipGraph.EdgeType.Distance.ADJACENT
                elif edge.dist < prox_thresh:
                    et.dist = RelationshipGraph.EdgeType.Distance.PROXIMAL
                else:
                    et.dist = RelationshipGraph.EdgeType.Distance.DISTANT


    # If an (anchor, target) edge and a (target, anchor) edge exist, keep only one of them
    def __break_single_edge_cycles(self, stats):

        # category_importances = stats["category_importances"]
        category_importances = stats["category_sizes"]  # Switching to this b/c it makes more sense...
        hub_categories = stats["hub_categories"]

        # Record pairs of opposite edges
        opposite_edge_pairs = []
        edges_seen = set([])
        for edge in self.edges:
            # We don't bother checking "wall ->" edges
            if not edge.start_node.is_wall:
                # Check if the opposite edge exists
                opposite_edges = [e for e in self.edges \
                    if (e.start_node == edge.end_node) and (e.end_node == edge.start_node)]
                if len(opposite_edges) > 0:
                    assert(len(opposite_edges) == 1)
                    opposite_edge = opposite_edges[0]
                    if not opposite_edge in edges_seen:
                        opposite_edge_pairs.append((edge, opposite_edge))
                        edges_seen.add(edge)

        # If the edge matches one of our hub patterns, then orient
        #    the edge from hub --> spoke
        pairs_handled = []
        for pair in opposite_edge_pairs:
            edge,opposite_edge = pair
            # Only valid for adjacent/proximal edges
            if edge.edge_type.is_distant:
                continue
            cat1 = edge.start_node.category_name
            cat2 = edge.end_node.category_name
            # CASE: There is a hub, and it goes 1 --> 2
            if cat1 in hub_categories and cat2 in hub_categories[cat1]:
                self.remove_edge(opposite_edge)
                pairs_handled.append(pair)
            # CASE: There is a hub, and it goes 2 --> 1
            elif cat2 in hub_categories and cat1 in hub_categories[cat2]:
                self.remove_edge(edge)
                pairs_handled.append(pair)
        opposite_edge_pairs = list(set(opposite_edge_pairs) - set(pairs_handled))

        # For the pairs that still remain:
        # If the categories are different, then orient the edge
        #    from most --> least important category
        pairs_handled = []
        for pair in opposite_edge_pairs:
            edge,opposite_edge = pair
            cat1 = edge.start_node.category_name
            cat2 = edge.end_node.category_name
            if cat1 != cat2:
                if cat2 == 'window':
                    print(edge)
                    print(opposite_edge)
                imp1 = category_importances[cat1]
                imp2 = category_importances[cat2]
                if imp1 > imp2:
                    self.remove_edge(opposite_edge)
                else:
                    self.remove_edge(edge)
                pairs_handled.append(pair)
        opposite_edge_pairs = list(set(opposite_edge_pairs) - set(pairs_handled))

        # For the pairs that still remain (the same-category pairs):
        # We impose an an arbitrary preference on
        #    directions. This has the side-benefit of ensuring that "chains"
        #    of objects of the same category get handled consistently.
        # Preference is FRONT -> BACK -> RIGHT -> LEFT
        dir2pref = {Direction.FRONT:0, Direction.BACK:1, Direction.RIGHT:2, Direction.LEFT:3}
        pairs_handled = []
        for pair in opposite_edge_pairs:
            edge,opposite_edge = pair
            lst = sorted([edge, opposite_edge], key=lambda e: dir2pref[e.edge_type.direction])
            self.remove_edge(lst[1])
            pairs_handled.append(pair)
        opposite_edge_pairs = list(set(opposite_edge_pairs) - set(pairs_handled))
        assert(len(opposite_edge_pairs) == 0)

        self.__record_node_edges()

    # Find hubs in this graph and mark them as such
    # Have hubs inherit inbound edges of their spokes
    def make_hubs(self, stats):
        # First, un-set the is_hub and is_spoke flag on all nodes
        # We want to do this from scratch
        for node in self.nodes:
            node.make_not_hub()
            node.make_not_spoke()

        hub_categories = stats["hub_categories"]
        self.__record_node_edges()
        for node in self.nodes:
            if node.category_name in hub_categories:
                spoke_cats = hub_categories[node.category_name]
                # Keep track of hub neighbors we already have
                # Initialize this to be the set of current neighbors
                hub_neigbors_seen = set(node.all_neighbors)
                # Check if this node has any spokes for each possible spoke type
                for sc in spoke_cats:
                    cls = RelationshipGraph
                    sc_coarse = cls.cat_final_to_coarse(sc)
                    spoke_nodes = [e.end_node for e in node.out_edges if \
                        cls.cat_final_to_coarse(e.end_node.category_name) == sc_coarse and \
                        (e.edge_type.is_adjacent or e.edge_type.is_proximal)]
                    # spoke nodes also can't be part of a superstructure already
                    # i.e. we can't have a hub for a spoke, we can't steal the spokes of another hub,
                    #    and we can't have a link in a chain as a spoke
                    spoke_nodes = [n for n in spoke_nodes if not n.is_superstructure]
                    if len(spoke_nodes) > 1:    # Have to have at least two spokes for it to be a hub
                        node.make_hub()
                        if "num_hubs" not in stats:
                            stats["num_hubs"] = 0
                        stats["num_hubs"] += 1
                        # Also go through and find all the outbound edges which connect
                        #    to these nodes, mark them as spokes
                        spoke_nodes = set(spoke_nodes)
                        for e in node.out_edges:
                            if e.end_node in spoke_nodes:
                                e.make_spoke()
                                e.end_node.make_spoke()
                        # node inherits all inbound edges to the spoke nodes, except for:
                        #   - edges from the hub itself
                        #   - edges from other spokes
                        #   - edges from nodes that the hub is already a neighbor of
                        for sn in spoke_nodes:
                            for e in sn.in_edges:
                                if e.start_node != node and \
                                not e.start_node in spoke_nodes and \
                                (not e.start_node in hub_neigbors_seen):
                                    inherit_edge = e.with_hub_endpoint(node)
                                    # If the edge is an adjacent one, we turn it into a proximal
                                    #   edge instead (better reflect the actual situation)
                                    if inherit_edge.edge_type.is_adjacent:
                                        inherit_edge = inherit_edge.with_distance(
                                            RelationshipGraph.EdgeType.Distance.PROXIMAL
                                        )
                                    self.edges.append(inherit_edge)
                                    hub_neigbors_seen.add(e.start_node)
                        # delete all inbound edges to spokes *except* for:
                        #  - the one from the hub
                        #  - any adjacent edges
                        for sn in spoke_nodes:
                            in_edges = sn.in_edges[:]
                            for e in in_edges:
                                if e.start_node != node and (e.edge_type.is_distant or e.edge_type.is_proximal):
                                    self.remove_edge(e)


    # Find chains in this graph and mark them as such
    # A chain is a series (>= 2) of adjacent nodes in the same direction that have the same category
    # *OR* a chain is a series (>= 3) of proximal nodes in the same direction that have the same
    #    category *and* the same modelID
    def make_chains(self, stats=None):
        # First, un-set the is_chain bit on all nodes and edges (if it's set)
        # We want to start from a clean slate
        for node in self.nodes:
            node.make_not_chain()
        for edge in self.edges:
            edge.make_not_chain()

        # Helper
        def find_chain_from(node, direction, distance, nodes_seen=set([])):
            # Can't do it from walls, doors, or windows
            if node.is_arch:
                return []
            cat = node.category_name
            # First, filter out everything that isn't in the desired direction or doesn't 
            #    have the desired distance
            next_edges = [e for e in node.out_edges if e.edge_type.is_spatial and \
                e.edge_type.direction == direction and e.edge_type.dist == distance]
            # If the desired distance is ADJACENT, filter out things that are drastically
            #   different in size
            def keep_adj_size(e):
                size_ratio = e.start_node.diag_length / e.end_node.diag_length
                return size_ratio > 0.5 and size_ratio < 1.5
            if distance == RelationshipGraph.EdgeType.Distance.ADJACENT:
                next_edges = [e for e in next_edges if keep_adj_size(e)]
            # If the desired distance is PROXIMAL, filter out things that don't have the
            #   same modelID
            elif distance == RelationshipGraph.EdgeType.Distance.PROXIMAL:
                next_edges = [e for e in next_edges if e.end_node.modelID == node.modelID]
            # If no edges meet these criteria, this is not a chain
            if len(next_edges) == 0:
                return []
            # If there is more than one next edge in this direction, this does not have 
            #    a linear chain topology
            if len(next_edges) > 1:
                return []
            next_edge = next_edges[0]
            # If the next node has more than one inbound edge in this direction, this
            #    does not have a linear chain topology
            next_in_edges = [e for e in next_edge.end_node.in_edges if e.edge_type.is_spatial and \
                e.edge_type.direction == direction and e.edge_type.dist == distance]
            if len(next_in_edges) > 1:
                return []
            # If the next edge would induce a cycle, then we're done
            if next_edge.end_node in nodes_seen:
                return []
            # Only at this point do we say that there is another link in this chain
            nodes_seen = nodes_seen | set([node])
            return [next_edge] + find_chain_from(next_edge.end_node, direction, distance, nodes_seen)

        SMOOTH_DOT_THRESH = 0.75
        def chain_is_approx_collinear(chain):
            nodes = [chain[0].start_node]
            for e in chain:
                nodes.append(e.end_node)
            # Check that subsequent vectors are not-too-far from aligned
            for i in range(0, len(nodes)-2):
                p0 = nodes[i].pos
                p1 = nodes[i+1].pos
                p2 = nodes[i+2].pos
                v01, v12 = p1 - p0, p2 - p1
                v01 = v01 / np.linalg.norm(v01)
                v12 = v12 / np.linalg.norm(v12)
                if np.dot(v01, v12) < SMOOTH_DOT_THRESH:
                    return False
            return True
        
        chains = []  # A chain is a list of edges
        edges_in_chains = set([])

        min_edges_for_chain = 2
        keep_looking = True
        while keep_looking:
            candidate_chains = []
            found_chain = False
            for node in self.nodes:
                for direction in Direction.all_directions():
                    # Check for adjacent chain
                    chain = find_chain_from(node, direction, \
                        RelationshipGraph.EdgeType.Distance.ADJACENT)
                    if len(chain) >= min_edges_for_chain and chain_is_approx_collinear(chain):
                        # Check if we already have a chain involving one or more of these edges
                        if not np.any(np.asarray([(e in edges_in_chains) for e in chain])):
                            found_chain = True
                            candidate_chains.append(chain)
                    # Check for proximal chain
                    chain = find_chain_from(node, direction, \
                        RelationshipGraph.EdgeType.Distance.PROXIMAL)
                    if len(chain) >= min_edges_for_chain and chain_is_approx_collinear(chain):
                        # Check if we already have a chain involving one or more of these edges
                        if not np.any(np.asarray([(e in edges_in_chains) for e in chain])):
                            found_chain = True
                            candidate_chains.append(chain)
            if found_chain:
                # Find the longest chains
                candidate_chains.sort(key=lambda chain: -len(chain))
                longest_length = len(candidate_chains[0])
                candidate_chains = [c for c in candidate_chains if len(c) == longest_length]
                # If there are multiple chains tied for the longest, we sort using the same
                #    preference order for directions that we use in breaking single edge cycles
                dir2pref = {Direction.FRONT:0, Direction.BACK:1, Direction.RIGHT:2, Direction.LEFT:3}
                candidate_chains.sort(key=lambda c: dir2pref[c[0].edge_type.direction])
                longest_chain = candidate_chains[0]
                # Keep this chain, and record which edges are involved in it
                chains.append(longest_chain)
                for e in longest_chain:
                    edges_in_chains.add(e)
                    # Also delete the opposite edge, if it exists in the graph
                    n1, n2 = e.start_node, e.end_node
                    if self.contains_edge_between(n2, n1):
                        self.remove_edge(self.get_edge_between(n2, n1))

            # We keep looking if we found a chain on this iteration
            # Otherwise, we've found all the possible chains and can stop
            keep_looking = found_chain

        # Go through all the chains and mark their nodes and edges
        if stats:
            if "num_chains" not in stats:
                stats["num_chains"] = 0
            stats["num_chains"] += len(chains)
        for chain in chains:
            if stats:
                if "chain_lengths" not in stats:
                    stats["chain_lengths"] = Counter()
                stats["chain_lengths"][len(chain)+1] += 1   # num nodes, not num edges
            for e in chain:
                e.make_chain()
                e.start_node.make_chain()
                e.end_node.make_chain()

    def __postprocess_phase2(self, stats):
        self.__prune_unimportant_edges(stats)
        self.__record_node_edges()
        assert(not self.__has_duplicate_edges())

    @classmethod
    def __determine_keep_vs_filter_cats(cls, stats):
        stats["keep_edge_type"] = {}
        stats["edge_rel_freq"] = {}
        cls = RelationshipGraph
        # for edgetypestr,edge_count in stats["edge_occurrences"].items():
        for edgetypestr,edge_count in stats["coarse_edge_occurrences"].items():
            cat1, cat2, edge_type = unhash_edge(edgetypestr)
            cat_co_count = stats["cat_co-occurrences"][hash_cat_pair(cat1, cat2)]
            relfreq = edge_count / cat_co_count
            # We keep an edge if it occurs frequently enough
            if edge_type.is_adjacent:
                keep = (relfreq >= RelationshipGraph.adjacent_filter_thresh)
            elif edge_type.is_proximal:
                keep = (relfreq >= RelationshipGraph.proximal_filter_thresh)
            elif edge_type.is_distant:
                keep = (relfreq >= RelationshipGraph.distant_filter_thresh)
            finalcats1 = cls.cat_coarse_to_finals(cat1)
            finalcats2 = cls.cat_coarse_to_finals(cat2)
            edgestrs = [hash_cat_pair_edgetype(fc1, fc2, edge_type) \
                for fc1 in finalcats1 for fc2 in finalcats2]
            for es in edgestrs:
                # Only record this for the final cat pairs that were actually observed in the data
                if es in stats["edge_occurrences"]:
                    stats["keep_edge_type"][es] = keep
                    stats["edge_rel_freq"][es] = relfreq

        def print_kept_filtered(kept, filtered):
            def printlist(lst):
                for name,freq in lst:
                    print(f'{name} ({100.0 * freq:.2f}%)')
            
            # Sort by decreasing frequency
            kept.sort(key=lambda p: 1-p[1])
            filtered.sort(key=lambda p: 1-p[1])
            # Get rid of the support and wall edges
            kept = [p for p in kept if not unhash_edge(p[0])[0] == 'wall' and \
                not unhash_edge(p[0])[1] == 'wall' and not unhash_edge(p[0])[2].is_support]
            filtered = [p for p in filtered if not unhash_edge(p[0])[0] == 'wall' and \
                not unhash_edge(p[0])[1] == 'wall' and not unhash_edge(p[0])[2].is_support]
            # Divide these each into, adjacent, proximal, distant
            kept_adjacent = [p for p in kept if unhash_edge(p[0])[2].is_adjacent]
            kept_proximal = [p for p in kept if unhash_edge(p[0])[2].is_proximal]
            kept_distant = [p for p in kept if unhash_edge(p[0])[2].is_distant]
            filtered_adjacent = [p for p in filtered if unhash_edge(p[0])[2].is_adjacent]
            filtered_proximal = [p for p in filtered if unhash_edge(p[0])[2].is_proximal]
            filtered_distant = [p for p in filtered if unhash_edge(p[0])[2].is_distant]
            # Within each of those bins, print kept first and then filtered
            print('======================== ADJACENT ========================')
            printlist(kept_adjacent)
            print('----------------------------------------------------------')
            printlist(filtered_adjacent)
            print('======================== PROXIMAL ========================')
            printlist(kept_proximal)
            print('----------------------------------------------------------')
            printlist(filtered_proximal)
            print('======================== DISTANT  ========================')
            printlist(kept_distant)
            print('----------------------------------------------------------')
            printlist(filtered_distant)
            print('----------------------------------------------------------')

        # Print the log of what relationships we kept/filtered
        print('')
        kept = [(es,stats["edge_rel_freq"][es]) for es,kept in stats["keep_edge_type"].items() if kept]
        filtered = [(es,stats["edge_rel_freq"][es]) for es,kept in stats["keep_edge_type"].items() if not kept]
        n_total = len(kept) + len(filtered)
        print_kept_filtered(kept, filtered)
        print(f'EDGE TYPES KEPT: {len(kept)}/{len(filtered)} ({100.0 * len(kept)/n_total:.2f}%)')

    def __prune_unimportant_edges(self, stats):
        output_edges = []
        for edge in self.edges:
            cat1 = edge.start_node.category_name
            cat2 = edge.end_node.category_name

            keep = False

            # We keep support edges (have to have, for parent child relationships)
            if edge.edge_type.is_support:
                keep = True
            # We keep wall-wall edges
            if cat1 == 'wall' and cat2 == 'wall':
                keep = True
            # It is a wall adjacent edge (these are super important for positioning) or a
            #    window adjacent edge (these are functionally the same thing as wall adjacent edges)
            if (cat1 == 'wall' or cat1 == 'window') and edge.edge_type.is_adjacent:
                keep = True
            # It is a wall -> arch edge (can't remove these; the only way that things attach)
            if cat1 == 'wall' and edge.end_node.is_arch:
                keep = True
            # We don't remove spokes or chain edges
            if edge.is_spoke or edge.is_chain:
                keep = True
            # We also don't remove nearby facing relationships: that is, FRONT PROXIMAL edges
            #    where the anchor has a meaningful front.
            # The idea here is that these relationships reflect intentionality: thing A may be
            #    incidentally placed next to thing B, but it's unlikely to have been incidentally
            #    placed directly in front of thing B
            if edge.edge_type.is_proximal and edge.edge_type.direction == Direction.FRONT and \
                edge.start_node.has_unique_front:
                keep = True
            # We also don't remove adjacent edges between two objects of the same coarse category
            if edge.edge_type.is_adjacent:
                cls = RelationshipGraph
                coarse_cat_1 = cls.cat_final_to_coarse_force(cat1)
                coarse_cat_2 = cls.cat_final_to_coarse_force(cat2)
                if coarse_cat_1 == coarse_cat_2:
                    keep = True
            # Finally, we keep the edge types that our stats say to keep.
            if stats["keep_edge_type"][hash_edge(edge)]:
                keep = True
            # I no longer always keep wall/window proximal edges. I figure that these are
            #   better handled by __ensure_connectivity (i.e. the object should be connected to
            #   the wall/window only if there's nothing better/closer for it to connect to)
            if keep:
                output_edges.append(edge)
        self.edges = output_edges

        self.__record_node_edges()

    def set_chain_middle_edges(self):
        def hash_in_edge(e):
            return f'{e.start_node.id},{e.edge_type.base_name},{e.edge_type.dist}'
        chains = self.get_all_chains()
        for chain in chains:
            start_node = chain[0].start_node
            end_node = chain[-1].end_node
            middle_nodes = [e.end_node for e in chain][0:-1]
            # Middle nodes: delete all their inbound edges.
            # They inherit whatever adj/proximal inbound edges are shared by both their
            #    start and end node
            start_in_edges = [hash_in_edge(e) for e in start_node.in_edges \
                if not e in chain and not e.edge_type.is_distant]
            end_in_edges = [hash_in_edge(e) for e in start_node.in_edges \
                if not e in chain and not e.edge_type.is_distant]
            shared_edges = set(start_in_edges) & set(end_in_edges)
            for mnode in middle_nodes:
                in_edges = mnode.in_edges[:]
                # Remove all of the middle node's in edges (that aren't part of the chain)
                for e in in_edges:
                    if not e in chain:
                        self.remove_edge(e)
                # Add an instance of each shared edge
                for estr in shared_edges:
                    start_id,direction,dist = estr.split(',')
                    et = RelationshipGraph.EdgeType(direction, dist)
                    edge = RelationshipGraph.Edge(start_id, mnode.id, et, graph=self)
                    self.add_edge(edge)
        self.__sort_edges()

    def fix_corner_symmetries(self):
        for node in self.nodes:
            if node.is_corner_1_symmetric and node.modelID.endswith('mirror'):
                node.sym_types = set(['__CORNER_2'])
            elif node.is_corner_2_symmetric and node.modelID.endswith('mirror'):
                node.sym_types = set(['__CORNER_1']) 

    def assert_is_connected(self, msg='Graph is disconnected!'):
        is_connected, disconnected_nodes = self.is_connected()
        if not is_connected:
            print(msg)
            print('------------------------------------------------------------------')
            print(self)
            print('------------------------------------------------------------------')
            print('Offending nodes:')
            for n in disconnected_nodes:
                print(n)
            exit()

    # Returns True if removing this edge would disconnect the graph
    def edge_connects_graph(self, edge):
        self.remove_edge(edge)
        result = self.check_is_connected()
        self.add_edge(edge)
        return not result
    
    def check_is_connected(self, msg=None):
        is_connected, disconnected_nodes = self.is_connected()
        if not is_connected and msg is not None:
            print(msg)
        return is_connected


    # Verify that there's a path from some wall node to every other node
    def is_connected(self):
        wall_nodes = [n for n in self.nodes if n.is_wall]
        start_node = random.choice(wall_nodes)
        # Run a DFS and verify that we've hit every node
        visited = set([])
        fringe = [start_node]
        while len(fringe) > 0:
            node = fringe.pop()
            visited.add(node)
            for e in node.out_edges:
                next_node = e.end_node
                if not (next_node in visited):
                    fringe.append(next_node)
        # If we've visited everything, return true
        # Else, return false, plus a list of nodes we haven't visited
        if len(visited) == len(self.nodes):
            return True, None
        else:
            return False, [n for n in self.nodes if not (n in visited)]

    def has_cycle(self):
        return len(self.find_cycles()) > 0

    def find_cycles(self):
        graph = self.copy()
        # Remove all the edge that involve walls
        edges = graph.edges[:]
        for e in edges:
            if e.start_node.is_wall or e.end_node.is_wall:
                graph.remove_edge(e)
        cycles = []
        while True:
            cycle = graph.__find_one_cycle()
            if cycle is None:
                break
            cycles.append(cycle)
            for edge in cycle:
                graph.remove_edge(edge)
        return cycles

    def __find_one_cycle(self):
        # We run a DFS to check for the presence of a cycle, and we keep around
        #    the entire path so that we can recover the cycle itself.
        # Since we remove offending edges after every cycle detection, this may
        #    cause the graph to become disconnected. Thus, we actually have to
        #    run DFS repeatedly until all edges have been visited.
        edges_touched = set([])
        while len(edges_touched) < len(self.edges):
            # Grab the first edge that's not been touched yet, start with that one
            start_edge = [e for e in self.edges if not (e in edges_touched)][0]
            fringe = [ [start_edge] ]
            while len(fringe) > 0:
                path = fringe.pop()
                last_edge = path[-1]
                edges_touched.add(last_edge)
                last_node = last_edge.end_node
                for e in last_node.out_edges:
                    if e in path:
                        # This is a cycle
                        return path
                    else:
                        fringe.append(path + [e])
        # If we finish checking all edges and no cycle was found, then return None
        return None

    # This is meant to be applied on generated graphs
    # Does a whole bunch of checks/clean up.
    # Returns False if we detect a chain that is topologically impossible
    def clean_up_chains(self, verbose=False):

        def LOG(msg):
            if verbose:
                print(msg)

        LOG('==================================================================')

        # Helper for both phases
        def try_insert_path(path):
            graph = self.copy()
            graph.add_path(path)
            if graph.check_is_connected() and not graph.has_cycle():
                self.add_path(path)
                return True
            return False

        # As a preprocess, we first have to mark all chain edges (b/c the net doesn't
        #    predict edge attributes).
        for n in self.nodes:
            if n.is_chain:
                for e in n.in_edges:
                    if e.start_node.is_chain:
                        # Verify that this has at least one more edge in the same direction
                        #    before we're ok with calling it a chain
                        n2 = e.start_node
                        for e2 in n2.in_edges:
                            if e2.start_node.is_chain and \
                                e2.edge_type.direction == e.edge_type.direction and \
                                e2.edge_type.dist == e.edge_type.dist:
                                e.make_chain()
                                e2.make_chain()

        # The graph net may have generated some graphs thave nodes annotated as chain nodes
        #    but which are not actually part of a chain. Detect these and make them not chains
        # Easy to detect: if there's no edge between them, or if there is but the previous
        #    step didn't mark it as a chain, then this is not a chain
        for n in self.nodes:
            if n.is_chain:
                # Check n's edges in both directions. If none of them is a chain edge,
                #    then this node can't be a chain
                for e in n.all_edges:
                    if e.is_chain:
                        break
                else:
                    n.make_not_chain()

        # Now we can ask for all the chains
        chains = self.get_all_chains()

        # Detect topologically impossible chains (and bail if we find any)
        for chain in chains:
            # (1) A link has more than one inbound edge from / outbound edge to some other node in the chain
            chain_nodes = set([chain[0].start_node] + [e.end_node for e in chain])
            for node in chain_nodes:
                chain_in_neighbors = [n for n in node.in_neighbors if n in chain_nodes]
                chain_out_neighbors = [n for n in node.out_neighbors if n in chain_nodes]
                if len(chain_in_neighbors) > 1 or len(chain_out_neighbors) > 1:
                    print('Topologically impossible chain (Link has more than one edge to/from another link)')
                    return False
            # (2) Link has more than one inbound or more than one outbound edge in the chain direction
            chain_dir = chain[0].edge_type.direction
            chain_opp_dir = Direction.opposite(chain_dir)
            for n in chain_nodes:
                dir_in_edges = [e for e in n.in_edges if \
                    e.edge_type.is_spatial and \
                    (e.edge_type.direction == chain_dir or e.edge_type.direction == chain_opp_dir)]
                dir_out_edges = [e for e in n.out_edges if \
                    e.edge_type.is_spatial and \
                    (e.edge_type.direction == chain_dir or e.edge_type.direction == chain_opp_dir)]
                if len(dir_in_edges) > 1 or len(dir_out_edges) > 1:
                    print('Topologically impossible chain (Link has more than one edge in the chain direction)')
                    return False
            # (3) Chain is not all proximal or all adjacent edges
            chain_dist = chain[0].edge_type.dist
            for e in chain:
                if e.edge_type.dist != chain_dist:
                    print('Topologically impossible chain (edges with different distance labels)')
                    return False
            # (4) Chain is all proximal but nodes are not all the same category
            if chain[0].edge_type.is_proximal:
                cats = set([n.category_name for n in chain_nodes])
                if len(cats) > 1:
                    print('Topologically impossible chain (proximal chain w/ different category nodes')

        #print(self)

        # (1) Flip whole chain b/c end node is more constrained than start node?
        for chain in chains:
            start_node = chain[0].start_node
            end_node = chain[-1].end_node
            # We weight adjacent edges twice as much as proximal edges (e.g. two inbound
            #    proximal edges counts as much as one inbound adjacent edge)
            start_constraint_count = \
                2 * len([e for e in start_node.in_edges if e.edge_type.is_adjacent]) + \
                len([e for e in start_node.in_edges if e.edge_type.is_proximal])
            end_constraint_count = \
                2 * len([e for e in end_node.in_edges if e.edge_type.is_adjacent]) + \
                len([e for e in end_node.in_edges if e.edge_type.is_proximal])
            if end_constraint_count > start_constraint_count:
                LOG('Flipping a chain whose end node was more constrained than its start')
                LOG(f'start constraint count: {start_constraint_count}')
                LOG(f'end constraint count: {end_constraint_count}')
                LOG('Chain:')
                for e in chain:
                    LOG(f'  {e}')
                LOG('----------------------------------------------------------')
                # Flip the chain
                flipped_chain = [e.flipped() for e in chain]
                for e in flipped_chain:
                    e.make_chain()
                # Add it (but only if it doesn't create a cycle or disconnection)
                try_insert_path(flipped_chain)

        LOG('==================================================================')

        
        # Some helpers for part (2)

        def get_start_extend_candidate_edge_direct(chain):
            # What kind of chain is this (proximal or adjacent)?
            chain_type = chain[0].edge_type
            start_node = chain[0].start_node
            # Could this chain be extended from its start?
            # (i.e. does the start have any inbound edges that could work?)
            start_candidate_edges = [e for e in start_node.in_edges if \
                (e.edge_type.direction == chain_type.direction) and \
                ((chain_type.is_adjacent and e.edge_type.is_adjacent) or \
                (chain_type.is_proximal and e.edge_type.is_proximal and \
                    e.end_node.category_name == start_node.category_name))]
            # We additionally filter out any arch nodes
            start_candidate_edges = [e for e in start_candidate_edges if not e.start_node.is_arch]
            if len(start_candidate_edges) > 0:
                # Dunno if this ever occurs, but it wouldn't be a chain structure
                if len(start_candidate_edges) > 1:
                    return None
                return start_candidate_edges[0]
            return None

        def get_end_extend_candidate_edge_direct(chain):
            # What kind of chain is this (proximal or adjacent)?
            chain_type = chain[0].edge_type
            end_node = chain[-1].end_node
            # Could this chain be extended from its end?
            # (i.e. does the end have any outbound edges that could work?)
            end_candidate_edges = [e for e in end_node.out_edges if \
                (e.edge_type.base_name != RelationshipGraph.EdgeType.SUPPORT) and \
                (e.edge_type.direction == chain_type.direction) and \
                ((chain_type.is_adjacent and e.edge_type.is_adjacent) or \
                (chain_type.is_proximal and e.edge_type.is_proximal and \
                    e.start_node.category_name == end_node.category_name))]
            if len(end_candidate_edges) > 0:
                # Dunno if this ever occurs, but it wouldn't be a chain structure
                if len(end_candidate_edges) > 1:
                    return None
                return end_candidate_edges[0]
            return None

        def get_start_extend_candidate_edge_for_flip(chain):
            # What kind of chain is this (proximal or adjacent)?
            chain_type = chain[0].edge_type
            start_node = chain[0].start_node
            # Could this chain be extended from its start?
            # (i.e. does the start have any outbound edges that could work?)
            start_candidate_edges = [e for e in start_node.out_edges if \
                (e.edge_type.base_name != RelationshipGraph.EdgeType.SUPPORT) and \
                (e.edge_type.direction == Direction.opposite(chain_type.direction)) and \
                ((chain_type.is_adjacent and e.edge_type.is_adjacent) or \
                (chain_type.is_proximal and e.edge_type.is_proximal and \
                    e.end_node.category_name == start_node.category_name))]
            if len(start_candidate_edges) > 0:
                # Dunno if this ever occurs, but it wouldn't be a chain structure
                if len(start_candidate_edges) > 1:
                    return None
                return start_candidate_edges[0]
            return None

        def get_end_extend_candidate_edge_for_flip(chain):
            # What kind of chain is this (proximal or adjacent)?
            chain_type = chain[0].edge_type
            end_node = chain[-1].end_node
            # Could this chain be extended from its end?
            # (i.e. does the end have any inbound edges that could work?)
            end_candidate_edges = [e for e in end_node.in_edges if \
                (e.edge_type.base_name != RelationshipGraph.EdgeType.SUPPORT) and \
                (e.edge_type.direction == Direction.opposite(chain_type.direction)) and \
                ((chain_type.is_adjacent and e.edge_type.is_adjacent) or \
                (chain_type.is_proximal and e.edge_type.is_proximal and \
                    e.start_node.category_name == end_node.category_name))]
            # We additionally filter out any arch nodes
            end_candidate_edges = [e for e in end_candidate_edges if not e.start_node.is_arch]
            if len(end_candidate_edges) > 0:
                # Dunno if this ever occurs, but it wouldn't be a chain structure
                if len(end_candidate_edges) > 1:
                    return None
                return end_candidate_edges[0]
            return None

        def get_chain_containing_edge(chains, edge):
            for chain in chains:
                if edge in chain:
                    return chain
            return None

        def try_extend_chain(chain, chains, candidate_edge):
            # If this edge is itself part of a chain, we don't do the extension
            # (Because our chain flipping routine in part 1 has already put the 
            #    final world on which direction chains should go in)
            chain_for_edge = get_chain_containing_edge(chains, candidate_edge)
            if chain_for_edge:
                return False
            new_edges = [candidate_edge.flipped()]
            # Try adding the edge
            if try_insert_path(new_edges):
                for e in new_edges:
                    e.make_chain()
                    e.start_node.make_chain()
                    e.end_node.make_chain()
                return True
            else:
                return False

        # (2) Try to extend chains existing chains as much as possible
        # Start with the longest one first, try to extend
        no_extendable_chains = False
        while not no_extendable_chains:
            chains = self.get_all_chains()
            LOG('************************* OUTER ITERATION **************************')
            LOG('ALL CHAINS:')
            for chain in chains:
                LOG('  Chain:')
                for e in chain:
                    LOG(f'    {e}')
            for chain in chains:
                # Could this chain be extended directly from its start by an edge that already
                #    exists?
                candidate_edge = get_start_extend_candidate_edge_direct(chain)
                if candidate_edge:
                    LOG('Extended a chain from its start node (directly)')
                    LOG('Chain:')
                    for e in chain:
                        print(f'  {e}')
                    LOG('Edge added:')
                    LOG(f'  {candidate_edge}')
                    LOG('----------------------------------------------------------')
                    # Mark this edge (and the node it connects to) as chains, move on
                    candidate_edge.make_chain()
                    candidate_edge.start_node.make_chain()
                    candidate_edge.end_node.make_chain()
                    break
                # Could this chain be extended directly from its end by an edge that already
                #    exists?
                candidate_edge = get_end_extend_candidate_edge_direct(chain)
                if candidate_edge:
                    LOG('Extended a chain from its end node (directly)')
                    LOG('Chain:')
                    for e in chain:
                        print(f'  {e}')
                    LOG('Edge added:')
                    LOG(f'  {candidate_edge}')
                    LOG('----------------------------------------------------------')
                    # Mark this edge (and the node it connects to) as chains, move on
                    candidate_edge.make_chain()
                    candidate_edge.start_node.make_chain()
                    candidate_edge.end_node.make_chain()
                    break
                # Could this chain be extended from its start by flipping an edge?
                candidate_edge = get_start_extend_candidate_edge_for_flip(chain)
                if candidate_edge and try_extend_chain(chain, chains, candidate_edge):
                    LOG('Extended a chain from its start node (by flipping)')
                    LOG('Chain:')
                    for e in chain:
                        print(f'  {e}')
                    LOG('Edge flipped:')
                    LOG(f'  {candidate_edge}')
                    LOG('----------------------------------------------------------')
                    # Great, it worked. Now the set of chains has changed, so we start the
                    #    outer loop over again
                    break
                # Could this chain be extended from its end by flipping and edge?
                candidate_edge = get_end_extend_candidate_edge_for_flip(chain)
                if candidate_edge and try_extend_chain(chain, chains, candidate_edge):
                    LOG('Extended a chain from its end node (by flipping)')
                    LOG('Chain:')
                    for e in chain:
                        print(f'  {e}')
                    LOG('Edge flipped:')
                    LOG(f'  {candidate_edge}')
                    LOG('----------------------------------------------------------')
                    # Great, it worked. Now the set of chains has changed, so we start the
                    #    outer loop over again
                    break
            else:
                # Got through all chains and never broke due to finding an extension
                # So we must be done
                no_extendable_chains = True

        # At the end of all of this, make sure that chain middle nodes inherit the
        #    inbound edges shared by the chain endpoints (and discard any others)
        #self.set_chain_middle_edges()

        # Recompute and re-sort node edge lists before returning
        self.__record_node_edges()
        self.__sort_edges()

        return True

    def get_nodes_in_instantiation_order(self, data_folder, data_root_dir=None, verbose=False):
        if data_root_dir is None:
            data_root_dir = utils.get_data_root_dir()

        # List of object nodes in the order they should be instantiated
        nodes_in_order = []
        # Initialize the set of nodes in the scene to be the set of arch nodes
        nodes_in_scene = set([n for n in self.nodes if n.is_arch])

        # Parameters
        ADJ_WEIGHT = 3
        PROX_WEIGHT = 2
        DIST_WEIGHT = 1
        AREA_POWER = 0.8

        def is_sandwiched(chain_start_node, chains):
            # Check each chain that this node is a part of
            for chain in chains:
                chain_end_node = chain[-1].end_node
                chaindir = chain[0].edge_type.direction
                # Criteria for sandwiching:
                #  (1) Both start and end nodes have an inbound proximal/adjacent edge
                #  (2) The edge is not part of the chain itself
                #  (3) The two edges are from different objects
                start_in_neighbors = [e.start_node for e in chain_start_node.in_edges if \
                    (e.edge_type.is_adjacent or e.edge_type.is_proximal)]
                end_in_neighbors = [e.start_node for e in chain_end_node.in_edges if \
                    (e.edge_type.is_adjacent or e.edge_type.is_proximal) and \
                    not (e.is_chain and e.edge_type.direction == chaindir)]
                # If these two sets are both nonempty and nonidentical, 
                #    then we're sandwiched
                if len(start_in_neighbors) > 0 and len(end_in_neighbors) > 0 and \
                    len(set(start_in_neighbors).symmetric_difference(set(end_in_neighbors))) > 0:
                    return True
            return False

        # Expected size of each category of object
        catmap = RelationshipGraph.category_map
        category_names = catmap.all_non_arch_categories(data_root_dir, data_folder)
        category_sizes = catmap.all_non_arch_category_sizes(data_root_dir, data_folder)
        cat_size_map = {}
        for i,cat in enumerate(category_names):
            cat_size_map[cat] = category_sizes[i]
        def size(n):
            return cat_size_map[n.category_name]

        def can_be_added(node):
            DEBUG = False
            # DEBUG = (node.id == 'synth_10')
            if DEBUG:
                print(f'<<< Trying to add {node.id} >>>')
            # If it's already in the scene, we can't add this node again
            if node in nodes_in_scene:
                if DEBUG:
                    print('<<<  Fail because already in scene >>>')
                return False
            # Can only be added if all inbound neighbors are in the scene already
            for n in node.in_neighbors:
                if not n in nodes_in_scene:
                    if DEBUG:
                        print(f'<<<  Fail because missing inbound neighbor: {n} >>>')
                    return False
            # For chain starts: all inbound neighbors to all chain links (except other chain
            #    links) must also be in the scene
            is_chain_start = node.is_chain_start
            if is_chain_start:
                chains = self.get_chains_starting_from(node)
                for chain in chains:
                    chain_nodes = set([node] + [e.end_node for e in chain])
                    prev_node = node
                    for e in chain:
                        curr_node = e.end_node
                        for n in curr_node.in_neighbors:
                            if not n in chain_nodes and not n in nodes_in_scene:
                                if DEBUG:
                                    print('<<<  Fail because chain link missing inbound neighbor >>>')
                                    print('CHAIN:')
                                    for e in chain:
                                        print(f'* {e}')
                                    print('Chain link:', curr_node)
                                    print('Missing inbound neighbor:', n)
                                    print('NODES IN SCENE:')
                                    for nis in nodes_in_scene:
                                        print(f'* {nis}')
                                return False
                        prev_node = curr_node
            # Other chain nodes (non-start nodes) can't be added (they're grouped with their starts)
            if node.is_chain and not is_chain_start:
                if DEBUG:
                    print('<<<  Fail because a chain node, but not the start node >>>')
                return False
            return True

        # How constrained is this individual node?
        # This is based on how many (and how severe) incoming edges it has (from objects that are already
        #    in the scene)
        Dist = RelationshipGraph.EdgeType.Distance
        dist2weight = {Dist.ADJACENT: ADJ_WEIGHT, Dist.PROXIMAL: PROX_WEIGHT, Dist.DISTANT: DIST_WEIGHT}
        def how_constrained(node, in_scene=None):
            if in_scene is None:
                in_scene = nodes_in_scene
            return sum([dist2weight[e.edge_type.dist] for e in node.in_edges if \
                e.edge_type.is_spatial and e.start_node in in_scene])

        # How constraining is the subgraph reachable from this node, if we were to instantiate it?
        # This is based on the sum of expected area of the all the subgraph nodes that could be 
        #    added at this point, where each node area is weighted by how constrained that node is.
        # We also do area ^ AREA_POWER, which is a super rough way of accounting for the fact that
        #    sometimes area matters and sometimes linear extent matters (e.g. along walls)
        def how_constraining(node, in_scene=None):
            if in_scene is None:
                in_scene = nodes_in_scene
            # First, find all nodes we can reach from this node (including this node) that
            #    have not yet been added to the scene.
            visited = set()
            fringe = [node]
            while len(fringe) > 0:
                n = fringe.pop()
                visited.add(n)
                for n2 in n.out_neighbors:
                    if not n2 in visited and not n2 in in_scene:
                        fringe.append(n2)
            reachable = visited
            # Next, filter for only the nodes whose inbound neighbors are either
            #  (a) also reachable, or
            #  (b) already in the scene
            # These are the nodes that we'd actually be able to instantiate at this time
            nodes_can_add = [n for n in reachable if np.all([
                (p in reachable) or (p in in_scene) for p in n.in_neighbors
            ])]
            # Finally, sum up the constrained-ness scores for all these nodes
            # In evaluating how constrained they are, consider not just the nodes in the scene,
            #    but also the reachable nodes (i.e the nodes that would be in the scene, were
            #    we to go down this subgraph)
            constraining_nodes = in_scene | reachable
            return sum([pow(size(n), AREA_POWER) * how_constrained(n, constraining_nodes) \
                for n in nodes_can_add])

        # Sort one level of nodes based on constrained/constraining
        def constraint_based_sort(nodes):
            constrained = {n:how_constrained(n) for n in nodes}
            constraining = {n:how_constraining(n) for n in nodes}
            def compare(n1, n2):
                # First, sort based on how constrained they are
                if constrained[n1] > constrained[n2]:
                    return -1
                if constrained[n2] > constrained[n1]:
                    return 1
                # Next, sort based on how constraining they are
                if constraining[n1] > constraining[n2]:
                    return -1
                if constraining[n2] > constraining[n1]:
                    return 1
                # I highly doubt this ever happens...
                return 0
            nodes.sort(key=cmp_to_key(compare))
            # print('---------------------------------------------------------')
            # for n in nodes:
            #     print(f'{n} | constrained: {constrained[n]:.2f}, constraining: {constraining[n]:.2f}')
            
        def constraint_ordered_traversal(nodes, depth=0):
            def LOG(msg):
                if verbose:
                    print('   ' * depth, msg)

            # Filter out any nodes that we can't add yet
            filtered_nodes = [n for n in nodes if not can_be_added(n)]
            nodes = [n for n in nodes if can_be_added(n)]
            constraint_based_sort(nodes)
            if len(nodes) == 0:
                LOG('No nodes to add...')
            if len(filtered_nodes) > 0:
                LOG('These nodes were filtered out b/c they cannot be added:')
                for n in filtered_nodes:
                    LOG(f'* {n}')
            else:
                LOG('=================================================================================')
                LOG('Consdering adding these nodes:')
                for node in nodes:
                    LOG(f'* {node}')
            for node in nodes:
                LOG('Nodes already added to the scene:')
                for n in nodes_in_scene:
                    LOG(f'* {n}')
                LOG('Trying this node:')
                LOG(f'* {node}')
                if node in nodes_in_scene: # We've already done this node
                    LOG('Already in scene; skipping')
                    continue
                LOG('NOT Already in scene; adding to scene!')
                nodes_in_order.append(node)
                nodes_in_scene.add(node)
                # If this is a chain node, we can assume it will be a chain start
                # We treat a chain like one object: we add the nodes in the chain,
                #    in order, and then all the rest of the chain's neighbors
                if node.is_chain:
                    LOG('This node is part of a chain; adding all of its chain nodes at once')
                    # A node can be part of two orthogonal chains
                    chains = self.get_chains_starting_from(node)
                    assert len(chains) > 0
                    # Check if this node is sandwiched
                    if is_sandwiched(node, chains):
                        node.is_sandwiched = True
                    else:
                        node.is_sandwiched = False
                    # If there are multiple chains, which one do we do first?
                    # Well, we sort based on how constraining the chains are
                    if len(chains) > 1:
                        chainconsts = [sum([how_constraining(e.end_node) for e in chain]) \
                            for chain in chains]
                        indices = list(range(len(chains)))
                        indices.sort(key = lambda i: -chainconsts[i])
                        chains = [chains[i] for i in indices]
                    # Store these chains in an attribute on the start node, so Kai can
                    #   have easy access
                    node.chains_starting_from = chains
                    LOG('These are the chains the node is the start of:')
                    for chain in chains:
                        LOG('CHAIN:')
                        for e in chain:
                            LOG(f'* {e}')
                    # Do all the chains in order
                    for chain in chains:
                        # Don't actually add the chain nodes to the nodes_in_order, but *do*
                        #    record that they are now in the scene
                        for e in chain:
                            nodes_in_scene.add(e.end_node)
                        # Treat all the out neighbors as if they were the neighbors of one object
                        first_tier_ns = first_tier_out_neighbors(node)
                        second_tier_ns = second_tier_out_neighbors(node)
                        for e in chain:
                            first_tier_ns.extend(first_tier_out_neighbors(e.end_node))
                            second_tier_ns.extend(second_tier_out_neighbors(e.end_node))
                        LOG('Recursing on chain first tier neighbors')
                        constraint_ordered_traversal(first_tier_ns, depth+1)
                        LOG('Recursing on chain second tier neighbors')
                        constraint_ordered_traversal(second_tier_ns, depth+1)
                # If this is a hub, we do spokes before other neighbors
                elif node.is_hub:
                    LOG('This node is a hub; being sure to add its spokes before other neighbors')
                    # Find all the spoke out neighbors
                    spokes = [n for n in node.out_neighbors if n.is_spoke]
                    # Split these into groups by category
                    cat2spokes = {}
                    for s in spokes:
                        cat = s.category_name
                        if not cat in cat2spokes:
                            cat2spokes[cat] = []
                        cat2spokes[cat].append(s)
                    spoke_groups = list(cat2spokes.values())
                    # Sort these groups by how constraining they are
                    if len(spoke_groups) > 1:
                        groupconsts = [sum([how_constraining(s) for s in spokes]) \
                            for spoke in spoke_groups]
                        indices = list(range(len(spoke_groups)))
                        indices.sort(key = lambda i: -groupconsts[i])
                        spoke_groups = [spoke_groups[i] for i in indices]
                    # Recurse on each of the spoke groups in order
                    for spokes in spoke_groups:
                        LOG(f'Recursing on spoke group for category: {spokes[0].category_name}')
                        constraint_ordered_traversal(spokes)
                    # Then recurse on other neighbors
                    LOG('Recursing on hub first tier neighbors')
                    constraint_ordered_traversal(first_tier_out_neighbors(node), depth+1)
                    LOG('Recursing on hub second tier neighbors')
                    constraint_ordered_traversal(second_tier_out_neighbors(node), depth+1)
                else:
                    LOG('This is just a regular node; recursing on neighbors normally')
                    # Otherwise, we recurse on all the node's out neighbors
                    LOG('Recursing on first tier neighbors')
                    constraint_ordered_traversal(first_tier_out_neighbors(node), depth+1)
                    LOG('Recursing on first second neighbors')
                    constraint_ordered_traversal(second_tier_out_neighbors(node), depth+1)
                LOG('--------------------------------------------------------------------------------')

        def first_tier_out_neighbors(node):
            return [n for n in node.out_neighbors if not n.is_second_tier]
        def second_tier_out_neighbors(node):
            return [n for n in node.out_neighbors if n.is_second_tier]

        # Start by trying to add all the nodes
        constraint_ordered_traversal(self.nodes)
        # As a second pass, move all second tiers to the end (while preserving relative
        #    order)
        first_tiers = [n for n in nodes_in_order if not n.is_second_tier]
        second_tiers = [n for n in nodes_in_order if n.is_second_tier]
        nodes_in_order = first_tiers + second_tiers

        # Finally, go through and tag all nodes with their constrained/constraining score
        #    based on what would be in the scene as of this point
        in_scene = set([n for n in self.nodes if n.is_arch])
        for node in nodes_in_order:
            node.how_constrained = how_constrained(node, in_scene)
            node.how_constraining = how_constraining(node, in_scene)
            in_scene.add(node)

        return nodes_in_order


    # If we've got some graphs that don't have diag lengths on them, add that info
    #    in by reading from houses in the /json/ directory
    @classmethod
    def add_diag_lengths_from_json(cls, data_root_dir, data_folder):
        dirname = RelationshipGraph.dirname(data_folder, data_root_dir)
        json_dirname = f'{dirname}/../json'
        assert(os.path.exists(json_dirname))
        for fname in os.listdir(dirname):
            if fname.endswith('.pkl'):
                graph_filename = dirname + '/' + fname
                graph = RelationshipGraph().load_from_file(graph_filename)
                roomnum = os.path.splitext(fname)[0]
                json_filename = json_dirname + '/' + roomnum + '.json'
                with open(json_filename, 'r')as f:
                     house = House(house_json=json.load(f), include_support_information=False)
                rooms = house.get_rooms()
                assert(len(rooms) == 1)
                room = rooms[0]
                nodes = room.get_nodes() + room.walls
                for graph_node in graph.nodes:
                    node = [n for n in nodes if n.id == graph_node.id][0]
                    # Get object-space OBJ
                    if isinstance(node, Wall):
                        # Inverse-transform wall Obj into object space
                        obj = Obj(wall=node).transform(np.linalg.inv(node.transform))
                    else:
                        obj = Obj(node.modelId)
                    diag = obj_to_2d_diag_len(obj)
                    graph_node.set_diag_length(diag)
                # Save the graph back to the same file
                graph.save_to_file(graph_filename)

    # If we've got some graphs that don't have modelIDs on them, add that info
    #    in by reading from houses in the /json/ directory
    @classmethod
    def add_modelIDs_from_json(cls, data_root_dir, data_folder):
        dirname = RelationshipGraph.dirname(data_folder, data_root_dir)
        json_dirname = f'{dirname}/../json'
        assert(os.path.exists(json_dirname))
        for fname in os.listdir(dirname):
            if fname.endswith('.pkl'):
                graph_filename = dirname + '/' + fname
                graph = RelationshipGraph().load_from_file(graph_filename)
                roomnum = os.path.splitext(fname)[0]
                json_filename = json_dirname + '/' + roomnum + '.json'
                with open(json_filename, 'r')as f:
                     house = House(house_json=json.load(f), include_support_information=False)
                rooms = house.get_rooms()
                assert(len(rooms) == 1)
                room = rooms[0]
                nodes = room.get_nodes()
                for graph_node in graph.nodes:
                    if not graph_node.is_wall:
                        node = [n for n in nodes if n.id == graph_node.id][0]
                        graph_node.set_modelID(node.modelId)
                # Save the graph back to the same file
                graph.save_to_file(graph_filename)

    @classmethod
    def add_pos_from_json(cls, data_root_dir, data_folder, orig_dir=False):
        dirname = RelationshipGraph.dirname(data_folder, data_root_dir)
        if orig_dir:
            dirname += '/../graph_orig'
        json_dirname = f'{dirname}/../json'
        assert(os.path.exists(json_dirname))
        for fname in os.listdir(dirname):
            if fname.endswith('.pkl'):
                graph_filename = dirname + '/' + fname
                graph = RelationshipGraph().load_from_file(graph_filename)
                roomnum = os.path.splitext(fname)[0]
                json_filename = json_dirname + '/' + roomnum + '.json'
                with open(json_filename, 'r')as f:
                     house = House(house_json=json.load(f), include_support_information=False)
                rooms = house.get_rooms()
                assert(len(rooms) == 1)
                room = rooms[0]
                nodes = room.get_nodes()
                for graph_node in graph.nodes:
                    if not graph_node.is_wall:
                        node = [n for n in nodes if n.id == graph_node.id][0]
                        pos = np.array([(node.xmin+node.xmax)/2, (node.ymin+node.ymax)/2])
                        graph_node.set_pos(pos)
                # Save the graph back to the same file
                graph.save_to_file(graph_filename)
    
    @classmethod
    def regenerate_all_from_json(cls, data_root_dir, data_folder):
        graph_dirname = RelationshipGraph.dirname(data_folder, data_root_dir)
        json_dirname = f'{graph_dirname}/../json'
        assert(os.path.exists(json_dirname))
        assert(os.path.exists(graph_dirname))
        fnames = [f for f in os.listdir(json_dirname) if f.endswith('.json')]
        fnames.sort(key=lambda f: int(os.path.splitext(f)[0]))
        for fname in fnames:
            roomnum = os.path.splitext(fname)[0]
            sys.stdout.write(f' Regenerating graph {roomnum}...                               \r')
            json_filename = json_dirname + '/' + fname
            with open(json_filename, 'r') as f:
                # Suppress all the warnings
                stdout = sys.stdout
                sys.stdout = open('/dev/null', 'w')
                house = House(house_json=json.load(f))
                sys.stdout = stdout
            rooms = house.get_rooms()
            assert(len(rooms) == 1)
            room = rooms[0]
            graph = RelationshipGraph().extract_raw_from_room(room, int(roomnum))
            graph_filename = graph_dirname + '/' + roomnum + '.pkl'
            graph.save_to_file(graph_filename)


# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

def hash_cat_pair(cat1, cat2):
    return f'{cat1},{cat2}' if cat1 < cat2 else f'{cat2},{cat1}'
def hash_cat_pair_edgetype(cat1, cat2, edge_type):
    ret = f'{cat1},{cat2},{edge_type.base_name}'
    if edge_type.has_dist:
        ret += f',{edge_type.dist}'
    return ret
def hash_edge(edge):
    cat1 = edge.start_node.category_name
    cat2 = edge.end_node.category_name
    et = edge.edge_type
    return hash_cat_pair_edgetype(cat1, cat2, et)
def unhash_edge(edge_str):
    toks = edge_str.split(',')
    cat1 = toks[0]
    cat2 = toks[1]
    et = RelationshipGraph.EdgeType(*toks[2:])
    return cat1, cat2, et
def hash_cat_dir(cat, direction):
    return f'{cat},{direction}'

# -----------------------------------------------------------------------------------

# Functions for (quickly) symmetry-transforming images
# Assumes that images are [C,H,W] tensors

def __img_transpose(x):
    return torch.transpose(x, 1, 2)

def img_identity(x):
    return x

def img_reflect_horiz(x):
    return x.flip(2)

def img_reflect_vert(x):
    return x.flip(1)

def img_reflect_c2(x):
    return __img_transpose(x)

def img_reflect_c1(x):
    return img_reflect_horiz(__img_transpose(img_reflect_horiz(x)))

def img_rot_90(x):
    return img_reflect_vert(__img_transpose(x))

def img_rot_180(x):
    return img_reflect_horiz(img_reflect_vert(x))

def img_rot_270(x):
    return img_reflect_horiz(__img_transpose(x))

# Functions for (quickly) symmetry-transforming points
# Assumes that images are 2*n tensors, where the first dimension is [y, x]
def __pnt_transpose(x):
    return torch.stack([x[1], x[0]])

def pnt_identity(x):
    return x

def pnt_reflect_horiz(x):
    return x * torch.tensor([1, -1]).view(-1, 1).float()

def pnt_reflect_vert(x):
    return x * torch.tensor([-1, 1]).view(-1, 1).float()

def pnt_reflect_c2(x):
    return __pnt_transpose(x)

def pnt_reflect_c1(x):
    return pnt_reflect_horiz(__pnt_transpose(pnt_reflect_horiz(x)))

def pnt_rot_90(x):
    return pnt_reflect_vert(__pnt_transpose(x))

def pnt_rot_180(x):
    return pnt_reflect_horiz(pnt_reflect_vert(x))

def pnt_rot_270(x):
    return pnt_reflect_horiz(__pnt_transpose(x))
# -----------------------------------------------------------------------------------    

# Test whether two points are approximately colocated
def approx_colocated(pa, pb):
    return np.linalg.norm(np.asarray(pa)-np.asarray(pb)) < 1e-3

def wall_adjacent(wa,wb):
    f = approx_colocated
    return f(wa.points[0], wb.points[0]) \
        or f(wa.points[0], wb.points[1]) \
        or f(wa.points[1], wb.points[0]) \
        or f(wa.points[1], wb.points[1])

def normalize(x):
    return np.asarray(x) / np.linalg.norm(np.asarray(x))

# Collect all objectspace objs
# Also record the diagonal length (in 2D) of these Obj bboxes on the nodes themselves
def obj_to_2d_diag_len(obj):
    xmin, _, ymin, _ = obj.bbox_min
    xmax, _, ymax, _ = obj.bbox_max
    box2d_min = np.array([xmin, ymin])
    box2d_max = np.array([xmax, ymax])
    return np.linalg.norm(box2d_max - box2d_min)

# Compute the diagonal length (in object space) of an object
def diag_length(node, node2obj):
    # If it's a wall, we ignore the negligible thickness and just use the 
    #    wall segment length
    if node.id.startswith('wall'):
        return np.linalg.norm(node.pts[1] - node.pts[0])
    else:
        obj = node2obj[node]
        return obj_to_2d_diag_len(obj)

# -----------------------------------------------------------------------------------

# Utility function that takes a room which we know to have a closed loop of wall
#    segments, and modifies the room in place such that the wall segments are
#    ordered along a counterclockwise loop.
# It also modifies the 'index' and 'adjacent' properties of each Wall object to
#    to make them consistent with the new ordering
# It does *not* modify the 'id' property, b/c this must be kept as-is in order
#    for us to match walls between graphs and rendered scenes
def guarantee_ccw_wall_loop(room, room_index):
    # Can start the loop at an arbitrary point. WLoG, start with wall seg 0.
    start_wall = room.walls[0]

    # Find connection points (swapping point 0 for point 1 as necessary) until
    #    we have a complete loop (until the next wall becomes the start wall)
    wall_loop = []
    curr_wall = start_wall
    while True:
        wall_loop.append(curr_wall)
        # Find the wall segment connected to the end of this one
        adj_walls = [room.walls[ai] for ai in curr_wall.adjacent]
        found_adjacency = False
        for wall in adj_walls:
            if approx_colocated(curr_wall.pts[1], wall.pts[0]):
                curr_wall = wall
                found_adjacency = True
                break
            elif approx_colocated(curr_wall.pts[1], wall.pts[1]):
                wall.points.reverse()
                curr_wall = wall
                found_adjacency = True
                break
        if not found_adjacency:
            print(f'Could not find wall adjacent to {curr_wall.id} in room {room_index}')
            print('Walls in room:')
            for wall in room.walls:
                print('  ', wall.id, wall.adjacent, wall.pts[0], wall.pts[1])
            assert False
        if curr_wall is start_wall:
            break
    # Sanity check
    assert len(wall_loop) == len(room.walls)

    # Check if the loop is clockwise or counterclockwise.
    # If clockwise, reverse it (including swapping all point 0's and point 1's)
    # Using a cw/ccw test based on the shoelace formula
    # https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
    area = 0
    for wall in wall_loop:
        p0 = wall.pts[0]
        p1 = wall.pts[1]
        area += (p1[0] - p0[0])*(p1[1] + p0[1])
    if area > 0:    # positive area == clockwise
        wall_loop.reverse()
        for wall in wall_loop:
            wall.points.reverse()

    # Compute map from old index to new index, for each wall seg
    old2new = {}
    for i, wall in enumerate(wall_loop):
        old2new[wall.index] = i
    # Update all the 'index' and 'adjacent' properties
    for wall in wall_loop:
        wall.index = old2new[wall.index]
        wall.adjacent = [old2new[ai] for ai in wall.adjacent]
    # Reset room.walls to be wall_loop
    room.walls = wall_loop


# Search for nodes that have the same modelId and that are close to colocated.
# If such groups are found, delete all except one.
# The collision filter should catch these, but for whatever reason I've seen at least oen
#    instance of this popping up.
def remove_colocated_duplicates(room):
    def node_pos(node):
        return np.array([(node.xmin + node.xmax)/2, (node.ymin + node.ymax)/2, (node.zmin + node.zmax)/2])
    def node_max_dim(node):
        return max(node.length, node.width) 
    def is_dup(node, othernode):
        DUP_DIST_THRESH = 0.07
        return node.modelId == othernode.modelId and \
            np.linalg.norm(node_pos(node) - node_pos(othernode)) < DUP_DIST_THRESH * node_max_dim(node)

    nodes = room.get_nodes()
    dup_groups = []
    all_dups = set([])
    for node in nodes:
        if not node in all_dups:    # Skip dups we've already found
            dups = [n for n in nodes if n != node and is_dup(node, n)]
            if len(dups) > 0:
                dups.append(node)
                dup_groups.append(dups)
                for d in dups:
                    all_dups.add(d)
    # Fetch the list of all nodes which appear as parents of other nodes in the room
    parents = set()
    for node in nodes:
        if hasattr(node, 'parent') and node.parent is not None and \
            node.parent not in ['Wall', 'Floor']:
            parents.add(node.parent)
    removed = set()
    for dup_group in dup_groups:
        # We would like to remove all but one of these duplicates.
        # However, we can't remove any nodes which are parents of other nodes
        # So, we'll see which ones of these are parents.
        is_parent = [d in parents for d in dup_group]
        num_parents = sum(is_parent)
        # If none of them are, then great, we'll keep the last one (arbitrary)
        if num_parents == 0:
            dup_group.pop() # keep
            for n in dup_group:
                if not n in removed:
                    removed.add(n)
                    room.nodes.remove(n)
        # Otherwise, we have to keep all the ones that are parents (but we
        #    print a warning if there's more than one of these, as we technically
        #    haven't removed all duplicates...)
        if num_parents > 1:
            print('')
            print('Warning: could not remove all colocated duplicates due to parenting')
        for i,n in enumerate(dup_group):
            if not is_parent[i]:
                if not n in removed:
                    removed.add(n)
                    room.nodes.remove(n)


# -----------------------------------------------------------------------------------

if __name__ == '__main__':
    pass
    graph = RelationshipGraph().load_from_file('scratch/graphs/0.pkl')
    print(graph)
