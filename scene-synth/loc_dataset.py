from torch.utils import data
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
import numpy as np
import scipy.misc as m
import random
import math
import pickle
from data import ObjectCategories, RenderedScene, RenderedComposite, House, ProjectionGenerator, DatasetToJSON, ObjectData
from data import RelationshipGraph
import copy
import utils
from collections import defaultdict
import os
from functools import cmp_to_key

# ---------------------------------------------------------------------------------------
import torchvision
def save_input_local_unscaled(input_depth, edge, count, index):

    img_dir = f"test_fcn_edge/{index}/local_unscaled/"
    utils.ensuredir(img_dir)
    # print(f"Save local unscaled image {index}_{count}")
    edge_name = f'{edge.start_node.category_name}_{edge.edge_type.name}_{edge.end_node.category_name}'

    input_img = input_depth.unsqueeze(0).cpu()
    input_img = torchvision.transforms.ToPILImage()(input_img)
    input_img.save(f"{img_dir}/input_{count}_{edge_name}.png")

# ---------------------------------------------------------------------------------------

class FCNDatasetGraph():
    def __init__(self, scene_indices=(0,4000), data_folder="bedroom", data_root_dir=None, seed=None, \
                 canvas_scale=2):
        super(FCNDatasetGraph, self).__init__()
        self.seed = seed
        self.data_folder = data_folder
        self.data_root_dir = data_root_dir
        self.scene_indices = scene_indices
        self.canvas_scale = canvas_scale    # How much to scale the input/target canvas

    def __len__(self):
        return self.scene_indices[1]-self.scene_indices[0]

    def get_scene(self, index):
        i = index+self.scene_indices[0]
        return RenderedScene(i, self.data_folder, self.data_root_dir)

    def get_graph(self, index):
        i = index+self.scene_indices[0]
        return RelationshipGraph().load(i, self.data_folder, self.data_root_dir)

    def __getitem__(self,index):
        if self.seed:
            random.seed(self.seed)
        
        self.index = index

        scene = self.get_scene(index)
        composite = scene.create_composite()
        object_nodes = scene.object_nodes
        random.shuffle(object_nodes)

        # Also load the graph for index i
        # The rendered scene node["id"] field corresponds to the graph node id property
        graph = self.get_graph(index)
        # print(graph)

        all_nodes = object_nodes[:] + scene.door_window_nodes + scene.wall_segments
        def graphnode_to_renderednode(graph_node):
            return utils.find(lambda n: n['id'] == graph_node.id, all_nodes)

        if 'parent' in object_nodes[0]:
            #print([a["category"] for a in object_nodes])
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
            #print([a["category"] for a in object_nodes])
            #print("________________")
        
        def cmp_adjacent(node1, node2):
            # Start of adjacent chain should always be placing before other part of it
            n1 = utils.find(lambda n: n.id == node1['id'], graph.nodes)
            n2 = utils.find(lambda n: n.id == node2['id'], graph.nodes)
            if n1 is not None and n2 is not None:
                if n1.is_adjacent_chain and n2.is_adjacent_chain:
                    if n1.is_adjacent_chain_start:
                        return -1
                    elif n2.is_adjacent_chain_start:
                        return 1
            return 0
        
        object_nodes.sort(key = cmp_to_key(cmp_adjacent))

        # Select a split point at random as target (which objects we'll keep in the scene)
        def target_node_filter(target_node):
            # target_node is a scene node
            n = utils.find(lambda n: n.id == target_node['id'], graph.nodes)
            if n is None:
                # It's possible that there a scene node is not added to graph
                #   simply filtered it out
                return False
            # A target node should meet following condition:
            # The node should not be part of an adjacent chain. being the start of a chain is fine
            if n.is_adjacent_chain and not n.is_adjacent_chain_start:
                return False
            return True
        
        # Have to leave at least one object out
        target_indices = [i for i, n in enumerate(object_nodes) if target_node_filter(n) and i != len(object_nodes)-1]
        # print([object_nodes[i]["id"] for i in target_indices])
        if len(target_indices) == 0:
            print(f"Error: Scene {index} has no target nodes")
            for i, tn in enumerate(object_nodes):
                gn = utils.find(lambda n: n.id == tn['id'], graph.nodes)
                if gn is None:
                    print(f"{tn['id']} not in graph")
                elif gn.is_adjacent_chain and not gn.is_adjacent_chain_start:
                    print(f"{tn['id']} is in an adjecent chain (not the start)")
                else:
                    print(f"{tn['id']} is valid")

        ####
        while True:
            try:
                num_objects = random.choice(target_indices)
                #num_objects = random.randint(0, len(object_nodes)-1)
                pre_split_ids = set([n["id"] for n in object_nodes[:num_objects]])
                post_split_ids = set([n["id"] for n in object_nodes[num_objects:]])

                # print(pre_split_ids, post_split_ids)
                # The first 'num_objects' objects go into the input composite
                parent_ids = ["Floor", "Wall"]
                for i in range(num_objects):
                    node = object_nodes[i]
                    if node["parent"] == "Wall":
                        print("Massive messup!")
                    composite.add_node(node)
                    parent_ids.append(node["id"])
                
                parent_ids.append(None)

                # Select an 'anchor'
                # print('-----------------------------------------------')
                anchor_loc, anchor_orient, anchor_mask = None, None, None
                # print(pre_split_ids)
                # print(post_split_ids)
                # print(graph)
                def anchor_node_filter(n):
                    # n is a graph node
                    # Criteria 1: a valid node should be either (a) walls/windows/doors
                    #    or (b) in the first 'num_objects' objects in our object list.
                    if not (n.is_arch or n.id in pre_split_ids):
                        return False
                    
                    # A valid node need to have at least one edge meets both of the following:
                    # Criteria 2: only choose from those that have at least
                    #    one outgoing edge to some remaining object to be added to the scene.
                    # Criteria 3: must have at least one outgoing edge whose endpoint parent is already in the scene        
                    # input(str(len(n.out_edges)) + f': {n.category_name} ({n.id})')
                    for e in n.out_edges:
                        # print(e)
                        # print(f"end node to be added?: {e.end_node.id in post_split_ids}") # Criteria 2
                        # print(f"end node has valid parent?: {e.end_node.is_wall or graphnode_to_renderednode(e.end_node)['parent'] in parent_ids}") # Criteria 3
                        # if not e.end_node.is_wall:
                        #     print(f"...... {e.end_node.category_name} has parent {graphnode_to_renderednode(e.end_node)['parent']}")

                        if e.end_node.id in post_split_ids and (e.end_node.is_wall or \
                            graphnode_to_renderednode(e.end_node)['parent'] in parent_ids):
                            return True
                    
                    return False

                # We do this by selecting from among the graph nodes that are either (a) walls/windows/doors
                #    or (b) in the first 'num_objects' objects in our object list.
                # candidate_graphnodes = [n for n in graph.nodes if (n.is_arch or n.id in pre_split_ids)]
                #print(candidate_graphnodes)
                # We further must filter these nodes to only choose from those that have at least
                #    one outgoing edge to some remaining object to be added to the scene.
                # candidate_graphnodes = [n for n in candidate_graphnodes \
                #    if len([e for e in n.out_edges if e.end_node.id in post_split_ids]) > 0]
                # One final filter: must have at least one outgoing edge whose endpoint parent is already in the scene
                # candidate_graphnodes = [n for n in candidate_graphnodes \
                #     if len([e for e in n.out_edges if e.end_node.is_wall or graphnode_to_renderednode(e.end_node)['parent'] in parent_ids]) > 0]
                candidate_graphnodes = [n for n in graph.nodes if anchor_node_filter(n)]
                # print(candidate_graphnodes)
                # Select the anchor randomly from among these graphnodes
                if not len(candidate_graphnodes) > 0:
                    print("Something's wrong, check splitting point")
                    print(pre_split_ids, post_split_ids)
                    for n in graph.nodes:
                        if not (n.is_arch or n.id in pre_split_ids):
                            continue
                        print(str(len(n.out_edges)) + f': {n.category_name} ({n.id})')
                        for e in n.out_edges:
                            print("  ", e)
                            print(f"    end node to be added?: {e.end_node.id in post_split_ids}") # Criteria 2
                            print(f"    end node has valid parent?: {e.end_node.is_wall or graphnode_to_renderednode(e.end_node)['parent'] in parent_ids}") # Criteria 3
                            #if not e.end_node.is_wall:
                            #    print(f"...... {e.end_node.category_name} has parent {graphnode_to_renderednode(e.end_node)['parent']}")

                if not (len(candidate_graphnodes) > 0):
                    print(f"No candidate node at {index}")
                    raise
                #assert(len(candidate_graphnodes) > 0), f"No candidate node at {index}"

                anchor_graphnode = random.choice(candidate_graphnodes)
                # print(anchor_graphnode)
                # What we do next depends on whether this is a wall or an object
                if anchor_graphnode.is_wall:
                    #### It's a wall segment
                    # Grab the rendered scene segment that corresponds to this graphnode
                    segments = scene.wall_segments
                    seg_index = int(anchor_graphnode.id.split('_')[1]) # Assumes that wall ids are of the form 'wall_n' for index 
                    anchor = segments[seg_index]
                    anchor_seg = anchor["points"]
                    anchor_normal = anchor["normal"]
                    # Compute anchor location
                    anchor_loc = torch.Tensor(anchor_seg[0] + anchor_seg[1])/2
                    anchor_loc = (anchor_loc/composite.size - 0.5) * 2
                    # Compute anchor orientation
                    anchor_orient = torch.Tensor(anchor_normal)
                    # Compute anchor mask
                    anchor_mask = compute_object_mask(anchor, composite.size, index)
                else:
                    #### It's an object
                    # Grab the rendered scene obj node that corresponds to this graph node
                    anchor = graphnode_to_renderednode(anchor_graphnode)
                    # Compute anchor location
                    xmin, _, ymin, _ = anchor["bbox_min"]
                    xmax, _, ymax, _ = anchor["bbox_max"]
                    anchor_loc = torch.Tensor([(xmin+xmax)/2, (ymin+ymax)/2])
                    anchor_loc = (anchor_loc/composite.size - 0.5) * 2
                    # Compute anchor orientation
                    xform = anchor["transform"]
                    # cos = xform[0]
                    # sin = xform[8]
                    scale = (xform[0]**2+xform[8]**2)**0.5
                    # sin, cos are swapped because here the first axis is y direction
                    # The y direction is flipped because positive is defined as pointing up here
                    sin = xform[0]/scale
                    cos = -xform[8]/scale
                    anchor_orient = torch.Tensor([cos, sin])
                    # Compute anchor mask
                    # assert (xmin >= 0 and ymin >= 0), f"{index}, {anchor['category']}, bounding box out of canvas"
                    # print(index)
                    anchor_mask = compute_object_mask(anchor, composite.size, index)

                # TODO: Use the anchor object's symmetry type to choose a coordinate frame...

                # Now we need to choose an outgoing edge from the anchor node (the 'anchor edge')
                # Filter for edges whose endpoints occur after 'num_objects'
                # print(anchor_graphnode)
                candidate_edges = [e for e in anchor_graphnode.out_edges if e.end_node.id in post_split_ids]
                # Filter for edges whose endpoints have parents already in the scene
                candidate_edges = [e for e in candidate_edges if graphnode_to_renderednode(e.end_node)['parent'] in parent_ids]
                if not (len(candidate_edges) > 0):
                    print(f"No candidate edge at scene {index}")
                    raise
                # assert(len(candidate_edges) > 0), f"No candidate edge at scene {index}"
                # Choose one of these edges at random to be our anchor edge
                edge = random.choice(candidate_edges)
            except:
                continue
            break

        ###

        # Randomly select symmetrically edge
        sym_edge, sym_xform_img, sym_xform_pnt = random.choice(list(edge.sym_equivalences_with_image_transforms()))

        # Create the composite input image
        # We need 'num_extra_channels=1' so that we can add the anchor mask channel
        inputs = composite.get_composite(num_extra_channels=1)
        inputs[inputs.shape[0]-1] = anchor_mask
        # Inverse transform the inputs to be relative to the anchor coordinate frame
        inputs = inverse_xform_img(inputs, anchor_loc, anchor_orient, self.canvas_scale)
        # Do symmetric transform
        inputs = sym_xform_img(inputs)

        # Any edge that has the same type as this: their endpoints go in the output 'centroids' image
        edges_of_type = [e for e in candidate_edges if e.edge_type == edge.edge_type]
        assert(len(edges_of_type) > 0)
        centroids, labels = [], []
        for e in edges_of_type:
            end_node = e.end_node
            obj_node = graphnode_to_renderednode(end_node)
            if obj_node["parent"] == "Wall":
                print("Massive messup!")
            if obj_node["parent"] in parent_ids:
                xmin, _, ymin, _ = obj_node["bbox_min"]
                xmax, _, ymax, _ = obj_node["bbox_max"]
                centroids.append(((xmin+xmax)/2, (ymin+ymax)/2))
                labels.append(obj_node["category"])
        
        # Transform centroids before writing values into 'output'
        size = inputs.shape[1]
        out_size = self.canvas_scale * 64
        
        centroids = torch.transpose(torch.tensor(centroids), 0, 1) # 2*n
        centroids = (centroids/size - 0.5) * 2
        # Do Inverse transfrom for centroids
        centroids = inverse_xform_pnts(centroids, anchor_loc, anchor_orient, 2)
        # Do symmetric transfrom for centroids
        centroids = sym_xform_pnt(centroids)
        centroids = (centroids/2 + 0.5) * out_size
        # Construct output img
        output = torch.zeros((out_size,out_size)).long()
        for i, label in enumerate(labels):
            x, y = centroids[:, i]
            # Only write centroids if they're still within the output image bounds
            if x >=0 and x < out_size and y >= 0 and y < out_size:
                output[math.floor(x),math.floor(y)] = label+1
        
        edge = sym_edge

        # Return
        #  - the input composite
        #  - the output centroids map
        #  - the anchor loc and orient
        #  - the edge type label
        return inputs, output, anchor_loc, anchor_orient, torch.Tensor([edge.edge_type.index])
        

# ---------------------------------------------------------------------------------------

class FCNDatasetGraph2():
    def __init__(self, scene_indices=(0,4000), data_folder="bedroom", data_root_dir=None, seed=None, \
                 canvas_scale=2):
        super(FCNDatasetGraph2, self).__init__()
        self.seed = seed
        self.data_folder = data_folder
        self.data_root_dir = data_root_dir
        self.scene_indices = scene_indices
        self.composite_size = None
        self.canvas_scale = canvas_scale    # How much to scale the input/target canvas

    def __len__(self):
        return self.scene_indices[1]-self.scene_indices[0]

    def get_scene(self, index):
        i = index+self.scene_indices[0]
        scene = RenderedScene(i, self.data_folder, self.data_root_dir)
        if not self.composite_size:
            self.composite_size = scene.size
        return scene

    def get_graph(self, index):
        i = index+self.scene_indices[0]
        return RelationshipGraph().load(i, self.data_folder, self.data_root_dir)

    def __getitem__(self, index):
        if self.seed:
            random.seed(self.seed)
        self.index = index
        ### At test time a graphnode is choosen to be added into the scene
        # 1. Pick a graphnode to insert
        # 2. Get all incoming edge to that node
        # 3. All start node of candidate edge is used as anchor
        # 4. Transform the input according to the anchor
        scene = self.get_scene(index)
        graph = self.get_graph(index)

        composite = scene.create_composite()

        all_nodes = scene.object_nodes + scene.door_window_nodes
        def graphnode_to_renderednode(graph_node):
            if graph_node.is_wall:
                # Grab the rendered scene segment that corresponds to this graphnode
                segments = scene.wall_segments
                # Assumes that wall ids are of the form 'wall_n' for index 
                seg_index = int(graph_node.id.split('_')[1])
                return segments[seg_index]
            else:
                # The rendered scene node["id"] field corresponds to the graph node id property
                return utils.find(lambda n: n['id'] == graph_node.id, all_nodes)
        
        while True:
            try:
                # For the purpose here, we randomly pick a graph node to add
                graphnode = random.choice([n for n in graph.nodes if not n.is_arch])
                scenenode = graphnode_to_renderednode(graphnode)

                candidate_edges = self._get_edges(scene, graph, graphnode)
            except:
                continue
            break
        
        # Add back objects already in scence
        pre_split_ids = self._get_pre_split_ids(graph, graphnode)
        exist_graphnodes = [n for n in graph.nodes if n.id in pre_split_ids and not n.is_arch]
        for node in exist_graphnodes:
            scene_node = graphnode_to_renderednode(node)
            composite.add_node(scene_node)
        inputs = composite.get_composite(num_extra_channels=1)

        # Get target mask
        # For convinence, target is NOT TRANSFORMED
        target_loc, target_orient, target_mask = self._get_anchor(graphnode, graphnode_to_renderednode)
        target_direction_mask = draw_direction_mask(target_loc, target_orient, target_mask)
        target = torch.stack([target_mask, target_direction_mask], 0)

        # inputs, anchor_loc, anchor_orient, edge_type
        items = []
        # print(f"Index: {index}")
        # print(candidate_edges)
        for edge in candidate_edges:
            data = inputs.clone()
            anchor_graphnode = edge.start_node
            anchor_loc, anchor_orient, anchor_mask = self._get_anchor(anchor_graphnode, graphnode_to_renderednode)
            data[data.shape[0]-1] = anchor_mask
            # Inverse transform the inputs to be relative to the anchor coordinate frame
            data = inverse_xform_img(data, anchor_loc, anchor_orient, self.canvas_scale)
            items.append([data, target, anchor_loc, anchor_orient, torch.Tensor([edge.edge_type.index]), torch.Tensor([scenenode["category"]])])
        
        return items

    
    """
    def __getitem__(self, index):
        if self.seed:
            random.seed(self.seed)
        ### At test time, following information is needed:
        #   - anchor_graphnode and candidate_edges from graph
        #   - composite, the ids of nodes already in the scene, and the ids of all possible nodes in the scene
        #   - the mapping between graph node and scene node `graphnode_to_renderednode`
        ### At training, these information are choosed randomly
        #   The graph and scene must be in sync; i.e., all nodes prior to adding anchor_graphnode
        # must already be added to composite. scene_nodes are all nodes (objects, doors, windows, walls) 
        # avaliable in the scene

        # Note: scene_nodes and parent_ids should be stored somewhere, possibly in the composite?
        anchor_graphnode, candidate_edges, composite, scene_nodes, parent_ids, graphnode_to_renderednode = self._get_random_input(index)
        
        anchor_loc, anchor_orient, anchor_mask = self._get_anchor(anchor_graphnode, graphnode_to_renderednode)
        # TODO: Use the anchor object's symmetry type to choose a coordinate frame...

        # Create the composite input image
        # We need 'num_extra_channels=1' so that we can add the anchor mask channel
        inputs = composite.get_composite(num_extra_channels=1)
        inputs[inputs.shape[0]-1] = anchor_mask
        # Inverse transform the inputs to be relative to the anchor coordinate frame
        inputs = inverse_xform_img(inputs, anchor_loc, anchor_orient, self.canvas_scale)

        if self.all_edges:
            # This is only used in *TEST* time. The output is kept as None for simplicity (ans since it's not used)
            output = torch.Tensor([0])
            edge_type = [e.edge_type.index for e in candidate_edges]
        else:
            # Choose one of these edges at random to be our anchor edge
            edge = random.choice(candidate_edges)
            # Any edge that has the same type as this: their endpoints go in the output 'centroids' image
            edges_of_type = [e for e in candidate_edges if e.edge_type == edge.edge_type]
            assert(len(edges_of_type) > 0)
            output = self._get_output_transform(edges_of_type, anchor_loc, anchor_orient, parent_ids, graphnode_to_renderednode)
            edge_type = [edge.edge_type.index]
        
        # Return
        #  - the input composite
        #  - the output centroids map
        #  - the anchor loc and orient
        #  - the edge type label
        return inputs, output, anchor_loc, anchor_orient, torch.Tensor(edge_type)
        """
    
    def _get_output_transform(self, edges_of_type, anchor_loc, anchor_orient, parent_ids, graphnode_to_renderednode):
        # Return the transfromed output based on the selected edge and anchor
        centroids, labels = [], []
        for e in edges_of_type:
            end_node = e.end_node
            obj_node = graphnode_to_renderednode(end_node)
            if obj_node["parent"] == "Wall":
                print("Massive messup!")
            if obj_node["parent"] in parent_ids:
                xmin, _, ymin, _ = obj_node["bbox_min"]
                xmax, _, ymax, _ = obj_node["bbox_max"]
                centroids.append(((xmin+xmax)/2, (ymin+ymax)/2))
                labels.append(obj_node["category"])
        # Inverse transform centroids before writing values into 'output'
        size = self.composite_size
        out_size = self.canvas_scale * 64
        centroids = np.transpose(torch.tensor(centroids))
        centroids = (centroids/size - 0.5) * 2
        centroids = inverse_xform_pnts(centroids, anchor_loc, anchor_orient, 2)
        centroids = (centroids/2 + 0.5) * out_size
        # Construct output img
        output = torch.zeros((out_size,out_size)).long()
        for i, label in enumerate(labels):
            x, y = centroids[:, i]
            # Only write centroids if they're still within the output image bounds
            if x >=0 and x < out_size and y >= 0 and y < out_size:
                output[math.floor(x),math.floor(y)] = label+1
        
        return output

    
    def _get_edges(self, scene, graph, graphnode):
        # Get all incoming edges to a graph node
        all_graph_nodes_id = self._sort_graphnode(graph)
        # print(all_graph_nodes_id, graphnode.id)

        # Find the split point of graphnode
        num_objects = all_graph_nodes_id.index(graphnode.id)
        # pre_split_ids = set([n for n in all_graph_nodes_id[:num_objects]])
        post_split_ids = set([n for n in all_graph_nodes_id[num_objects:]])

        # Find all incoming edges to graphnode
        candidate_edges = []
        for node in graph.nodes:
            if node.id in post_split_ids:
                continue
            for e in node.out_edges:
                if e.end_node.id == graphnode.id:
                    candidate_edges.append(e)
        # print(candidate_edges)
        if not (len(candidate_edges) > 0):
            print("No candidate edgs found")
            raise
        # assert (len(candidate_edges) > 0), "No candidate edgs found"
        return candidate_edges

    def _get_pre_split_ids(self, graph, graphnode):
        # Get ids of graph node already in scenes
        # This function is seperated because it may be handle differently in pipeline
        all_graph_nodes_id = self._sort_graphnode(graph)
        num_objects = all_graph_nodes_id.index(graphnode.id)
        pre_split_ids = set([n for n in all_graph_nodes_id[:num_objects]])
        return pre_split_ids


    def _get_anchor(self, anchor_graphnode, graphnode_to_renderednode):
        anchor_loc, anchor_orient = None, None
        anchor = graphnode_to_renderednode(anchor_graphnode)
        # What we do next depends on whether this is a wall or an object
        if anchor_graphnode.is_wall:
            #### It's a wall segment
            anchor_seg = anchor["points"]
            anchor_normal = anchor["normal"]
            # Compute anchor location
            anchor_loc = torch.Tensor(anchor_seg[0] + anchor_seg[1])/2
            anchor_loc = (anchor_loc/self.composite_size - 0.5) * 2
            # Compute anchor orientation
            anchor_orient = torch.Tensor(anchor_normal)
        else:
            #### It's an object
            # Compute anchor location
            xmin, _, ymin, _ = anchor["bbox_min"]
            xmax, _, ymax, _ = anchor["bbox_max"]
            anchor_loc = torch.Tensor([(xmin+xmax)/2, (ymin+ymax)/2])
            anchor_loc = (anchor_loc/self.composite_size - 0.5) * 2
            # Compute anchor orientation
            xform = anchor["transform"]
            # cos = xform[0]
            # sin = xform[8]
            scale = (xform[0]**2+xform[8]**2)**0.5
            # sin, cos are swapped because here the first axis is y direction
            # The y direction is flipped because positive is defined as pointing up here
            sin = xform[0]/scale
            cos = -xform[8]/scale
            anchor_orient = torch.Tensor([cos, sin])
        
        # Compute anchor mask
        anchor_mask = compute_object_mask(anchor, self.composite_size, self.index)

        return anchor_loc, anchor_orient, anchor_mask

    def _sort_graphnode(self, graph):
        # Do topological sort of graph node
        import collections
        nodes = collections.deque()
        vertices = [n.id for n in graph.nodes if not n.is_arch]
        all_edges = [e for n in graph.nodes for e in n.out_edges \
                    if not e.start_node.is_arch and not e.end_node.is_arch]
        edges = collections.defaultdict(list)
        for e in all_edges:
            edges[e.start_node.id].append(e.end_node.id)
        marked = dict(zip(vertices, [0 for _ in range(len(vertices))])) # 0: not visited; 1: visited; 2: temporarily visited

        DAG = True
        def visit(v):
            nonlocal DAG
            if marked[v] == 2:
                DAG = False
            if marked[v] == 0:
                marked[v] = 2
                if v in edges:
                    for n in edges[v]:
                        visit(n)
                marked[v] = 1
                nodes.appendleft(v)
            return

        for v in vertices:
            visit(v)
        if not DAG:
            print(f"Warning: Scene {self.index} not a DAG")
        return list(nodes)

# ---------------------------------------------------------------------------------------

class FCNDatasetGraphOneScene():
    ###
    # The computation and functions are totally the same as FCNDatasetGraph2
    # The only difference is that the dataset only contains one test scene
    # And the index of the dataset is set to be the same as the order of the objects in that scnen
    ###
    def __init__(self, scene_indices=0, data_folder="bedroom", data_root_dir=None, seed=None, \
                 canvas_scale=2):
        super(FCNDatasetGraphOneScene, self).__init__()
        self.data_folder = data_folder
        self.data_root_dir = data_root_dir
        self.scene_indices = scene_indices
        self.composite_size = None
        self.canvas_scale = canvas_scale    # How much to scale the input/target canvas

        self.graph = self.get_graph(scene_indices)
        self.all_graph_nodes_id = self._sort_graphnode(self.graph)

        self.count = 0


    def __len__(self):
        return len(self.all_graph_nodes_id)

    def get_scene(self, index):
        scene = RenderedScene(index, self.data_folder, self.data_root_dir)
        if not self.composite_size:
            self.composite_size = scene.size
        return scene

    def get_graph(self, index):
        return RelationshipGraph().load(index, self.data_folder, self.data_root_dir)

    def __getitem__(self, index):
        self.index = index
        ### At test time a graphnode is choosen to be added into the scene
        # 1. Pick a graphnode to insert
        # 2. Get all incoming edge to that node
        # 3. All start node of candidate edge is used as anchor
        # 4. Transform the input according to the anchor
        scene = self.get_scene(self.scene_indices)
        graph = self.graph

        composite = scene.create_composite()

        all_nodes = scene.object_nodes + scene.door_window_nodes
        def graphnode_to_renderednode(graph_node):
            if graph_node.is_wall:
                # Grab the rendered scene segment that corresponds to this graphnode
                segments = scene.wall_segments
                # Assumes that wall ids are of the form 'wall_n' for index 
                seg_index = int(graph_node.id.split('_')[1])
                return segments[seg_index]
            else:
                # The rendered scene node["id"] field corresponds to the graph node id property
                return utils.find(lambda n: n['id'] == graph_node.id, all_nodes)
        
        graphnode = [n for n in graph.nodes if n.id == self.all_graph_nodes_id[index]][0]
        scenenode = graphnode_to_renderednode(graphnode)
        
        candidate_edges = self._get_edges(scene, graph, graphnode)
        
        # Add back objects already in scence
        pre_split_ids = self._get_pre_split_ids(graph, graphnode)
        exist_graphnodes = [n for n in graph.nodes if n.id in pre_split_ids and not n.is_arch]
        for node in exist_graphnodes:
            scene_node = graphnode_to_renderednode(node)
            composite.add_node(scene_node)
        inputs = composite.get_composite(num_extra_channels=1)

        # Get target mask
        # For convinence, target is NOT TRANSFORMED
        target_loc, target_orient, target_mask = self._get_anchor(graphnode, graphnode_to_renderednode)
        target_direction_mask = draw_direction_mask(target_loc, target_orient, target_mask)
        target = torch.stack([target_mask, target_direction_mask], 0)

        # inputs, anchor_loc, anchor_orient, edge_type
        items = []
        #print(f"Index: {index}")
        #print(candidate_edges)
        for edge in candidate_edges:
            data = inputs.clone()
            anchor_graphnode = edge.start_node
            anchor_loc, anchor_orient, anchor_mask = self._get_anchor(anchor_graphnode, graphnode_to_renderednode)
            data[data.shape[0]-1] = anchor_mask
            ###
            # If you want to save some images, do it here
            input_depth = inverse_xform_img(data[3:4, ...], anchor_loc, anchor_orient, 1)
            save_input_local_unscaled(input_depth, edge, self.count, self.scene_indices)
            self.count += 1
            ###
            # Inverse transform the inputs to be relative to the anchor coordinate frame
            data = inverse_xform_img(data, anchor_loc, anchor_orient, self.canvas_scale)
            items.append([data, target, anchor_loc, anchor_orient, torch.Tensor([edge.edge_type.index]), torch.Tensor([scenenode["category"]])])
        
        return items
    
    def _get_output_transform(self, edges_of_type, anchor_loc, anchor_orient, parent_ids, graphnode_to_renderednode):
        # Return the transfromed output based on the selected edge and anchor
        centroids, labels = [], []
        for e in edges_of_type:
            end_node = e.end_node
            obj_node = graphnode_to_renderednode(end_node)
            if obj_node["parent"] == "Wall":
                print("Massive messup!")
            if obj_node["parent"] in parent_ids:
                xmin, _, ymin, _ = obj_node["bbox_min"]
                xmax, _, ymax, _ = obj_node["bbox_max"]
                centroids.append(((xmin+xmax)/2, (ymin+ymax)/2))
                labels.append(obj_node["category"])
        # Inverse transform centroids before writing values into 'output'
        size = self.composite_size
        out_size = self.canvas_scale * 64
        centroids = np.transpose(torch.tensor(centroids))
        centroids = (centroids/size - 0.5) * 2
        centroids = inverse_xform_pnts(centroids, anchor_loc, anchor_orient, 2)
        centroids = (centroids/2 + 0.5) * out_size
        # Construct output img
        output = torch.zeros((out_size,out_size)).long()
        for i, label in enumerate(labels):
            x, y = centroids[:, i]
            # Only write centroids if they're still within the output image bounds
            if x >=0 and x < out_size and y >= 0 and y < out_size:
                output[math.floor(x),math.floor(y)] = label+1
        
        return output

    
    def _get_edges(self, scene, graph, graphnode):
        # Get all incoming edges to a graph node
        all_graph_nodes_id = self._sort_graphnode(graph)
        # print(all_graph_nodes_id, graphnode.id)

        # Find the split point of graphnode
        num_objects = all_graph_nodes_id.index(graphnode.id)
        # pre_split_ids = set([n for n in all_graph_nodes_id[:num_objects]])
        post_split_ids = set([n for n in all_graph_nodes_id[num_objects:]])

        # Find all incoming edges to graphnode
        candidate_edges = []
        for node in graph.nodes:
            if node.id in post_split_ids:
                continue
            for e in node.out_edges:
                if e.end_node.id == graphnode.id:
                    candidate_edges.append(e)
        # print(candidate_edges)
        assert (len(candidate_edges) > 0), "No candidate edgs found"
        return candidate_edges

    def _get_pre_split_ids(self, graph, graphnode):
        # Get ids of graph node already in scenes
        # This function is seperated because it may be handle differently in pipeline
        all_graph_nodes_id = self._sort_graphnode(graph)
        num_objects = all_graph_nodes_id.index(graphnode.id)
        pre_split_ids = set([n for n in all_graph_nodes_id[:num_objects]])
        return pre_split_ids


    def _get_anchor(self, anchor_graphnode, graphnode_to_renderednode):
        anchor_loc, anchor_orient = None, None
        anchor = graphnode_to_renderednode(anchor_graphnode)
        # What we do next depends on whether this is a wall or an object
        if anchor_graphnode.is_wall:
            #### It's a wall segment
            anchor_seg = anchor["points"]
            anchor_normal = anchor["normal"]
            # Compute anchor location
            anchor_loc = torch.Tensor(anchor_seg[0] + anchor_seg[1])/2
            anchor_loc = (anchor_loc/self.composite_size - 0.5) * 2
            # Compute anchor orientation
            anchor_orient = torch.Tensor(anchor_normal)
        else:
            #### It's an object
            # Compute anchor location
            xmin, _, ymin, _ = anchor["bbox_min"]
            xmax, _, ymax, _ = anchor["bbox_max"]
            anchor_loc = torch.Tensor([(xmin+xmax)/2, (ymin+ymax)/2])
            anchor_loc = (anchor_loc/self.composite_size - 0.5) * 2
            # Compute anchor orientation
            xform = anchor["transform"]
            # cos = xform[0]
            # sin = xform[8]
            scale = (xform[0]**2+xform[8]**2)**0.5
            # sin, cos are swapped because here the first axis is y direction
            # The y direction is flipped because positive is defined as pointing up here
            sin = xform[0]/scale
            cos = -xform[8]/scale
            anchor_orient = torch.Tensor([cos, sin])
        
        # Compute anchor mask
        anchor_mask = compute_object_mask(anchor, self.composite_size, self.index)

        return anchor_loc, anchor_orient, anchor_mask

    def _sort_graphnode(self, graph):
        # Do topological sort of graph node
        import collections
        nodes = collections.deque()
        vertices = [n.id for n in graph.nodes if not n.is_arch]
        all_edges = [e for n in graph.nodes for e in n.out_edges \
                    if not e.start_node.is_arch and not e.end_node.is_arch]
        edges = collections.defaultdict(list)
        for e in all_edges:
            edges[e.start_node.id].append(e.end_node.id)
        marked = dict(zip(vertices, [0 for _ in range(len(vertices))])) # 0: not visited; 1: visited; 2: temporarily visited

        DAG = True
        def visit(v):
            nonlocal DAG
            if marked[v] == 2:
                DAG = False
            if marked[v] == 0:
                marked[v] = 2
                if v in edges:
                    for n in edges[v]:
                        visit(n)
                marked[v] = 1
                nodes.appendleft(v)
            return

        for v in vertices:
            visit(v)
        if not DAG:
            print(f"Warning: Scene {self.index} not a DAG")
        return list(nodes)

# ---------------------------------------------------------------------------------------
def draw_direction_mask(loc, orient, mask):
    # Plotting utility to draw a line from loc along orient on mask
    # Draw a [1, 0] pointing up and forward transfeer to align with mask
    # Input:
    #   loc, orient: 1D tensor of lenght 2
    #   mask: 2D tensor [H, W]
    # Output:
    #   normal_mask: 2D tensor [H, W]
    loc, orient = loc.unsqueeze(0), orient.unsqueeze(0)
    mask = mask.unsqueeze(0).unsqueeze(0)

    length = 10
    normal_mask = torch.zeros_like(mask)
    center = (mask.shape[-1] + 1) // 2
    for y in range(length):
        normal_mask[..., center-y, center] = 1.0
    normal_mask = forward_xform_img(normal_mask, loc, orient, 1).squeeze(0).squeeze(0)
    return normal_mask


def compute_object_mask(node, img_size, index=0):
    h = node["height_map"]
    xsize, ysize = h.shape
    xmin = math.floor(node["bbox_min"][0])
    ymin = math.floor(node["bbox_min"][2])
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

# pnts is a 2xn matrix of 2d points
def inverse_xform_pnts(pnts, loc, orient, scale=1):
    # First, build the inverse rotation matrix
    inv_rot_matrix = torch.eye(3, 3)
    cos, sin = orient
    inv_rot_matrix[0, 0] = cos
    inv_rot_matrix[1, 1] = cos
    inv_rot_matrix[0, 1] = -sin
    inv_rot_matrix[1, 0] = sin
    # Then, build the inverse translation matrix
    inv_trans_matrix = torch.eye(3, 3)
    inv_trans_matrix[0, 2] = -loc[0]
    inv_trans_matrix[1, 2] = -loc[1]
    # Build the scaling transform matrix
    scale_matrix = torch.eye(3, 3)
    scale_matrix[0, 0] = 1/scale
    scale_matrix[1, 1] = 1/scale
    # Multiply them to get the full affine matrix
    inv_matrix = torch.matmul(torch.matmul(scale_matrix, inv_rot_matrix), inv_trans_matrix)
    # Discard the last row
    inv_matrix = inv_matrix[0:2, :]
    # Add an extra homogenous coordinate to the points
    pnts_3d = torch.ones(3, pnts.shape[1])
    pnts_3d[0:2, :] = pnts
    # Transform and return
    pnts_out = torch.matmul(inv_matrix, pnts_3d)
    return pnts_out


def inverse_xform_img(img, loc, orient, scale=1):
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
    batch_size = img.shape[0]
    inv_rot_matrices = torch.zeros(batch_size, 3, 3)
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
    inv_trans_matrices = torch.stack(inv_trans_matrices)
    inv_trans_matrices[:, 0, 2] = -loc[:, 1]
    inv_trans_matrices[:, 1, 2] = -loc[:, 0]
    # Build scaling transform matrices
    scale_matrices = torch.stack([torch.eye(3, 3) for i in range(0, batch_size)])
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

if __name__ == "__main__":
    import torchvision

    # ######### Check FCNDataset4

    # a = FCNDataset4(data_folder="bedroom_graph_6x6", relative=True)
    # #a = LocationDatasetMultilevel()
    # data, target, loc, orient = [t.unsqueeze(0) for t in a[0]] # 907 square | 931 L-shaped | 932 non-rectilinear
    # print("Location:", loc, "Orient:", orient)
    # img = data[0][1].unsqueeze(0)
    # out_img = torchvision.transforms.ToPILImage()(img)
    # out_img.save("wall_xform.png")
    # target_img = torchvision.transforms.ToPILImage()(target)
    # target_img.save("target_xform.png")

    # img = forward_xform_img(img, loc, orient, 2)
    # restore_img = torchvision.transforms.ToPILImage()(img.squeeze(0))
    # restore_img.save("wall_restore.png")
    # target = forward_xform_img(target, loc, orient, 2).squeeze(0)
    # restore_target = torchvision.transforms.ToPILImage()(target[0].unsqueeze(0))
    # restore_target.save("target_restore.png")

    # # Check if the wall anchor masks look sensible
    # dataset = FCNDataset4(data_folder="bedroom_graph_6x6", relative=True)
    # idx = 0
    # inp, out, loc, orient = dataset[idx]
    # wall_mask = inp[1].unsqueeze(0)
    # wall_img = torchvision.transforms.ToPILImage()(wall_mask)
    # wall_img.save('wall_mask.png')
    # scene = dataset.get_scene(idx)
    # for i, seg in enumerate(scene.wall_segments):
    #     anchor_mask = compute_object_mask(seg, wall_mask.shape[1])
    #     anchor_img = torchvision.transforms.ToPILImage()(anchor_mask.unsqueeze(0))
    #     anchor_img.save(f'anchor_mask_{i}.png')


    # ########### Check FCNDatasetGraph
    
    # # # Check if we can load a graph
    # # graph = RelationshipGraph().load(8, 'bedroom_graph_6x6')
    # # print(graph)

    # Check if the wall segments are objects or just endpoint arrays
    scene = RenderedScene(907, 'bedroom_graph_6x6')
    #print([w["points"] for w in scene.wall_segments])

    # # Check if we can load a dataset item that includes graphs
    idx = 5000

    # dataset = FCNDataset4(data_folder="bedroom_graph_6x6", seed = 42)
    # data, target, loc, orient = [t.unsqueeze(0) for t in dataset[idx]]
    train_size = 6000
    data_folder = 'toilet_graph_6x6'
    data_root_dir = utils.get_data_root_dir()

    dataset = FCNDatasetGraph(scene_indices = (0, train_size), data_folder=data_folder, data_root_dir=data_root_dir)
    data, target, loc, orient, edge_type = [t.unsqueeze(0) for t in dataset[idx]]

    print("Edge type:", edge_type[0])
    print("Location:", loc[0], "Orient:", orient[0])
    # print(edge_type)
    img = data[0][1].unsqueeze(0)
    anchor_mask = data[0][-1].unsqueeze(0)
    img += anchor_mask
    
    img_mask = utils.nearest_downsample(img.clone().unsqueeze(0), 2)

    out_img = torchvision.transforms.ToPILImage()(img)
    out_img.save("anchor_xform.png")
    img = forward_xform_img(img, loc, orient, 2)
    restore_img = torchvision.transforms.ToPILImage()(img.squeeze(0))
    restore_img.save("anchor_restore.png")
    anchor_img = torchvision.transforms.ToPILImage()(anchor_mask)
    anchor_img.save("anchor_mask.png")

    target = target.float() + img_mask.squeeze(0)

    target_img = torchvision.transforms.ToPILImage()(target)
    target_img.save("target_xform.png")
    target = forward_xform_img(target, loc, orient, 2).squeeze(0)
    restore_target = torchvision.transforms.ToPILImage()(target[0].unsqueeze(0))
    restore_target.save("target_restore.png")

    # ########### Check FCNDatasetGraph2
    # dataset = FCNDatasetGraph2(data_folder="bedroom_graph_6x6_newer", seed=1)
    # data, loc, orient, edge_type = [t.unsqueeze(0) for t in dataset[idx][0]]

    # print("Edge type:", edge_type)
    # print("Location:", loc[0], "Orient:", orient[0])
    # # print(edge_type)
    # img = data[0][1].unsqueeze(0)
    # anchor_mask = data[0][-1].unsqueeze(0)
    # img += anchor_mask
    # out_img = torchvision.transforms.ToPILImage()(img)
    # out_img.save("anchor_xform.png")
    # img = forward_xform_img(img, loc, orient, 2)
    # restore_img = torchvision.transforms.ToPILImage()(img.squeeze(0))
    # restore_img.save("anchor_restore.png")
    # anchor_img = torchvision.transforms.ToPILImage()(anchor_mask)
    # anchor_img.save("anchor_mask.png")
    
