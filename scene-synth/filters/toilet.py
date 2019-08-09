from data.house import *
from data.dataset import DatasetFilter
from data.object_data import ObjectCategories
from .global_category_filter import *
import utils

"""
Verions:
set1: baseline
"""

def toilet_filter(version, room_size, second_tier, source):
    data_dir = utils.get_data_root_dir()
    with open(f"{data_dir}/{source}/coarse_categories_frequency", "r") as f:
        coarse_categories_frequency = ([s[:-1] for s in f.readlines()])
        coarse_categories_frequency = [s.split(" ") for s in coarse_categories_frequency]
        coarse_categories_frequency = dict([(a,int(b)) for (a,b) in coarse_categories_frequency])
    category_map = ObjectCategories()
    if version == "set1":
        rejected_object_categories = ["bathroom_stuff", "partition", "kitchenware", \
                                      "sofa", "kitchen_appliance", "column", "bed", \
                                      "table", "hanger"]
        filtered_object_categories = ["person", "pet", "rug", "curtain", "trash_can", \
                                      "toy", "mirror", "picture_frame"]
        no_count_object_categories = category_map.all_arch_categories()
        freq_threshold = 1000

        def node_criteria(node, room):
            coarse_category = category_map.get_coarse_category(node.modelId)
            #Recursively check if all children are filtered out or not
            #This is done because if there is a node that we do not want
            #But it has a child that isn't filtered out
            #Then we have to reject the room completely, see room_criteria
            if node.child:
                if any(node_criteria(c, room) for c in node.child):
                    return True
            #Still nodes without parent, so can't just check if floor parent
            if node.parent and node.parent !="Floor" and \
                not coarse_category in no_count_object_categories:
                return False
            if coarse_category in rejected_object_categories: return True
            if coarse_categories_frequency[coarse_category] < freq_threshold: return False
            if coarse_category in filtered_object_categories: return False
            if node.length * node.width < 0.05: return False
            return True

        def room_criteria(room, house):
            if room.height > 4: return False
            if room.length > 6: return False
            if room.width > 6: return False
            true_node_count = 0
            category_dict = {}
            for node in room.nodes:
                coarse_category = category_map.get_coarse_category(node.modelId)
                if coarse_categories_frequency[coarse_category] < freq_threshold: return False
                if coarse_category in rejected_object_categories: return False
                if coarse_category in filtered_object_categories: return False
                if not coarse_category in no_count_object_categories:
                    if node.zmin - room.zmin > 0.1:
                        return False
                    category_dict[coarse_category] = category_dict.get(coarse_category,0) + 1
                    true_node_count += 1

            if true_node_count < 4 or true_node_count > 20: return False
            
            if not os.path.isfile(f"./data/data/room/{room.house_id}/{room.modelId}f.obj"):
                return False
            if not os.path.isfile(f"./data/data/room/{room.house_id}/{room.modelId}w.obj"):
                return False

            return True

    elif version == "final":
        filtered, rejected, door_window = GlobalCategoryFilter.get_filter()
        rejected += ["short_kitchen_cabinet", "coffee_table", "straight_chair"]
        with open(f"./data/{source}/final_categories_frequency", "r") as f:
            frequency = ([s[:-1] for s in f.readlines()])
            frequency = [s.split(" ") for s in frequency]
            frequency = dict([(a,int(b)) for (a,b) in frequency])

        def node_criteria(node, room):
            category = category_map.get_final_category(node.modelId)
            if category in filtered: return False
            return True

        def room_criteria(room, house):
            node_count = 0
            toilet_count = 0
            bathtub_count = 0
            for node in room.nodes:
                category = category_map.get_final_category(node.modelId)
                if category in rejected:
                    return False
                if not category in door_window:
                    node_count += 1

                    t = np.asarray(node.transform).reshape((4,4)).transpose()
                    a = t[0][0]
                    b = t[0][2]
                    c = t[2][0]
                    d = t[2][2]
                    
                    xscale = (a**2 + c**2)**0.5
                    yscale = (b**2 + d**2)**0.5
                    zscale = t[1][1]

                    if not 0.8<xscale<1.2:
                        return False
                    if not 0.8<yscale<1.2:
                        return False
                    if not 0.8<zscale<1.2:
                        return False

                if category == "toilet":
                    toilet_count += 1
                if category == "bathtub":
                    bathtub_count += 1

                if frequency[category] < 200: return False

            if node_count < 3 or node_count > 10: return False
            if toilet_count > 2: return False
            if bathtub_count > 3: return False

            return True

    elif version == "latent":
        filtered, rejected, door_window, second_tier_include = \
            GlobalCategoryFilter.get_filter_latent()
        rejected += ["desk", "television", "towel_rack", "sofa"]
        filtered, rejected, door_window, second_tier_include = \
            set(filtered), set(rejected), set(door_window), set(second_tier_include)
        with open(f"{data_dir}/{source}/final_categories_frequency", "r") as f:
            frequency = ([s[:-1] for s in f.readlines()])
            frequency = [s.split(" ") for s in frequency]
            frequency = dict([(a,int(b)) for (a,b) in frequency])

        def node_criteria(node, room):
            category = category_map.get_final_category(node.modelId)
            if category in door_window:
                return True
            if category in filtered: return False

            if second_tier:
                if node.zmin - room.zmin > 0.1 and \
                    (category not in second_tier_include or node.parent is None):
                        return False

                if node.parent:
                    if isinstance(node.parent, Node) and node.zmin < node.parent.zmax-0.1:
                        return False
                    node_now = node
                    while isinstance(node_now, Node) and node_now.parent:
                        node_now = node_now.parent
                    if node_now != "Floor":
                        return False
            else:
                if node.zmin - room.zmin > 0.1:
                    return False

            #Quick filter for second-tier non ceiling mount
            #if node.zmin - room.zmin < 0.1:
            #    return False
            #else:
            #    if node.zmax - room.zmax > -0.2:
            #        return False
            return True

        def room_criteria(room, house):
            if not room.closed_wall: return False
            if room.height > 4: return False
            if room.length > room_size: return False
            if room.width > room_size: return False
            floor_node_count = 0
            node_count = 0
            scaled = False
            #dirty fix!
            for i in range(5):
                room.nodes = [node for node in room.nodes if not \
                                ((node.parent and isinstance(node.parent, Node) and \
                                    (node.parent) not in room.nodes))
                             ]
            for node in room.nodes:
                category = category_map.get_final_category(node.modelId)
                if category in rejected:
                    return False
                if not category in door_window:
                    node_count += 1

                    if node.zmin - room.zmin < 0.1:
                        floor_node_count += 1

                    t = np.asarray(node.transform).reshape((4,4)).transpose()
                    a = t[0][0]
                    b = t[0][2]
                    c = t[2][0]
                    d = t[2][2]
                    
                    xscale = (a**2 + c**2)**0.5
                    yscale = (b**2 + d**2)**0.5
                    zscale = t[1][1]
                    
                    if not 0.9<xscale<1.1: #Reject rooms where any object is scaled by too much
                        return False
                    if not 0.9<yscale<1.1:
                        return False
                    if not 0.9<zscale<1.1:
                        return False

                    #if not 0.99<xscale<1.01: 
                    #    scaled = True
                    #if not 0.99<yscale<1.01:
                    #    scaled = True
                    #if not 0.99<zscale<1.01:
                    #    scaled = True

                    t[0][0] /= xscale
                    t[0][2] /= yscale
                    t[2][0] /= xscale
                    t[2][2] /= yscale
                    t[1][1] /= zscale
                    node.transform = list(t.transpose().flatten())

                if frequency[category] < 500: return False
            
            #if not scaled:
                #return False
            if floor_node_count < 4 or node_count > 20: return False

            return True
    else:
        raise NotImplementedError

    dataset_f = DatasetFilter(room_filters = [room_criteria], node_filters = [node_criteria])

    return dataset_f
