"""
Three level House-Level-Node/Room representation of whatever
"""
import os
import json
import numpy as np
from data import ObjectData
import utils

class House():
    """
    Represents a House
    describing a house
    """
    object_data = ObjectData()
    def __init__(self, index=0, id_=None, house_json=None, file_dir=None,
                 include_support_information=True, include_arch_information=True):
        """
        Get a set of rooms from the house which satisfies a certain criteria
        Parameters
        ----------
        index (int): The index of the house among all houses sorted in alphabetical order
            default way of loading a house
        id_ (string, optional): If set, then the house with the specified directory name is chosen
        house_json(json, optional): If set, then the specified json object
            is used directly to initiate the house
        file_dir (string, optional): If set, then the json pointed to by file_dir will be loaded
        include_support_information(bool): If true, then support information is loaded from data/house_relations
            might not be available, so defaults to False
        include_arch_information (bool): If true, then arch information is loaded from data/wall
            might not be available, so defaults to False
        """
        data_dir = utils.get_data_root_dir()
        if house_json is None:
            if file_dir is None:
                house_dir = data_dir + "/data/house/"

                if id_ is None:
                    houses = dict(enumerate(os.listdir(house_dir)))
                    self.__dict__ = json.loads(open(house_dir+houses[index]+"/house.json", 'r').read())
                else:
                    self.__dict__ = json.loads(open(house_dir+id_+"/house.json", 'r').read())
            else:
                self.__dict__ = json.loads(open(file_dir, 'r').read())
        else:
            self.__dict__ = house_json

        self.filters = []
        self.levels = [Level(l,self) for l in self.levels]
        self.rooms = [r for l in self.levels for r in l.rooms]
        self.nodes = [n for l in self.levels for n in l.nodes]
        self.node_dict = {id_: n for l in self.levels for id_,n in l.node_dict.items()}
        if include_support_information:
            house_stats_dir = data_dir + "/data/house_relations/"
            stats = json.loads(open(house_stats_dir+self.id+"/"+self.id+".stats.json", 'r').read())
            supports = [(s["parent"],s["child"]) for s in stats["relations"]["support"]]
            for parent, child in supports:
                if child not in self.node_dict:
                    print(f'Warning: support relation {supports} involves not present {child} node')
                    continue
                if "f" in parent:
                    self.get_node(child).parent = "Floor"
                elif "c" in parent:
                    self.get_node(child).parent = "Ceiling"
                elif len(parent.split("_")) > 2:
                    self.get_node(child).parent = "Wall"
                else:
                    if parent not in self.node_dict:
                        print(f'Warning: support relation {supports} involves not present {parent} node')
                        continue
                    self.get_node(parent).child.append(self.get_node(child))
                    self.get_node(child).parent = self.get_node(parent)
        if include_arch_information:
            house_arch_dir = data_dir + '/data/wall/'
            arch = json.loads(open(house_arch_dir+self.id+'/'+self.id+'.arch.json', 'r').read())
            default_depth = arch["defaults"]["Wall"]["depth"]
            extra_height = arch["defaults"]["Wall"]["extraHeight"]
            self.walls = [w for w in arch['elements'] if w['type'] == 'Wall']
            
            r_dict = {r.original_id:r for r in self.rooms}
            for wall in self.walls:
                if "depth" not in wall:
                    wall["depth"] = default_depth
                wall["height"] += extra_height
                rid = wall['roomId']
                if rid in r_dict:
                    r_dict[rid].walls.append(Wall(wall, len(r_dict[rid].walls)))

            def wall_adjacent(wa,wb):
                def f(pa, pb):
                    return np.linalg.norm(np.asarray(pa)-np.asarray(pb)) < 1e-3
                return f(wa.points[0], wb.points[0]) \
                    or f(wa.points[0], wb.points[1]) \
                    or f(wa.points[1], wb.points[0]) \
                    or f(wa.points[1], wb.points[1])

            for r in self.rooms:
                N = len(r.walls)
                for i in range(N):
                    for j in range(i+1,N):
                        if wall_adjacent(r.walls[i], r.walls[j]):
                            r.walls[i].adjacent.append(j)
                            r.walls[j].adjacent.append(i)
                
                if len(r.walls) > 2:
                    r.closed_wall = True
                else:
                    r.closed_wall = False

                if r.closed_wall:
                    for wall in r.walls:
                        if len(wall.adjacent) != 2:
                            r.closed_wall = False
        
                if r.closed_wall:
                    visited = [0]
                    cur = 0
                    while True:
                        if r.walls[cur].adjacent[0] not in visited:
                            cur = r.walls[cur].adjacent[0]
                            visited.append(cur)
                        elif r.walls[cur].adjacent[1] not in visited:
                            cur = r.walls[cur].adjacent[1]
                            visited.append(cur)
                        else:
                            break
                    if len(visited) < len(r.walls):
                        r.closed_wall = False

                    #print(len(visited), len(r.walls))
                   
                #print(r.closed_wall)
                
            #print(self.rooms[0].walls)
            #for r in self.rooms:
            #    print(r.id)
    
    def get_node(self, id_):
        return self.node_dict[id_]

    def has_node(self, id_):
        return id_ in self.node_dict

    def get_rooms(self, filters=None):
        """
        Get a set of rooms from the house which satisfies a certain criteria
        Parameters
        ----------
        filters (list[room_filter]): room_filter is tuple[Room,House] which returns
            if the Room should be included
        
        Returns
        -------
        list[Room]
        """
        if filters is None: filters = self.filters
        if not isinstance(filters, list): filters = [filters]
        rooms = self.rooms
        for filter_ in filters:
            rooms = [room for room in rooms if filter_(room, self)]
        return rooms
    
    def filter_rooms(self, filters):
        """
        Similar to get_rooms, but overwrites self.node instead of returning a list
        """
        self.rooms = self.get_rooms(filters)
    
    def trim(self):
        """
        Get rid of some intermediate attributes
        """
        nodes = list(self.node_dict.values())
        if hasattr(self, 'rooms'):
            nodes.extend(self.rooms)
        for n in nodes:
            for attr in ['xform', 'obb', 'frame', 'model2world']:
                if hasattr(n, attr):
                    delattr(n, attr)
        self.nodes = None
        self.walls = None
        for room in self.rooms:
            room.filters = None
        self.levels = None
        self.node_dict = None
        self.filters = None


class Wall():
    def __init__(self, dict_, index):
        self.__dict__ = dict_
        self.index = index
        self.original_id = self.id
        self.id = f"wall_{index}"
        #print(self.__dict__)
        self.adjacent = []

        (xmin, zmin, ymin) = self.points[0]
        (xmax, zmax, ymax) = self.points[1]
        
        length_delta = np.asarray([xmax-xmin, ymax-ymin])
        length_delta = length_delta / np.linalg.norm(length_delta) * self.depth * 0.4
        (dx, dy) = length_delta
        xmin -= dx
        ymin -= dy
        xmax += dx
        ymax += dy

        zmax += self.height

        depth_delta = np.asarray([ymin-ymax, xmax-xmin])
        depth_delta = depth_delta / np.linalg.norm(depth_delta)
        depth_delta = depth_delta * self.depth / 2
        (dx, dy) = depth_delta
        xmin -= dx*2/19
        ymin -= dy*2/19
        xmax -= dx*2/19
        ymax -= dy*2/19

        
        self.vertices = [[xmin-dx, zmin, ymin-dy,1],
                         [xmin+dx, zmin, ymin+dy,1],
                         [xmax-dx, zmin, ymax-dy,1],
                         [xmax+dx, zmin, ymax+dy,1],
                         [xmin-dx, zmax, ymin-dy,1],
                         [xmin+dx, zmax, ymin+dy,1],
                         [xmax-dx, zmax, ymax-dy,1],
                         [xmax+dx, zmax, ymax+dy,1]]
        self.vertices = np.asarray(self.vertices)
        
        dx, dy = abs(dx), abs(dy)
        #print(self.vertices)
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        xmin -= dx
        xmax += dx
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        ymin -= dy
        ymax += dy
        #print(point_min)
        #print(xmin, zmin, ymin)
        #print(point_max)
        #print(xmax, zmax, ymax)
        #print("____")

        self.xmin, self.ymin, self.zmin = xmin, ymin, zmin
        self.xmax, self.ymax, self.zmax = xmax, ymax, zmax

    @property
    def pts(self):
        if not hasattr(self, '_pts'):
            self._pts = np.asarray(self.points)
        return self._pts

    @property
    def length(self):
        return np.linalg.norm(self.pts[0] - self.pts[1])

    @classmethod
    def compute_transform_from_points(cls, pts):
        pos = (pts[0] + pts[1]) / 2
        orient = pts[1] - pts[0]
        orient = np.array([orient[0], orient[2]])
        orient /= np.linalg.norm(orient)
        cos, sin = orient
        tx, ty, tz = pos[0:3]
        # Rotation about y
        xform = np.array([[cos,  0, -sin, tx],
                          [0,    1, 0,   ty],
                          [sin, 0, cos, tz],
                          [0,    0, 0,   1]])
        # Convert to format appropriate for post-multiplicaton
        # i.e. np.dot(v, xform)
        return np.transpose(xform)

    # Get the object -> world space transform of this wall
    # NOTE: This assumes that the wall is part of a CCW loop
    @property
    def transform(self):
        return Wall.compute_transform_from_points(self.pts)

    # Get the inward-facing normal for the wall
    # NOTE: This assumes that the wall is part of a CCW loop
    @property
    def normal(self):
        p0 = np.array([self.pts[0][0], self.pts[0][2]])
        p1 = np.array([self.pts[1][0], self.pts[1][2]])
        par = p1 - p0
        perp = np.array([par[1], -par[0]])
        # Orientation check
        if np.dot((p1+perp) - p0, perp) < 0:
            perp = -perp
        return perp / np.linalg.norm(perp)
        

class Level():
    """
    Represents a floor level in the house
    Currently mostly just used to parse the list of nodes and rooms
    Might change in the future
    """
    def __init__(self, dict_, house):
        self.__dict__ = dict_
        self.house = house
        invalid_nodes = [n["id"] for n in self.nodes if (not n["valid"]) and "id" in n]
        self.nodes = [Node(n,self) for n in self.nodes if n["valid"]]
        self.node_dict = {n.id: n for n in self.nodes}
        self.nodes = list(self.node_dict.values())  # deduplicate nodes with same id
        self.rooms = [Room(n, ([self.node_dict[i] for i in [f"{self.id}_{j}" \
                      for j in list(set(n.nodeIndices))] if i not in invalid_nodes]), self) \
                      for n in self.nodes if n.isRoom() and hasattr(n, 'nodeIndices')]

class Room():
    """
    Represents a room in the house
    """
    def __init__(self, room, nodes, level):
        self.__dict__ = room.__dict__
        #self.room = room
        self.nodes = nodes
        #self.level = level
        self.filters = []
        self.house_id = level.house.id
        self.original_id = room.modelId.replace('fr_', '').replace('rm', '')
        self.walls = []
    
    def get_nodes(self, filters=None):
        """
        Get a set of nodes from the room which satisfies a certain criteria
        Parameters
        ----------
        filters (list[node_filter]): node_filter is tuple[Node,Room] which returns
            if the Node should be included
        
        Returns
        -------
        list[Node]
        """
        if filters is None: filters = self.filters
        if not isinstance(filters, list): filters = [filters]
        nodes = self.nodes
        for filter_ in filters:
            nodes = [node for node in nodes if filter_(node, self)]
        return nodes
    
    def filter_nodes(self, filters):
        """
        Similar to get_nodes, but overwrites self.node instead of returning a list
        """
        self.nodes = self.get_nodes(filters)

class Node():
    """
    Basic unit of representation of whatever
    Usually a room or an object
    """
    warning = True
    def __init__(self, dict_, level):
        self.__dict__ = dict_
        #self.level = level
        self.parent = None
        self.child = []
        if hasattr(self, 'bbox'):
            (self.xmin, self.zmin, self.ymin) = self.bbox["min"]
            (self.xmax, self.zmax, self.ymax) = self.bbox["max"]
            (self.width, self.length) = sorted([self.xmax - self.xmin, self.ymax - self.ymin])
            self.height = self.zmax - self.zmin
        else:  # warn and populate with default bbox values
            if self.warning:
                print(f'Warning: node id={self.id} is valid but has no bbox, setting default values')
            (self.xmin, self.zmin, self.ymin) = (0, 0, 0)
            (self.xmax, self.zmax, self.ymax) = (0, 0, 0)
        if hasattr(self, 'transform') and hasattr(self, 'modelId'):
            t = np.asarray(self.transform).reshape(4,4)
            #Special cases of models with were not aligned in the way we want
            alignment_matrix = House.object_data.get_alignment_matrix(self.modelId)
            if alignment_matrix is not None:
                t = np.dot(np.linalg.inv(alignment_matrix), t)
                self.transform = list(t.flatten())
                
            #If a reflection is present, switch to the mirrored model
            #And adjust transform accordingly
            if np.linalg.det(t) < 0:
                t_reflec = np.asarray([[-1, 0, 0, 0], \
                                      [0, 1, 0, 0], \
                                      [0, 0, 1, 0], \
                                      [0, 0, 0, 1]])
                t = np.dot(t_reflec, t)
                self.modelId += "_mirror"
                self.transform = list(t.flatten())

    @property
    def has_valid_bbox(self):
        dx = abs(self.xmax - self.xmin)
        dy = abs(self.ymax - self.ymin)
        dz = abs(self.zmax - self.zmin)
        return dx > 0 or dy > 0 or dz > 0

    def isRoom(self):
        return self.type == "Room"

if __name__ == "__main__":
    a = House(id_ = "f53c0878c4db848bfa43163473b74245")
    r = a.rooms[0]
    size = 512
    print(len(r.walls))
    print(r.closed_wall)

    from data import *
    import scipy.misc as m
    pgen = ProjectionGenerator(img_size = size)
    projection = pgen.get_projection(r)
    t = projection.to_2d()
    #Render the floor 
    composite = np.zeros((size,size))
    for wall in r.walls:
        o = Obj(wall=wall)
        o.transform(t)
        rendered = TopDownView.render_object_full_size(o,size)
        composite = composite + rendered
    img = composite
    img = m.toimage(img, cmin=0, cmax=1)
    img.show()
    #print(a.walls)
    #for room in a.rooms:
    #    if room.id == "0_7":
    #        print(room.__dict__)
