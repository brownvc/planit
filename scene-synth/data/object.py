import pickle
import os
import numpy as np
from data import ObjectData
import utils
from math_utils.geometry_helpers import ObjRay

"""
Taking care of wavefront obj files
Convert to pickle for faster loading
Currently just geometric information.
Call this file once to create a pickled version of the objects
For faster loading in the future
"""

class Obj():
    """
    Standard vertex-face representation, triangulated
    Order: x, z, y
    """
    object_data = ObjectData()

    def __init__(self, modelId=None, houseId=None, from_source=False, is_room=False, mirror=False, wall=None):
        """
        Parameters
        ----------
        modelId (string): name of the object to be loaded
        houseId (string, optional): If loading a room, specify which house does the room belong to
        from_source (bool, optional): If false, loads the pickled version of the object
            need to call object.py once to create the pickled version.
            does not apply for rooms
        mirror (bool, optional): If true, loads the mirroed version
        """

        if wall:
            self.vertices = wall.vertices
            self.faces = [[0,1,2],[1,2,3],[4,5,6],[5,6,7],[0,1,4],[1,4,5],
                          [0,2,4],[2,4,6],[1,3,5],[3,5,7],[2,3,6],[3,6,7]]
        else:
            if is_room: from_source = True  #Don't want to save rooms...
            data_dir = utils.get_data_root_dir()
            self.vertices = []
            self.faces = []
            if from_source:
                if is_room:
                    path = f"{data_dir}/data/room/{houseId}/{modelId}.obj"
                else:
                    path = f"{data_dir}/data/object/{modelId}/{modelId}.obj"
                with open(path,"r") as f:
                    for line in f:
                        data = line.split()
                        if len(data) > 0:   
                            if data[0] == "v":
                                v = np.asarray([float(i) for i in data[1:4]]+[1])
                                self.vertices.append(v)
                            if data[0] == "f":
                                face = [int(i.split("/")[0])-1 for i in data[1:]]
                                if len(face) == 4:
                                    self.faces.append([face[0],face[1],face[2]])
                                    self.faces.append([face[0],face[2],face[3]])
                                elif len(face) == 3:
                                    self.faces.append([face[0],face[1],face[2]])
                                else:
                                    print(f"Found a face with {len(face)} edges!!!")

                self.vertices = np.asarray(self.vertices)
                if not is_room and Obj.object_data.get_alignment_matrix(modelId) is not None:
                    self.vertices = np.dot(self.vertices, Obj.object_data.get_alignment_matrix(modelId))
                    #self.transform(Obj.object_data.get_alignment_matrix(modelId))
            else:
                with open(f"{data_dir}/object/{modelId}/vertices.pkl", "rb") as f:
                    self.vertices = pickle.load(f)
                with open(f"{data_dir}/object/{modelId}/faces.pkl", "rb") as f:
                    self.faces = pickle.load(f)

        self.bbox_min = np.min(self.vertices, 0)
        self.bbox_max = np.max(self.vertices, 0)

        if mirror:
            t = np.asarray([[-1, 0, 0, 0], \
                            [0, 1, 0, 0], \
                            [0, 0, 1, 0], \
                            [0, 0, 0, 1]])
            self.vertices = np.dot(self.vertices, t)
            #self.transform(t)
            self.modelId = modelId+"_mirror"
        else:
            self.modelId = modelId

        # [relation-extraction] save objectspace corners (top-down 2d)
        self.back_left = np.array([self.xmin(), 0, self.ymin(), 1])
        self.front_left = np.array([self.xmin(), 0, self.ymax(), 1])
        self.back_right = np.array([self.xmax(), 0, self.ymin(), 1])
        self.front_right = np.array([self.xmax(), 0, self.ymax(), 1])

    # [relation-extraction] get the 4 lines that make up the object-space bbox 
    # with the transformations of this obj applied to them
    def bbox_lines(self):
        return [(self.front_left, self.back_left), \
            (self.back_left, self.back_right), \
            (self.back_right, self.front_right), \
            (self.front_right, self.front_left)]

    # [relation-extraction] getters for the rays in each of the 4 cardinal directions
    def front_rays(self):
        direction = (self.front_left - self.back_left)
        return [ObjRay(self.front_left, direction), ObjRay(self.front_right, direction)]

    def back_rays(self):
        direction = (self.back_left - self.front_left)
        return [ObjRay(self.back_left, direction), ObjRay(self.back_right, direction)]

    def left_rays(self):
        direction = (self.front_left - self.front_right)
        return [ObjRay(self.front_left, direction), ObjRay(self.back_left, direction)]

    def right_rays(self):
        direction = (self.front_right - self.front_left)
        return [ObjRay(self.front_right, direction), ObjRay(self.back_right, direction)]


    def save(self):
        data_dir = utils.get_data_root_dir()
        dest_dir = f"{data_dir}/object/{self.modelId}"
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        with open(f"{dest_dir}/vertices.pkl", "wb") as f:
            pickle.dump(self.vertices, f, pickle.HIGHEST_PROTOCOL)
        with open(f"{dest_dir}/faces.pkl", "wb") as f:
            pickle.dump(self.faces, f, pickle.HIGHEST_PROTOCOL)
                
    
    def transform(self, t):
        self.vertices = np.dot(self.vertices, t)
        
        # [relation-extraction] transform the 4 corner points of the obj-space bbox separately
        # because these will not necessarily be vertices of the object.
        self.front_left = np.dot(self.front_left, t)
        self.back_left = np.dot(self.back_left, t)
        self.front_right = np.dot(self.front_right, t)
        self.back_right = np.dot(self.back_right, t)
        
        return self

    
    def get_triangles(self):
        for face in self.faces:
            yield (self.vertices[face[0]][:3], \
                   self.vertices[face[1]][:3], \
                   self.vertices[face[2]][:3],)
    
    def xmax(self):
        return np.amax(self.vertices, axis = 0)[0]

    def xmin(self):
        return np.amin(self.vertices, axis = 0)[0]

    def ymax(self):
        return np.amax(self.vertices, axis = 0)[2]

    def ymin(self):
        return np.amin(self.vertices, axis = 0)[2]

    def zmax(self):
        return np.amax(self.vertices, axis = 0)[1]

    def zmin(self):
        return np.amin(self.vertices, axis = 0)[1]

def parse_objects():
    """
    parse .obj objects and save them to pickle files
    """
    data_dir = utils.get_data_root_dir()
    obj_dir = data_dir + "/data/object/"
    print("Parsing some object files...")
    l = len(os.listdir(obj_dir))
    for (i, modelId) in enumerate(os.listdir(obj_dir)):
        print(f"{i+1} of {l}...", end="\r")
        if not modelId in ["cube", ".DS_Store"]:
            o = Obj(modelId, from_source = True)
            o.save()
            o = Obj(modelId, from_source = True, mirror = True)
            o.save()
    print()

if __name__ == "__main__":
    parse_objects()



