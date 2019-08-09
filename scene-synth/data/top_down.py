from data import Obj, ProjectionGenerator, Projection, Node, ObjectCategories
import numpy as np
import math
import scipy.misc as m
from numba import jit
import torch

# For rendering OBBS
from math_utils.OBB import OBB
from math_utils import Transform

class TopDownView():
    """
    Take a room, pre-render top-down views
    Of floor, walls and individual objects
    That can be used to generate the multi-channel views used in our pipeline
    """
    #Padding to avoid problems with boundary cases
    def __init__(self, height_cap=4.05, length_cap=6.05, size=512):
        #Padding by 0.05m to avoid problems with boundary cases
        """
        Parameters
        ----------
        height_cap (int): the maximum height (in meters) of rooms allowed, which will be rendered with
            a value of 1 in the depth channel. To separate the floor from empty spaces,
            floors will have a height of 0.5m. See zpad below
        length_cap (int): the maximum length/width of rooms allowed.
        size (int): size of the rendered top-down image
        Returns
        -------
        visualization (Image): a visualization of the rendered room, which is
            simply the superimposition of all the rendered parts
        (floor, wall, nodes) (Triple[torch.Tensor, torch.Tensor, list[torch.Tensor]):
            rendered invidiual parts of the room, as 2D torch tensors
            this is the part used by the pipeline
        """
        self.size = size
        self.pgen = ProjectionGenerator(room_size_cap=(length_cap, height_cap, length_cap), \
                                        zpad=0.5, img_size=size)

    def render(self, room):
        projection = self.pgen.get_projection(room)
        
        visualization = np.zeros((self.size,self.size))
        nodes = []

        for node in room.nodes:
            modelId = node.modelId #Camelcase due to original json

            t = np.asarray(node.transform).reshape(4,4)

            o = Obj(modelId)
            t = projection.to_2d(t)
            o.transform(t)
            
            save_t = t
            t = projection.to_2d()

            if node.has_valid_bbox:
                print('using cached bbox')
                bbox_min = np.dot(np.asarray([node.xmin, node.zmin, node.ymin, 1]), t)
                bbox_max = np.dot(np.asarray([node.xmax, node.zmax, node.ymax, 1]), t)
            else:
                print('using computed bbox')
                bbox_min = np.asarray([o.xmin(), o.zmin(), o.ymin(), 1])
                bbox_max = np.asarray([o.xmax(), o.zmax(), o.ymax(), 1])
            xmin = math.floor(bbox_min[0])
            ymin = math.floor(bbox_min[2])
            xsize = math.ceil(bbox_max[0]) - xmin + 1
            ysize = math.ceil(bbox_max[2]) - ymin + 1

            description = {}
            description["modelId"] = modelId
            description["transform"] = node.transform
            description["bbox_min"] = bbox_min
            description["bbox_max"] = bbox_max
            description["id"] = node.id
            description["child"] = [c.id for c in node.child] if node.child else None
            description["parent"] = node.parent.id if isinstance(node.parent, Node) else node.parent
            #if description["parent"] is None or description["parent"] == "Ceiling":
            #    print(description["modelId"])
            #    print(description["parent"])
            #    print(node.zmin - room.zmin)
                #print("FATAL ERROR")
            
            #Since it is possible that the bounding box information of a room
            #Was calculated without some doors/windows
            #We need to handle these cases
            #if ymin < 0: 
            #    ysize += ymin
            #    ymin = 0
            #if xmin < 0: 
            #    xsize += xmin
            #    xmin = 0
            #if ymin + ysize > self.size:
            #    ysize = self.size - ymin
            #if xmin + xsize > self.size:
            #    xsize = self.size - xmin

            #if xsize == 0:
            #    xmin = 0
            #    xsize = 256
            #if ysize == 0:
            #    ymin = 0
            #    ysize = 256

            xmin = 0
            ymin = 0
            xsize = 256
            ysize = 256
            
            #print(list(bbox_min), list(bbox_max))
            #print(xmin, ymin, xsize, ysize)
            rendered = self.render_object(o, xmin, ymin, xsize, ysize, self.size)
            description["height_map"] = torch.from_numpy(rendered).float()

            tmp = np.zeros((self.size, self.size))
            tmp[xmin:xmin+rendered.shape[0],ymin:ymin+rendered.shape[1]] = rendered
            visualization += tmp

            ## Compute the pixel-space dimensions of the object before it has been
            ##    transformed (i.e. in object space)
            #objspace_bbox_min = np.dot(o.bbox_min, t)
            #objspace_bbox_max = np.dot(o.bbox_max, t)
            #description['objspace_dims'] = np.array([
            #    objspace_bbox_max[0] - objspace_bbox_min[0],
            #    objspace_bbox_max[2] - objspace_bbox_min[2]
            #])

            ## Render an OBB height map as well
            #bbox_dims = o.bbox_max - o.bbox_min
            #model_matrix = Transform(scale=bbox_dims[:3], translation=o.bbox_min[:3]).as_mat4()
            #full_matrix = np.matmul(np.transpose(save_t), model_matrix)
            #obb = OBB.from_local2world_transform(full_matrix)
            #obb_tris = np.asarray(obb.get_triangles(), dtype=np.float32)
            #bbox_min = np.min(np.min(obb_tris, 0), 0)
            #bbox_max = np.max(np.max(obb_tris, 0), 0)
            #xmin, ymin = math.floor(bbox_min[0]), math.floor(bbox_min[2])
            #xsize, ysize = math.ceil(bbox_max[0]) - xmin + 1, math.ceil(bbox_max[2]) - ymin + 1
            #if ymin < 0: 
            #    ysize += ymin
            #    ymin = 0
            #if xmin < 0: 
            #    xsize += xmin
            #    xmin = 0
            #if ymin + ysize > self.size:
            #    ysize = self.size - ymin
            #if xmin + xsize > self.size:
            #    xsize = self.size - xmin
            #rendered_obb = self.render_object_helper(obb_tris, xmin, ymin, xsize, ysize, self.size)
            #description["height_map_obb"] = torch.from_numpy(rendered_obb).float()
            #description['bbox_min_obb'] = bbox_min
            #description['bbox_max_obb'] = bbox_max

            ## tmp = np.zeros((self.size, self.size))
            ## tmp[xmin:xmin+rendered_obb.shape[0],ymin:ymin+rendered_obb.shape[1]] = rendered_obb
            ## visualization += tmp

            nodes.append(description)
        
        if hasattr(room, "transform"):
            t = projection.to_2d(np.asarray(room.transform).reshape(4,4))
        else:
            t = projection.to_2d()
        #Render the floor 
        o = Obj(room.modelId+"f", room.house_id, is_room=True)
        o.transform(t)
        floor = self.render_object(o, 0, 0, self.size, self.size, self.size)
        visualization += floor
        floor = torch.from_numpy(floor).float()
        
        #Render the walls
        o = Obj(room.modelId+"w", room.house_id, is_room=True)
        o.transform(t)
        wall_original = self.render_object(o, 0, 0, self.size, self.size, self.size)
        wall_original = torch.from_numpy(wall_original).float()
        wall = {}
        wall["height_map_from_obj"] = wall_original
        segments = []
        wall_combined = torch.zeros_like(wall_original)
        for w in room.walls:
            #print(w["points"])
            description = {}
            bbox_min = np.asarray([w.xmin, w.zmin, w.ymin, 1])
            bbox_max = np.asarray([w.xmax, w.zmax, w.ymax, 1])
            bbox_min = np.dot(bbox_min, t)
            bbox_max = np.dot(bbox_max, t)
            description["bbox_min"] = bbox_min
            description["bbox_max"] = bbox_max
            description["adjacent"] = w.adjacent
            description["id"] = w.id
            p0 = w.points[0]+[1]
            p1 = w.points[1]+[1]
            p0 = np.dot(p0, t)
            p1 = np.dot(p1, t)
            description["points"] = [p0, p1]

            xmin = math.floor(bbox_min[0])
            ymin = math.floor(bbox_min[2])
            xsize = math.ceil(bbox_max[0]) - xmin + 1
            ysize = math.ceil(bbox_max[2]) - ymin + 1

            o = Obj(wall=w)
            o.transform(t)
            rendered = TopDownView.render_object_full_size(o,self.size)
            rendered = torch.from_numpy(rendered).float()
            description["height_map"] = rendered[xmin:xmin+xsize, ymin:ymin+ysize]
            update = rendered > wall_combined
            wall_combined[update] = rendered[update]
            #visualization[0:xsize,0:ysize] += description["height_map"]
            
            segments.append(description)
        wall["segments"] = segments
        #wall["height_map"] = wall_combined
        wall["height_map"] = wall_original
        visualization += wall_original
        #print(segments)
        #_ = input()

        return (visualization, (floor, wall, nodes))
    

    def render_graph(self, room, root, targets):
        target_identifiers = list(map(lambda x: x[0], targets))

        projection = self.pgen.get_projection(room)
        
        visualization = np.zeros((self.size,self.size))
        nodes = []

        for i, node in enumerate(room.nodes):
            modelId = node.modelId #Camelcase due to original json

            t = np.asarray(node.transform).reshape(4,4)

            o = Obj(modelId)
            
            # NEW
            objspace_width = np.linalg.norm(o.front_left - o.front_right)
            objspace_depth = np.linalg.norm(o.front_left - o.back_left)
            #END NEW

            t = projection.to_2d(t)
            o.transform(t)

            # NEW
            worldspace_width = np.linalg.norm(o.front_left - o.front_right)
            worldspace_depth = np.linalg.norm(o.front_left - o.back_left)
            #END NEW

            t = projection.to_2d()
            bbox_min = np.dot(np.asarray([node.xmin, node.zmin, node.ymin, 1]), t)
            bbox_max = np.dot(np.asarray([node.xmax, node.zmax, node.ymax, 1]), t)
            xmin = math.floor(bbox_min[0])
            ymin = math.floor(bbox_min[2])
            xsize = math.ceil(bbox_max[0]) - xmin + 1
            ysize = math.ceil(bbox_max[2]) - ymin + 1

            description = {}
            description["modelId"] = modelId
            description["transform"] = node.transform
            description["bbox_min"] = bbox_min
            description["bbox_max"] = bbox_max
            
            #Since it is possible that the bounding box information of a room
            #Was calculated without some doors/windows
            #We need to handle these cases
            if ymin < 0: 
                ymin = 0
            if xmin < 0: 
                xmin = 0

            # render object
            rendered = self.render_object(o, xmin, ymin, xsize, ysize, self.size)
            description["height_map"] = torch.from_numpy(rendered).float()

            tmp = np.zeros((self.size, self.size))
            tmp[xmin:xmin+rendered.shape[0],ymin:ymin+rendered.shape[1]] = rendered

            # render bbox

            for idx, line in enumerate(o.bbox_lines()):
                direction = line[1] - line[0]
                distance = np.linalg.norm(direction)
                norm_direction = direction / distance
                for t in range(math.floor(distance)):
                    point = line[0] + t * norm_direction
                    x = min(math.floor(point[0]), self.size - 1)
                    y = min(math.floor(point[2]), self.size - 1)
                    tmp[x][y] = 1

            # temporarily darken image to see more easily
            category = ObjectCategories().get_coarse_category(modelId)
            identifier = f"{category}_{i}"
            if identifier in target_identifiers:
                tmp *= 0.8
            elif identifier != root:
                tmp *= 0.1
            else:
                for ray in o.front_rays():
                    for t in range(self.size):
                        point = ray.origin + t * ray.direction
                        if point[0] < 0 or point[0] > self.size or point[2] < 0 or point[2] > self.size:
                            break
                        color = 1 if int(t * objspace_depth / worldspace_depth) % 2 == 0 else -100
                        tmp[math.floor(point[0])][math.floor(point[2])] = color
                for ray in o.back_rays():
                    for t in range(self.size):
                        point = ray.origin + t * ray.direction
                        if point[0] < 0 or point[0] > self.size or point[2] < 0 or point[2] > self.size:
                            break
                        color = 1 if int(t * objspace_depth / worldspace_depth) % 2 else -100
                        tmp[math.floor(point[0])][math.floor(point[2])] = color
                for ray in o.left_rays():
                    for t in range(self.size):
                        point = ray.origin + t * ray.direction
                        if point[0] < 0 or point[0] > self.size or point[2] < 0 or point[2] > self.size:
                            break
                        color = 1 if int(t * objspace_width / worldspace_width) % 2 else -100
                        tmp[math.floor(point[0])][math.floor(point[2])] = color
                for ray in o.right_rays():
                    for t in range(self.size):
                        point = ray.origin + t * ray.direction
                        if point[0] < 0 or point[0] > self.size or point[2] < 0 or point[2] > self.size:
                            break
                        color = 1 if int(t * objspace_width / worldspace_width) % 2 else -100
                        tmp[math.floor(point[0])][math.floor(point[2])] = color

            visualization += tmp
            
            nodes.append(description)
        
        #Render the floor
        o = Obj(room.modelId+"f", room.house_id, is_room=True)
        t = projection.to_2d()
        o.transform(t)
        floor = self.render_object(o, 0, 0, self.size, self.size, self.size)
        visualization += floor
        floor = torch.from_numpy(floor).float()
    
        #Render the walls
        o = Obj(room.modelId+"w", room.house_id, is_room=True)
        t = projection.to_2d()
        o.transform(t)
        wall = self.render_object(o, 0, 0, self.size, self.size, self.size)
        visualization += wall
        wall = torch.from_numpy(wall).float()
        
        return (visualization, (floor, wall, nodes))


    def get_bounding_boxes(self, room, cap=False):
        """
         Return bounding boxes for overlaying graph nodes and edges
        """
        def get_bbox(node, transform):
            bbox_min = np.dot(np.asarray([node.xmin, node.zmin, node.ymin, 1]), transform)
            bbox_max = np.dot(np.asarray([node.xmax, node.zmax, node.ymax, 1]), transform)
            if cap:
                bbox_min = np.maximum(bbox_min, 0)
                bbox_max = np.minimum(bbox_max, self.size)
            return { "id": node.id, "min": bbox_min, "max": bbox_max }

        def get_bbox_object(node, transform):
            o = Obj(wall=node) if node.type == 'Wall' else Obj(node.modelId) 
            o.transform(transform)
#            bbox_min = np.dot(np.asarray([node.xmin, node.zmin, node.ymin, 1]), transform)
#            bbox_max = np.dot(np.asarray([node.xmax, node.zmax, node.ymax, 1]), transform)
            bbox_min = np.asarray([o.xmin(), o.zmin(), o.ymin(), 1])
            bbox_max = np.asarray([o.xmax(), o.zmax(), o.ymax(), 1])
            if cap:
                bbox_min = np.maximum(bbox_min, 0)
                bbox_max = np.minimum(bbox_max, self.size)
            return { "id": node.id, "min": bbox_min, "max": bbox_max }

        def add_directions_object(bbox, node, transform=None):
            o = Obj(wall=node) if node.type == 'Wall' else Obj(node.modelId) 
            o.transform(transform)
            bbox['front'] = o.front_rays()[0].direction
            bbox['left'] = o.left_rays()[0].direction
            bbox['right'] = o.right_rays()[0].direction
            bbox['back'] = o.back_rays()[0].direction
            bbox['front_left'] = o.front_left
            bbox['front_right'] = o.front_right
            bbox['back_left'] = o.back_left
            bbox['back_right'] = o.back_right

        def add_directions_wall(bbox, node, transform=None):
            wp1 = np.asarray([node.pts[0][0], node.pts[0][1], node.pts[0][2], 1])
            wp2 = np.asarray([node.pts[1][0], node.pts[1][1], node.pts[1][2], 1])
            d = (wp2 - wp1)/np.linalg.norm(wp2-wp1)
            delta = node.depth * 0.75
            wp1 = wp1 - delta*d
            wp2 = wp2 + delta*d
            p1 = np.dot(wp1, transform)
            p2 = np.dot(wp2, transform)
            m = (p1 + p2)/2
            c = np.asarray([self.size/2, m[1], self.size/2, 1])
            d = p2 - p1
            nd = d/np.linalg.norm(d)

            bbox['front'] = np.asarray([-nd[2], nd[1], nd[0], 1])
            bbox['right'] = nd
            bbox['left'] = -bbox['right']
            bbox['back'] = -bbox['front']

            depth = node.depth/2
            normal = np.asarray([node.normal[0], 0, node.normal[1], 0])
            bbox['front_left'] = np.dot(wp1 + depth*normal, transform)
            bbox['front_right'] = np.dot(wp2 + depth*normal, transform)
            bbox['back_left'] = np.dot(wp1 - depth*normal, transform)
            bbox['back_right'] = np.dot(wp2 - depth*normal, transform)

        projection = self.pgen.get_projection(room)
        bboxes = {}
        t = projection.to_2d()
        for node in room.nodes:
            nt = np.asarray(node.transform).reshape(4,4)
            nt = projection.to_2d(nt)
            bboxes[node.id] = get_bbox_object(node, nt)
            add_directions_object(bboxes[node.id], node, nt)

        room_bbox = get_bbox(room, t)

        if hasattr(room, "transform"):
            t = projection.to_2d(np.asarray(room.transform).reshape(4,4))
        else:
            t = projection.to_2d()
        for w in room.walls:
            bboxes[w.id] = get_bbox(w, t)
            add_directions_wall(bboxes[w.id], w, t)
        return bboxes, room_bbox


    def render_house_node_mask(self, room, node, xmin, ymin, xsize, ysize, img_size):
        projection = self.pgen.get_projection(room)

        modelId = node.modelId #Camelcase due to original json
        t = np.asarray(node.transform).reshape(4,4)
        o = Obj(modelId)

        t = projection.to_2d(t)
        o.transform(t)
        rendered = TopDownView.render_object(o, xmin, ymin, xsize, ysize, img_size)
        return np.greater(rendered, 0).astype(int)


    @staticmethod
    @jit(nopython=True)
    def render_object_helper(triangles, xmin, ymin, xsize, ysize, img_size):
        result = np.zeros((img_size, img_size), dtype=np.float32)
        N, _, _ = triangles.shape

        for triangle in range(N):
            x0,z0,y0 = triangles[triangle][0]
            x1,z1,y1 = triangles[triangle][1]
            x2,z2,y2 = triangles[triangle][2]
            a = -y1*x2 + y0*(-x1+x2) + x0*(y1-y2) + x1*y2
            if a != 0:
                for i in range(max(0,math.floor(min(x0,x1,x2))), \
                               min(img_size,math.ceil(max(x0,x1,x2)))):
                    for j in range(max(0,math.floor(min(y0,y1,y2))), \
                                   min(img_size,math.ceil(max(y0,y1,y2)))):
                        x = i+0.5
                        y = j+0.5
                        s = (y0*x2 - x0*y2 + (y2-y0)*x + (x0-x2)*y)/a
                        t = (x0*y1 - y0*x1 + (y0-y1)*x + (x1-x0)*y)/a
                        if s < 0 and t < 0:
                            s = -s
                            t = -t
                        if 0 < s < 1 and 0 < t < 1 and s + t <= 1:
                            height = z0 *(1-s-t) + z1*s + z2*t
                            result[i][j] = max(result[i][j], height)

        return result[xmin:xmin+xsize, ymin:ymin+ysize]

    @staticmethod
    def render_object(o, xmin, ymin, xsize, ysize, img_size):
        """
        Render a cropped top-down view of object
        
        Parameters
        ----------
        o (list[triple]): object to be rendered, represented as a triangle mesh
        xmin, ymin (int): min coordinates of the bounding box containing the object,
            with respect to the full image
        xsize, ysize (int); size of the bounding box containing the object
        img_size (int): size of the full image
        """
        triangles = np.asarray(list(o.get_triangles()), dtype=np.float32)
        return TopDownView.render_object_helper(triangles, xmin, ymin, xsize, ysize, img_size)
        
    @staticmethod
    @jit(nopython=True)
    def render_object_full_size_helper(triangles, size):
        result = np.zeros((size, size), dtype=np.float32)
        N, _, _ = triangles.shape

        for triangle in range(N):
            x0,z0,y0 = triangles[triangle][0]
            x1,z1,y1 = triangles[triangle][1]
            x2,z2,y2 = triangles[triangle][2]
            a = -y1*x2 + y0*(-x1+x2) + x0*(y1-y2) + x1*y2
            if a != 0:
                for i in range(max(0,math.floor(min(x0,x1,x2))), \
                               min(size,math.ceil(max(x0,x1,x2)))):
                    for j in range(max(0,math.floor(min(y0,y1,y2))), \
                                   min(size,math.ceil(max(y0,y1,y2)))):
                        x = i+0.5
                        y = j+0.5
                        s = (y0*x2 - x0*y2 + (y2-y0)*x + (x0-x2)*y)/a
                        t = (x0*y1 - y0*x1 + (y0-y1)*x + (x1-x0)*y)/a
                        if s < 0 and t < 0:
                            s = -s
                            t = -t
                        if 0 < s < 1 and 0 < t < 1 and s + t <= 1:
                            height = z0 *(1-s-t) + z1*s + z2*t
                            result[i][j] = max(result[i][j], height)
        
        return result

    @staticmethod
    def render_object_full_size(o, size):
        """
        Render a full-sized top-down view of the object, see render_object
        """
        triangles = np.asarray(list(o.get_triangles()), dtype=np.float32)
        return TopDownView.render_object_full_size_helper(triangles, size)

if __name__ == "__main__":
    from .house import House
    h = House(id_="51515da17cd4b575775cea4f5546737a")
    r = h.rooms[0]
    renderer = TopDownView()
    img = renderer.render(r)[0]
    img = m.toimage(img, cmin=0, cmax=1)
    img.show()
