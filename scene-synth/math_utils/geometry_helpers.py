import numpy as np
import math
from numpy import linalg

class ObjRay:
    DirectionNone = "obj_direction_none"
    DirectionBack = "obj_direction_back"
    DirectionFrontLeft = "obj_direction_front_left"
    DirectionFrontRight = "obj_direction_front_right"
    
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = np.array(direction) / linalg.norm(np.array(direction))

    def __str__(self):
        return f"ObjRay: Origin: {self.origin}, Direction: {self.direction}"

    def __repr__(self):
        return self.__str__()

    def transform_distance_along_ray(self, distance, transform):
        objspace_ray_start = self.origin
        objspace_ray_end = self.origin + distance * self.direction
        worldspace_ray_start = np.dot(objspace_ray_start, transform)
        worldspace_ray_end = np.dot(objspace_ray_end, transform)
        worldspace_dist = np.linalg.norm(worldspace_ray_end - worldspace_ray_start)

        return worldspace_dist


# 2d Ray
class Ray2D():

    DirectionNone = "2d_direction_none"
    DirectionBack = "2d_direction_back"
    DirectionFrontLeft = "2d_direction_front_left"
    DirectionFrontRight = "2d_direction_front_right"
    
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = np.array(direction) / linalg.norm(np.array(direction))

    # Returns the distance to the line if they intersect, otherwise None
    def distance_to_line(self, line):
        v1 = self.origin - line.p_one
        v2 = line.vector
        v3 = np.array([-self.direction[1], self.direction[0]])

        if math.isclose(np.dot(v2,v3), 0):
            return None # parallel lines

        t1 = np.cross(v2, v1) / np.dot(v2,v3)
        t2 = np.dot(v1, v3) / np.dot(v2, v3)

        if t1 >= 0.0 and t2 >= 0.0 and t2 <= 1.0:
            return t1 # distance along ray of intersection from origin
        else: 
            return None

    def distance_to_bbox(self, bbox):
        min_distance = (None, None)
        
        for line in bbox.lines:
            min_distance = get_min(min_distance, self.distance_to_line(line), None)

        return min_distance[0]


    # Returns a Ray.Direction
    def relative_position_of_point(self, point):
        d = point - self.origin
        if math.isclose(linalg.norm(d), 0):
            return Ray2D.DirectionNone # Given the original point
        
        if np.dot(self.direction, d) < 0:
            return Ray2D.DirectionBack

        if np.cross(self.direction, d / linalg.norm(d)) < 0:
            return Ray2D.DirectionFrontRight

        return Ray2D.DirectionFrontLeft

    def __str__(self):
        return f"Ray2D: Origin: {self.origin}, Direction: {self.direction}"

    def __repr__(self):
        return self.__str__()


# 2d Line segment
class Line2D():

    def __init__(self, p_one, p_two):
        self.p_one = np.asarray(p_one)
        self.p_two = np.asarray(p_two)
        self.vector = self.p_two - self.p_one
        self.length = linalg.norm(self.vector)
        self.direction = self.vector / linalg.norm(self.vector)

    def __str__(self):
        return f"Line2D: {self.p_one} -> {self.p_two}"

    def __repr__(self):
        return self.__str__()


#2d Bounding Box
class BBox2D():
    def __init__(self, corners):
        p_one, p_two, p_three, p_four = corners

        self.corners = [np.asarray(p_one), np.asarray(p_two), \
            np.asarray(p_three), np.asarray(p_four)]

        self.lines = [Line2D(p_one, p_two), Line2D(p_two, p_three), \
            Line2D(p_three, p_four), Line2D(p_four, p_one)]

        self.centroid = np.mean(corners, axis=0)

        self.mins = np.asarray(self.corners).min(axis=0)
        self.maxs = np.asarray(self.corners).max(axis=0)

    @classmethod
    def from_min_max(cls, mins, maxs):
        return BBox2D([
            [mins[0], mins[1]],
            [mins[0], maxs[1]],
            [maxs[0], mins[1]],
            [maxs[0], maxs[1]]
        ])

    def left_half(self):
        mins = self.mins
        maxs = [self.centroid[0], self.maxs[1]]
        return BBox2D.from_min_max(mins, maxs)
    def right_half(self):
        mins = [self.centroid[0], self.mins[1]]
        maxs = self.maxs
        return BBox2D.from_min_max(mins, maxs)
    def back_half(self):
        mins = self.mins
        maxs = [self.maxs[0], self.centroid[1]]
        return BBox2D.from_min_max(mins, maxs)
    def front_half(self):
        mins = [self.mins[0], self.centroid[1]]
        maxs = self.maxs
        return BBox2D.from_min_max(mins, maxs)
    
    # edge should be 2 parallel rays whose origins form an edge. 
    def distance_from_edge(self, edge):
        ray_one, ray_two = edge

        # First: Check if the line segment between the two rays intersects this bbox
        # If so, this means that the bbox is overlapping this edge, and so the distance
        #    should be zero
        p0, p1 = ray_one.origin, ray_two.origin
        # If either of these points are within the box, then there's an overlap
        if self.contains_point(p0) or self.contains_point(p1):
            return 0
        direction = p1 - p0
        dirlen = np.linalg.norm(direction)
        edge_ray = Ray2D(p0, direction)
        dist = edge_ray.distance_to_bbox(self)
        if dist is not None and dist >= 0 and dist <= dirlen:
            return 0

        min_distance = (None, None)
        min_distance = get_min(min_distance, ray_one.distance_to_bbox(self), "Ray One")
        min_distance = get_min(min_distance, ray_two.distance_to_bbox(self), "Ray Two")
        for i, corner in enumerate(self.corners):
            min_distance = get_min(min_distance, distance_to_point_from_edge(corner, edge), "Corner {}".format(i))

        # If there were no intersections, min_distance is None. Otherwise it's scalar
        if min_distance == (None, None):
            if is_point_between_parallel_rays(self.centroid, ray_one, ray_two):
                print("CENTROID WITHIN LINES BUT INTERSECTION NOT DETECTED")
        return min_distance[0]

    def intersects(self, other):
        if other.maxs[0] < self.mins[0] or other.mins[0] > self.maxs[0] or \
           other.maxs[1] < self.mins[1] or other.mins[1] > self.maxs[1]:
            return False
        else:
            return True

    def intersection(self, other):
        mins = np.maximum(self.mins, other.mins)
        maxs = np.minimum(self.maxs, other.maxs)
        return BBox2D.from_min_max(mins, maxs)

    def contains_point(self, p):
        return p[0] >= self.mins[0] and p[0] <= self.maxs[0] and \
            p[1] >= self.mins[1] and p[1] <= self.maxs[1]

    @property
    def width(self):
        return (self.maxs[0] - self.mins[0])
    @property
    def height(self):
        return (self.maxs[1] - self.mins[1])
    @property
    def area(self):
        return self.width * self.height
        
    def __str__(self):
        return f"BBox2D: {self.lines}"

    def __repr__(self):
        return self.__str__()

class Direction:
    FRONT = 'front'
    BACK = 'back'
    LEFT = 'left'
    RIGHT = 'right'

    @classmethod
    def to_vector(cls, direction):
        if direction == Direction.FRONT:
            return [0,1]
        elif direction == Direction.BACK:
            return [0,-1]
        elif direction == Direction.LEFT:
            return [-1,0]
        elif direction == Direction.RIGHT:
            return [1,0]

    @classmethod
    def is_horizontal(cls, direction):
        if direction == Direction.LEFT or direction == Direction.RIGHT:
            return True
        else:
            return False

    @classmethod
    def all_directions(cls):
        return [Direction.FRONT, Direction.BACK, Direction.LEFT, Direction.RIGHT]

    @classmethod
    def opposite(cls, direction):
        opp_dirs = [Direction.BACK, Direction.FRONT, Direction.RIGHT, Direction.LEFT]
        return opp_dirs[cls.all_directions().index(direction)]

class Range:
    def __init__(self, min_v=None, max_v=None):
        self.min = min_v
        self.max = max_v

    def length(self):
        return self.max - self.min

    def update(self, value):
        # update max
        if not self.max:
            self.max = value
        else:
            self.max = max(self.max, value)

        # update min
        if not self.min:
            self.min = value
        else:
            self.min = min(self.min, value)

    def clip(self, rng):
        if not self.overlaps(rng):
            self.min = None
            self.max = None
        else:    
            # clip min
            if self.min:
                self.min = min(max(self.min, rng.min), rng.max)

            #clip max
            if self.max:
                self.max = min(max(self.max, rng.min), rng.max)

    def is_within(self, value):
        if not self.is_valid():
            return False

        return self.min <= value and self.max >= value

    def overlaps(self, rng):
        if not self.is_valid():
            return False

        # no overlap if rng to left of min or right of max
        if self.min > rng.max or self.max < rng.min:
            return False

        return True

    def is_valid(self):
        if not self.min or not self.max:
            return False

        return True
    def __str__(self):
        return f"Range({self.min}, {self.max})"

    def __repr__(self):
        return self.__str__()


 # just gets the min of two values which are possibly None (with ids)
def get_min(orig, new_val, new_id):
    prev_val, prev_id = orig
    if prev_val == None:
        return new_val, new_id

    # val_one is a real value
    if new_val == None:
        return orig

    # both real values
    if new_val < prev_val:
        return new_val, new_id
    else:
        return orig

# edge should be 2 parallel rays whose origins form an edge. 
def distance_to_point_from_edge(point, edge):
    ray_one, ray_two = edge

    if not is_point_between_parallel_rays(point, ray_one, ray_two):
        return None

    direction_along_edge = ray_one.origin - ray_two.origin
    return linalg.norm(np.cross(point - ray_one.origin, direction_along_edge)) / linalg.norm(direction_along_edge)

def is_point_between_parallel_rays(point, ray_one, ray_two):
    return (ray_one.relative_position_of_point(point) == Ray2D.DirectionFrontRight and \
            ray_two.relative_position_of_point(point) == Ray2D.DirectionFrontLeft) or \
            (ray_one.relative_position_of_point(point) == Ray2D.DirectionFrontLeft and \
            ray_two.relative_position_of_point(point) == Ray2D.DirectionFrontRight)
