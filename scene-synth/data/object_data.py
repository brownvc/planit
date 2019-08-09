import csv
import os
import numpy as np
import utils
import pickle

# ---------------------------------------------------------------------------------------

# Symmetry classes

def with_approx_classes(sym_classes):
    return set(sym_classes + [s+'_APPROX' for s in sym_classes])

ref_lr_sym_classes = with_approx_classes([
    '__SYM_REFLECT_LR'
])

ref_fb_sym_classes = with_approx_classes([
    '__SYM_REFLECT_FB'
])

rot_2_sym_classes = with_approx_classes([
    '__SYM_ROTATE_UP_2'
])

rot_4_sym_classes = with_approx_classes([
    '__SYM_ROTATE_UP_4'
])

corner_1_sym_classes = with_approx_classes([
    '__SYM_REFLECT_C1',
    '__CORNER_C1'
])

corner_2_sym_classes = with_approx_classes([
    '__SYM_REFLECT_C2',
    '__CORNER_C2'
])

rad_sym_classes = with_approx_classes([
    '__SYM_SPHERE',
    '__SYM_ROTATE_UP_INF',
    '__SYM_ROTATE_UP_3',
    '__SYM_ROTATE_UP_5',
    '__SYM_ROTATE_UP_6',
    '__SYM_ROTATE_UP_8',
    '__SYM_ROTATE_UP_10'
])

# ---------------------------------------------------------------------------------------

class __ObjectCategories():
    singleton = None
    """
    Determine which categories does each object belong to
    """
    def __init__(self):
        print("Initializing Category Map...") #Debug purposes
        fname = "ModelCategoryMapping_graph.csv"
        # fname = "ModelCategoryMapping.csv"
        self.model_to_categories = {}

        root_dir = os.path.dirname(os.path.abspath(__file__))
        model_cat_file = f"{root_dir}/{fname}"

        with open(model_cat_file, "r") as f:
            categories = csv.reader(f)
            for l in categories:
                self.model_to_categories[l[1]] = l[2:]

        #fname = "ModelCategoryMapping_grains.csv"
        #self.model_to_categories2 = {}

        #root_dir = os.path.dirname(os.path.abspath(__file__))
        #model_cat_file = f"{root_dir}/{fname}"

        #with open(model_cat_file, "r") as f:
        #    categories = csv.reader(f)
        #    for l in categories:
        #        self.model_to_categories2[l[1]] = [l[8]]

        # Also store an inverse mapping of category to the id of some model
        #    of that category
        self.coarsecat_to_model = {}
        self.finecat_to_model = {}
        self.finalcat_to_model = {}
        # Also store mapping from coarse category to all possible fine/final categories
        self.coarse_to_fine = {}
        self.coarse_to_final = {}
        for model_id in self.model_to_categories.keys():
            coarsecat = self.get_coarse_category(model_id)
            finecat = self.get_fine_category(model_id)
            finalcat = self.get_final_category(model_id)
            self.coarsecat_to_model[coarsecat] = model_id
            self.finecat_to_model[finecat] = model_id
            self.finalcat_to_model[finalcat] = model_id
            if coarsecat not in self.coarse_to_fine:
                self.coarse_to_fine[coarsecat] = set([])
                self.coarse_to_final[coarsecat] = set([])
            self.coarse_to_fine[coarsecat].add(finecat)
            self.coarse_to_final[coarsecat].add(finalcat)


    def get_fine_category(self, model_id):
        model_id = model_id.replace("_mirror","")
        return self.model_to_categories[model_id][0]
    
    def get_coarse_category(self, model_id):
        model_id = model_id.replace("_mirror","")
        return self.model_to_categories[model_id][1]

    def get_final_category(self, model_id):
        """
        Final categories used in the generated dataset
        Minor tweaks from fine categories
        """
        model_id = model_id.replace("_mirror","")

        #return self.model_to_categories2[model_id][0]

        category = self.model_to_categories[model_id][0]
        if model_id == "199":
            category = "dressing_table_with_stool"
        if model_id in ["150", "453", "s__1138", "s__1251"]:
            category = "bidet"
        if category == "nightstand":
            category = "stand"
        if category == "bookshelf":
            category = "shelving"
        if category == "books":
            category = "book"
        if category == "xbox" or category == "playstation":
            category = "console"
        return category
    
    def get_coarse_category_from_final_category(self, finalcat):
        model_id = self.finalcat_to_model[finalcat]
        return self.get_coarse_category(model_id)
    def get_final_categories_for_coarse_category(self, coarsecat):
        return list(self.coarse_to_final[coarsecat])
    
    def get_symmetry_class(self, model_id):
        model_id = model_id.replace("_mirror","")
        sym_class = self.model_to_categories[model_id][8]
        sym_classes = sym_class.split(',')
        return sym_classes

    def has_symmetry_type(self, sym_classes_to_check, target_sym_classes):
        return len(sym_classes_to_check & target_sym_classes) > 0

    def is_not_symmetric(self, sym_classes):
        return len(sym_classes) == 0 or (len(sym_classes) == 1 and '' in sym_classes)
    def is_radially_symmetric(self, sym_classes):
        return self.has_symmetry_type(sym_classes, rad_sym_classes)
    def is_front_back_reflect_symmetric(self, sym_classes):
        return self.has_symmetry_type(sym_classes, ref_fb_sym_classes)
    def is_left_right_reflect_symmetric(self, sym_classes):
        return self.has_symmetry_type(sym_classes, ref_lr_sym_classes)
    def is_two_way_rotate_symmetric(self, sym_classes):
        return self.has_symmetry_type(sym_classes, rot_2_sym_classes)
    def is_four_way_rotate_symmetric(self, sym_classes):
        return self.has_symmetry_type(sym_classes, rot_4_sym_classes)
    def is_corner_1_symmetric(self, sym_classes):
        return self.has_symmetry_type(sym_classes, corner_1_sym_classes)
    def is_corner_2_symmetric(self, sym_classes):
        return self.has_symmetry_type(sym_classes, corner_2_sym_classes)

    def is_window(self, cat_name):
        return cat_name == "window"

    def is_door(self, cat_name):
        return cat_name.endswith("door")

    def is_arch(self, cat_name):
        return self.is_door(cat_name) or self.is_window(cat_name)

    
    def all_categories(self, data_root_dir, data_folder):
        with open(f"{data_root_dir}/{data_folder}/final_categories_frequency", "r") as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines]
        cats = [line.split()[0] for line in lines]
        return [c for c in cats]

    def all_arch_categories(self):
        return ["window", "door", "single_hinge_door", "double_hinge_door", "sliding_door"]

    def all_non_arch_categories(self, data_root_dir, data_folder):
        with open(f"{data_root_dir}/{data_folder}/final_categories_frequency", "r") as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines]
        cats = [line.split()[0] for line in lines]
        return [c for c in cats if not self.is_arch(c)]

    def all_arch_category_counts(self, data_root_dir, data_folder):
        with open(f"{data_root_dir}/{data_folder}/final_categories_frequency", "r") as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines]
        cats = [line.split()[0] for line in lines]
        counts = [int(line.split()[1]) for line in lines]
        return [counts[i] for i,c in enumerate(cats) if self.is_arch(c)]

    def all_non_arch_category_counts(self, data_root_dir, data_folder):
        with open(f"{data_root_dir}/{data_folder}/final_categories_frequency", "r") as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines]
        cats = [line.split()[0] for line in lines]
        counts = [int(line.split()[1]) for line in lines]
        return [counts[i] for i,c in enumerate(cats) if not self.is_arch(c)]

    # Compute a 'size' for each category by averaging over the bbox areas of all instances
    #    of that category
    def all_non_arch_category_sizes(self, data_root_dir, data_folder):
        catnames = self.all_non_arch_categories(data_root_dir, data_folder)
        cat_name2index = {name:i for i,name in enumerate(catnames)}
        n_categories = len(catnames)
        with open(f'{data_root_dir}/{data_folder}/model_dims.pkl', 'rb') as f:
            model_dims = pickle.load(f)
        cat_sizes = [0.0 for i in range(n_categories)]
        cat_nums = [0 for i in range(n_categories)]
        for model_id,dims in model_dims.items():
            catname = self.get_final_category(model_id)
            if not self.is_arch(catname):
                cat = cat_name2index[catname]
                size = dims[0]*dims[1]
                cat_nums[cat] += 1
                cat_sizes[cat] += size
        for i in range(n_categories):
            cat_sizes[i] /= cat_nums[i]
        return cat_sizes
    def all_arch_category_sizes(self, data_root_dir, data_folder):
        catnames = self.all_arch_categories()
        cat_name2index = {name:i for i,name in enumerate(catnames)}
        n_categories = len(catnames)
        with open(f'{data_root_dir}/{data_folder}/model_dims.pkl', 'rb') as f:
            model_dims = pickle.load(f)
        cat_sizes = [0.0 for i in range(n_categories)]
        cat_nums = [0 for i in range(n_categories)]
        for model_id,dims in model_dims.items():
            catname = self.get_final_category(model_id)
            if self.is_arch(catname):
                cat = cat_name2index[catname]
                size = dims[0]*dims[1]
                cat_nums[cat] += 1
                cat_sizes[cat] += size
        for i in range(n_categories):
            if cat_nums[i] > 0:
                cat_sizes[i] /= cat_nums[i]
        return cat_sizes

    # Compute a notion of 'importance' as a product of frequency and size
    def all_non_arch_category_importances(self, data_root_dir, data_folder):
        counts = self.all_non_arch_category_counts(data_root_dir, data_folder)
        sizes = self.all_non_arch_category_sizes(data_root_dir, data_folder)
        assert(len(counts) == len(sizes))
        maxcount = max(counts)
        freqs = [c/maxcount for c in counts]
        imps = [freqs[i] * sizes[i] for i in range(len(sizes))]
        return imps

    def all_non_arch_categories_importance_order(self, data_root_dir, data_folder):
        catnames = self.all_non_arch_categories(data_root_dir, data_folder)
        imps = self.all_non_arch_category_importances(data_root_dir, data_folder)
        assert(len(catnames) == len(imps))
        cat_imp = [(catnames[i], imps[i]) for i in range(len(catnames))]
        cat_imp = sorted(cat_imp, key=lambda x:-x[1])
        return [x[0] for x in cat_imp]

def ObjectCategories():
    if __ObjectCategories.singleton is None:
        __ObjectCategories.singleton = __ObjectCategories()
    return __ObjectCategories.singleton

# ---------------------------------------------------------------------------------------

class __ObjectCategoriesGrains():
    singleton = None
    """
    Determine which categories does each object belong to
    """
    def __init__(self):
        print("Initializing Category Map...") #Debug purposes
        fname = "ModelCategoryMapping_grains.csv"
        self.model_to_categories = {}

        root_dir = os.path.dirname(os.path.abspath(__file__))
        model_cat_file = f"{root_dir}/{fname}"

        with open(model_cat_file, "r") as f:
            categories = csv.reader(f)
            for l in categories:
                self.model_to_categories[l[1]] = [l[8]]

    def get_grains_category(self, model_id):
        model_id = model_id.replace("_mirror","")
        return self.model_to_categories[model_id][0]

def ObjectCategoriesGrains():
    if __ObjectCategoriesGrains.singleton is None:
        __ObjectCategoriesGrains.singleton = __ObjectCategoriesGrains()
    return __ObjectCategoriesGrains.singleton

# ---------------------------------------------------------------------------------------

class __ObjectData():
    singleton = None
    """
    Various information associated with the objects
    """
    def __init__(self):
        print("Initializing Object Data") #Debug purposes
        self.model_to_data = {}

        root_dir = os.path.dirname(os.path.abspath(__file__))
        model_data_file = f"{root_dir}/Models.csv"

        with open(model_data_file, "r") as f:
            data = csv.reader(f)
            for l in data:
                if l[0] != 'id':  # skip header row
                    self.model_to_data[l[0]] = l[1:]

    def get_front(self, model_id):
        model_id = model_id.replace("_mirror","")
        # TODO compensate for mirror (can have effect if not axis-aligned in model space)
        return [float(a) for a in self.model_to_data[model_id][0].split(",")]

    def get_aligned_dims(self, model_id):
        """Return canonical alignment dimensions of model *in meters*"""
        model_id = model_id.replace('_mirror', '')  # NOTE dims don't change since mirroring is symmetric on yz plane
        return [float(a)/100.0 for a in self.model_to_data[model_id][4].split(',')]

    def get_model_semantic_frame_matrix(self, model_id):
        """Return canonical semantic frame matrix for model.
           Transforms from semantic frame [0,1]^3, [x,y,z] = [right,up,back] to raw model coordinates."""
        up = np.array([0, 1, 0])  # NOTE: up is assumed to always be +Y for certain objects
        front = np.array(self.get_front(model_id))
        has_mirror = '_mirror' in model_id
        model_id = model_id.replace('_mirror', '')
        hdims = np.array(self.get_aligned_dims(model_id)) * 0.5
        p_min = np.array([float(a) for a in self.model_to_data[model_id][2].split(',')])
        p_max = np.array([float(a) for a in self.model_to_data[model_id][3].split(',')])
        if has_mirror:
            p_max[0] = -p_max[0]
            p_min[0] = -p_min[0]
        model_space_center = (p_max + p_min) * 0.5
        m = np.identity(4)
        m[:3, 0] = np.cross(front, up) * hdims[0]  # +x = right
        m[:3, 1] = np.array(up) * hdims[1]         # +y = up
        m[:3, 2] = -front * hdims[2]               # +z = back = -front
        m[:3, 3] = model_space_center              # origin = center
        # r = np.identity(3)
        # r[:3, 0] = np.cross(front, up)  # +x = right
        # r[:3, 1] = np.array(up)         # +y = up
        # r[:3, 2] = -front               # +z = back = -front
        # s = np.identity(3)
        # s[0, 0] = hdims[0]
        # s[1, 1] = hdims[1]
        # s[2, 2] = hdims[2]
        # sr = np.matmul(s, r)
        # m = np.identity(4)
        # m[:3, :3] = sr
        # m[:3, 3] = model_space_center
        return m

    def get_alignment_matrix(self, model_id):
        """
        Since some models in the dataset are not aligned in the way we want
        Generate matrix that realign them
        """
        #alignment happens BEFORE mirror, so make sure no mirrored 
        #object will ever call this!
        #model_id = model_id.replace("_mirror","")
        if self.get_front(model_id) == [0,0,1]:
            return None
        else:
            #Let's just do case by case enumeration!!!
            if model_id in ["106", "114", "142", "323", "333", "363", "364",
                            "s__1782", "s__1904"]:
                M = [[-1,0,0,0],
                     [0,1,0,0],
                     [0,0,-1,0],
                     [0,0,0,1]]
            elif model_id in ["s__1252", "s__400", "s__885"]:
                M = [[0,0,-1,0],
                     [0,1,0,0],
                     [1,0,0,0],
                     [0,0,0,1]]
            elif model_id in ["146", "190", "s__404", "s__406"]:
                M = [[0,0,1,0],
                     [0,1,0,0],
                     [-1,0,0,0],
                     [0,0,0,1]]
            else:
                print(model_id)
                raise NotImplementedError

            return np.asarray(M)
    
    def get_setIds(self, model_id):
        model_id = model_id.replace("_mirror","")
        return [a for a in self.model_to_data[model_id][8].split(",")]

def ObjectData():
    if __ObjectData.singleton is None:
        __ObjectData.singleton = __ObjectData()
    return __ObjectData.singleton

# ---------------------------------------------------------------------------------------

if __name__ == "__main__":
    a = ObjectCategories()
    print(a.get_symmetry_class("40"))
    print(a.get_final_category("40"))
    #b = ObjectData()
    #print(b.get_front("40"))
#print(b.get_setIds("s__2240"))
