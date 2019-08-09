from data import RenderedScene, ObjectCategories
import os
import pickle
import numpy as np
import utils


"""
Simple bigram model
"""
class SupportPrior():

    def __init__(self):
        pass

    def learn(self, data_folder="bedroom_final", data_root_dir=None):
        if not data_root_dir:
            data_root_dir = utils.get_data_root_dir()
        data_dir = f"{data_root_dir}/{data_folder}"
        self.data_dir = data_dir
        self.category_map = ObjectCategories()

        files = os.listdir(data_dir)
        files = [f for f in files if ".pkl" in f and not "domain" in f and not "_" in f]

        self.categories = self.category_map.all_non_arch_categories(data_root_dir, data_folder)
        self.category_count = self.category_map.all_non_arch_category_counts(data_root_dir, data_folder)
        self.cat_to_index = {self.categories[i]:i for i in range(len(self.categories))}
        self.num_categories = len(self.categories)
        self.categories.append("floor")
        N = self.num_categories
        
        self.support_count = [[0 for i in range(N+1)] for j in range(N)]

        for index in range(len(files)):
            print(index)
            with open(f"{data_dir}/{index}.pkl", "rb") as f:
                (_, _, nodes), _ = pickle.load(f)
            
            object_nodes = []
            id_to_cat = {}
            for node in nodes:
                modelId = node["modelId"]
                category = self.category_map.get_final_category(modelId)
                if not self.category_map.is_arch(category):
                    object_nodes.append(node)
                    id_to_cat[node["id"]] = self.cat_to_index[category]
                    node["category"] = self.cat_to_index[category]
            
            for node in object_nodes:
                parent = node["parent"]
                category = node["category"]
                if parent == "Floor" or parent is None:
                    self.support_count[category][-1] += 1
                else:
                    self.support_count[category][id_to_cat[parent]] += 1
            #quit()

        self.possible_supports={}
        for i in range(self.num_categories):
            print(f"Support for {self.categories[i]}:")
            supports = [(c, self.support_count[i][c]/self.category_count[i]) for c in range(N+1)]
            supports = sorted(supports, key = lambda x:-x[1])
            supports = [s for s in supports if s[1] > 0.01]
            for s in supports:
                print(f"    {self.categories[s[0]]}:{s[1]:4f}")
            self.possible_supports[i] = [s[0] for s in supports]
        
        print(self.possible_supports)       
        self.N = N

    def save(self, dest=None):
        if dest == None:
            dest = f"{self.data_dir}/support_prior.pkl"
        with open(dest, "wb") as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)

    def load(self, data_dir):
        source = f"{data_dir}/support_prior.pkl"
        with open(source, "rb") as f:
            self.__dict__ = pickle.load(f)
    
            
if __name__ == "__main__":
    a = SupportPrior()
    a.learn("")
    a.save()
