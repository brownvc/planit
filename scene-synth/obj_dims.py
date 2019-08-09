from data import *
from utils import *
import pickle

class ObjDims:

    def __init__(self, data_dir, data_root_dir=None, room_dim=6.05):
        if not data_root_dir:
            data_root_dir = utils.get_data_root_dir()
        self.data_dir = data_dir
        self.data_root_dir = data_root_dir
        self.room_dim = room_dim

    def run(self):
        with open(f"{self.data_root_dir}/{self.data_dir}/model_frequency", "r") as f:
            models = f.readlines()

        models = [l[:-1].split(" ") for l in models]
        models = [l[0] for l in models]
        #print(models)

        model_dims = dict()

        for model in models:
            o = Obj(model)
            dims = [(o.bbox_max[0] - o.bbox_min[0])/self.room_dim, \
                    (o.bbox_max[2] - o.bbox_min[2])/self.room_dim]

            model_dims[model] = dims

        with open(f"{self.data_root_dir}/{self.data_dir}/model_dims.pkl", 'wb') as f:
            pickle.dump(model_dims, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    ObjDims(data_dir).run()

