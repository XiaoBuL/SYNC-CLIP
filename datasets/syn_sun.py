import os
from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from .oxford_pets import OxfordPets

class Syn_Sun(DatasetBase):
    
    dataset_dir = 'dalle_sun397'

    def __init__(self, root, num_shots, subsample, dalle=True):
        # root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'SUN397')
        self.split_path = os.path.join(self.dataset_dir, 'dalle_sun.json')
        if not dalle:
            self.image_dir = '/data/mushui/dataset/StableDiffusion/sun397'
            self.split_path = os.path.join(self.dataset_dir, 'stable_diffusion_sun.json')
        train_u = OxfordPets.read_split(self.split_path, self.image_dir)[0]
        train_u = self.generate_fewshot_dataset(train_u, num_shots=num_shots)
        train = OxfordPets.subsample_classes(train_u, subsample=subsample)[0]
        super().__init__(train_x=train, train_u=train_u)