import os

from .utils import Datum, DatasetBase
from .oxford_pets import OxfordPets


template = ['a photo of a {}.']


class Caltech101(DatasetBase):

    dataset_dir = 'caltech-101'

    def __init__(self, root, num_shots, subsample, dalle=True):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '101_ObjectCategories')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_Caltech101.json')

        self.template = template

        train_u, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train_u = self.generate_fewshot_dataset(train_u, num_shots=num_shots)
        train, val= OxfordPets.subsample_classes(train_u, val, subsample=subsample)
        test_base = OxfordPets.subsample_classes(test, subsample="base")[0]
        test_novel = OxfordPets.subsample_classes(test, subsample="new")[0]
        super().__init__(train_x=train, val=val, test_base=test_base, test_novel=test_novel, train_u=train_u, test=test)