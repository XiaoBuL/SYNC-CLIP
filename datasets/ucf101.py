import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader

from .oxford_pets import OxfordPets


template = ['a photo of a person doing {}.']


class UCF101(DatasetBase):

    dataset_dir = 'ucf101'

    def __init__(self, root, num_shots, subsample, dalle=True):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'UCF-101-midframes')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_UCF101.json')

        self.template = template

        train_u, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train_u = self.generate_fewshot_dataset(train_u, num_shots=num_shots)
        train, val= OxfordPets.subsample_classes(train_u, val, subsample=subsample)
        test_base = OxfordPets.subsample_classes(test, subsample="base")[0]
        test_novel = OxfordPets.subsample_classes(test, subsample="new")[0]

        super().__init__(train_x=train, val=val, test_base=test_base, test_novel=test_novel, train_u=train_u, test=test)
    
    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')[0] # trainlist: filename, label
                action, filename = line.split('/')
                label = cname2lab[action]

                elements = re.findall('[A-Z][^A-Z]*', action)
                renamed_action = '_'.join(elements)

                filename = filename.replace('.avi', '.jpg')
                impath = os.path.join(self.image_dir, renamed_action, filename)

                item = Datum(
                    impath=impath,
                    label=label,
                    classname=renamed_action
                )
                items.append(item)
        
        return items
