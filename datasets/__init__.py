from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .syn_imagenet import Syn_Imagenet
from .syn_caltech import Syn_Caltech
from .syn_flowers import Syn_Flowers
from .syn_food import Syn_Food
from .syn_cars import Syn_Cars
from .syn_dtd import Syn_DTD
from .syn_eurosat import Syn_Eurosat
from .syn_pets import Syn_Pets
from .syn_sun import Syn_Sun
from .syn_ucf import Syn_UCF
from .syn_fgvc import Syn_fgvc
from .sd_caltech import SD_Caltech

dataset_list = {
                "oxford_pets": OxfordPets,
                "eurosat": EuroSAT,
                "ucf101": UCF101,
                "sun397": SUN397,
                "caltech101": Caltech101,
                "dtd": DescribableTextures,
                "fgvc": FGVCAircraft,
                "food101": Food101,
                "oxford_flowers": OxfordFlowers,
                "stanford_cars": StanfordCars,
                "syn_imagenet": Syn_Imagenet,
                "syn_caltech": Syn_Caltech,
                "syn_flowers": Syn_Flowers,
                "syn_food": Syn_Food,
                "syn_cars": Syn_Cars,
                "syn_dtd": Syn_DTD,
                "syn_eurosat": Syn_Eurosat,
                "syn_pets": Syn_Pets,
                "syn_sun": Syn_Sun,
                "syn_ucf": Syn_UCF,
                "syn_fgvc": Syn_fgvc,
                "sd_caltech": SD_Caltech
                }


# def build_dataset(dataset, root_path, shots, subsample):
#     return dataset_list[dataset](root_path, shots, subsample)
def build_dataset(dataset, root_path, shots, subsample, dalle=True):
    return dataset_list[dataset](root_path, shots, subsample, dalle=dalle)