from datasets import build_dataset
from datasets.imagenet import ImageNet
from datasets.oxford_pets import OxfordPets
from model.transform_all import TransformAll
from model.generate_fewshot_dataset import Generate_Fewshot_Dataset
from dassl.data.transforms import build_transform
from dassl.data.data_manager import build_data_loader
from PIL import Image
import os

class DataManager:
    def __init__(
        self,
        cfg,
        custom_tfm_train=None,
        custom_tfm_test=None,
        dataset_wrapper=None
    ):
        # Load dataset
        if cfg.DATASET.NAME == "imagenet":
            dataset = ImageNet(cfg)
        else:
            dataset = build_dataset(cfg.DATASET.NAME, cfg.DATASET.ROOT, cfg.DATASET.NUM_SHOTS, cfg.DATASET.SUBSAMPLE_CLASSES)
        syn_dataset = build_dataset(cfg.TRAINER.SynCLIP.SYN_NAME, cfg.DATASET.ROOT, cfg.TRAINER.SynCLIP.SYN_SHOTS, cfg.DATASET.SUBSAMPLE_CLASSES, cfg.TRAINER.SynCLIP.DALLE)
        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test
        # if cfg.DATASET.DALLE:
        #     path_dir = 'extracted_feature/DALLE/' + cfg.DATASET.NAME
        # else:
        #     path_dir = 'extracted_feature/StableDiffusion/' + cfg.DATASET.NAME
        # if not os.path.exists(path_dir):
        #     os.makedirs(path_dir)
        # path_dir = path_dir + '/pathdir.txt'
        # with open(path_dir, 'w') as f:
        #     f.close()
        # dalle_train_data = TransformAll(dalle_dataset, tfm_train, cfg.DATASET.NUM_SHOTS, path_dir)
        # self.real_train_data = TransformAll(dataset, tfm_train, cfg.DATASET.NUM_SHOTS, path_dir)
        syn_train_data = TransformAll(syn_dataset, tfm_train, cfg.TRAINER.SynCLIP.SYN_SHOTS)
        # self.real_train_data = TransformAll(dataset, tfm_train, cfg.DATASET.NUM_SHOTS)
        # Build train_loader_x
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        # Build train_loader_u
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )

        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_base_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test_base,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )
        test_novel_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test_novel,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Attributes
        self.num_classes = dataset.num_classes
        self.num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self.lab2cname = dataset.lab2cname
        self.syn_train_data = syn_train_data
        # Dataset and data-loaders
        self.dataset = dataset
        self.syn_dataset = syn_dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_base_loader = test_base_loader
        self.test_novel_loader = test_novel_loader
        self.test_loader = test_loader
