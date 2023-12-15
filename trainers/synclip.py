import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import MetricMeter, AverageMeter, load_pretrained_weights, load_checkpoint, mkdir_if_missing
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip

from collections import OrderedDict
from dassl.evaluation import build_evaluator
import numpy as np
from model.datamanager import DataManager
import time
import datetime
import math
from tqdm import tqdm
from model.batch_sample import Batch_Sample
from dassl.metrics import compute_accuracy
from model.promptlearner import VLPromptLearner
from model.text_encoder import TextEncoder
import os

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'SynCLIP',
                      "vision_depth": cfg.TRAINER.SynCLIP.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.SynCLIP.PROMPT_DEPTH_TEXT, 
                      "vision_ctx": cfg.TRAINER.SynCLIP.N_CTX_VISION,
                      "vision_ctx_syn": cfg.TRAINER.SynCLIP.N_CTX_VISION_SYN,                      
                      "language_ctx": cfg.TRAINER.SynCLIP.N_CTX_TEXT}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.syn_prompt = nn.ParameterList([nn.Parameter(torch.empty(cfg.TRAINER.SynCLIP.N_CTX_VISION_SYN, 768))
                                                      for _ in range(cfg.TRAINER.SynCLIP.PROMPT_DEPTH_VISION - 1)])
        
        for single_para in self.syn_prompt:
            nn.init.normal_(single_para, std=0.02)
        self.l1_criterion = nn.L1Loss() # nn.MSELoss()
        
    def triplet_loss(self, fake_features, real_features, labels):
        # similarity_matrix = torch.matmul(fake_features, real_features.t())
        
        # # margin = 0.5
        # # neg = []
        # # for i in range(len(labels)):
        # #     neg.append(similarity_matrix[i][labels != labels[i]].max())
        # # neg = torch.stack(neg)
        # # # pos = similarity_matrix.min(1)[0].unsqueeze(1)    
        # # pos = torch.diagonal(similarity_matrix)
         
        # # loss = F.relu(neg - pos + margin)
        # pos = torch.diagonal(similarity_matrix)
        # loss = 1 - pos.mean()
        # return loss
        # similarity_matrix = torch.norm(fake_features[:, None, :] - real_features[None, :, :], dim=2)
        
        D = fake_features.shape[1]
        similarity_matrix = torch.sum(torch.abs((fake_features[:, None, :] - real_features[None, :, :])), dim=2) / D
        margin = 0.0
        neg = []
        pos = torch.diagonal(similarity_matrix)
        for i in range(len(labels)):
            if len(similarity_matrix[i][labels != labels[i]]) != 0:
                neg.append(similarity_matrix[i][labels != labels[i]].min()) # 这个可能会是空集合的
            else:
                neg.append(pos[i] - margin)        
        neg = torch.stack(neg)
        loss = F.relu(pos - neg + margin) + pos
        # loss = F.relu(pos - neg + margin)
        
        return loss.mean()        
    
    def forward(self, subsample, real_batch, syn_image=None, real_label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        real_image_features_1 = self.image_encoder(real_batch.type(self.dtype))
        real_image_features = real_image_features_1 / real_image_features_1.norm(dim=-1, keepdim=True)

        if real_label is not None:
            if subsample == "base":
                logits_1 = logit_scale * real_image_features @ text_features[:math.ceil(prompts.shape[0] / 2)].t()
                syn_label = torch.cat([real_label, torch.randint(low=math.ceil(prompts.shape[0] / 2), high=prompts.shape[0], size=(real_batch.shape[0],)).cuda()])
                # syn_label = syn_label.repeat_interleave(2)
            else:
                logits_1 = logit_scale * real_image_features @ text_features.t()
                syn_label = real_label

            loss_real = F.cross_entropy(logits_1, real_label)
            syn_batch = Batch_Sample(syn_image, syn_label)
            syn_image_features_1 = self.image_encoder(syn_batch.type(self.dtype), self.syn_prompt)
            syn_image_features = syn_image_features_1 / syn_image_features_1.norm(dim=-1, keepdim=True)        
            
            seen_dim = math.ceil(prompts.shape[0] / 2)
            seen_sample = syn_image_features.shape[0] // 2                
            
            # logits_g_seen = logit_scale * syn_image_features[:seen_sample, :] @ text_features[:seen_dim].t()

            logits_g_seen = logit_scale * syn_image_features[:seen_sample, :] @ text_features.t()
            logits_g_unseen = logit_scale * syn_image_features[seen_sample:, :] @ text_features.t()
            loss_g_unseen = F.cross_entropy(logits_g_unseen, syn_label[seen_sample:])
            loss_g_seen = F.cross_entropy(logits_g_seen, syn_label[:seen_sample])
            loss_syn = loss_g_seen + loss_g_unseen
        
            # if subsample == "base":
            #     loss_3 = self.kl_criterion(syn_image_features[:syn_image_features.shape[0] // 2], real_image_features)
            # else:
            #     loss_3 = self.kl_criterion(syn_image_features, real_image_features)
            
            if subsample == "base":                
                loss_align = self.triplet_loss(syn_image_features_1[:syn_image_features.shape[0] // 2], real_image_features_1, real_label) # + self.triplet_loss(real_image_features_1, syn_image_features_1[:syn_image_features.shape[0] // 2].detach(), real_label)                 
            else:
                pass
                
            return loss_real, loss_syn, loss_align, logits_1
        else:
            if subsample == "base":
                logits = logit_scale * real_image_features @ text_features[:math.ceil(prompts.shape[0] / 2)].t()
            else:
                logits = logit_scale * real_image_features @ text_features[math.ceil(prompts.shape[0] / 2):].t()
            logits_open_vocabu = logit_scale * real_image_features @ text_features.t()
            return logits, logits_open_vocabu


@TRAINER_REGISTRY.register()
class SynCLIP(TrainerX):
    def __init__(self, cfg):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.output_dir = cfg.OUTPUT_DIR
        self.weight = cfg.TRAINER.SynCLIP.WEIGHT
        self.kl_weight = cfg.TRAINER.SynCLIP.KLWEIGHT
        
        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = -np.inf
        self.subsample = cfg.DATASET.SUBSAMPLE_CLASSES

    def check_cfg(self, cfg):
        assert cfg.TRAINER.SynCLIP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.SynCLIP.PREC == "fp32" or cfg.TRAINER.SynCLIP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name and "syn_prompt" not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name or "syn_projection" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("SynCLIP", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.SynCLIP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_base_loader = dm.test_base_loader
        self.test_novel_loader = dm.test_novel_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_base_class = math.ceil(self.num_classes / 2)
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}
        self.syn_train_data = dm.syn_train_data

        self.dm = dm

    def forward_backward(self, batch):
        real_image, label = self.parse_batch_train(batch)        
        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.SynCLIP.PREC
        if prec != "amp":
            loss_ce_real, loss_syn, loss_kl, logits = model(self.subsample, real_image, syn_image=self.syn_train_data, real_label=label)
            loss = loss_ce_real + self.weight * loss_syn + self.kl_weight * loss_kl
            acc = compute_accuracy(logits, label)[0].item()
            optim.zero_grad()
            loss.backward()
            optim.step()

        info = {
            "loss_ce_real": loss_ce_real,
            "loss_syn": loss_syn,
            # "loss_g_seen": loss_g_seen,
            # "loss_g_unseen": loss_g_unseen,
            "loss_kl": loss_kl,
            "accuracy": acc
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return info

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        end = time.time()
        self.num_batches = len(self.train_loader_x)
        for self.batch_idx, batch in enumerate(self.train_loader_x): # at least 2-shot
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"{losses}"]    
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)
            end = time.time()

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = 0

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def train(self, start_epoch):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = self.cfg.OPTIM.MAX_EPOCH

        self.before_train()
        for self.epoch in range(self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        if self.subsample == "base":
            self.test(subsample="base")
            self.test(subsample="novel")
        else:
            self.test(subsample="all")

    def after_train(self):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test(subsample="base")
            self.test(subsample="novel")

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    @torch.no_grad()
    def test(self, subsample="base", split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")

        if split is None:
            split = self.cfg.TEST.SPLIT

        # data_loader = self.syn_test_loader
        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            if subsample == "base":
                data_loader = self.test_base_loader
            elif subsample == "novel":
                data_loader = self.test_novel_loader
            else:
                data_loader = self.test_loader

        print(f"Evaluate on the base-to-novel *{split}* set")
        self.evaluator.reset()        
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output, _ = self.model(subsample, input)
            self.evaluator.process(output, label)
        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        print(f"Evaluate on the open-vocabulary *{split}* set")
        self.evaluator.reset()
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            _, output = self.model(subsample, input)
            if subsample == "novel":
                label = label + self.num_base_class
            self.evaluator.process(output, label)
        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)