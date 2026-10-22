# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import pickle
import numpy as np
import torch
import itertools as it
from monai import data, transforms
from monai.data import *
from torch.utils.data import DataLoader, ConcatDataset


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def load_fold(datalist_json, data_dir, fold=0):
    '''Load the fold of the dataset.'''
    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)

    train_files = []
    val_files = []
    for dd in datalist:
        if dd["fold"] != fold:
            train_files.append(dd)
        else:
            val_files.append(dd)

    return train_files, val_files


root = '/data/FreeTumor/'
cache_root = '/data/FreeTumor/cache/'

covid_dir = root + 'Covid19_20/Train_lungmask/'
covid_json = "./jsons/Covid19_20_folds.json"
covid_train_list, covid_val_list = load_fold(covid_json, covid_dir)
covid_cache_dir = cache_root + 'covid/'

TCIAcovid19_dir = root + 'TCIAcovid19/'
TCIAcovid19_json = "./jsons/TCIAcovid19.json"
TCIAcovid19_list = load_decathlon_datalist(TCIAcovid19_json, True, "training", base_dir=TCIAcovid19_dir)
TCIAcovid19_cache_dir = cache_root + 'TCIAcovid19/'

stoic21_dir = root + 'stoic21/'
stoic21_json = "./jsons/stoic21.json"
stoic21_list = load_decathlon_datalist(stoic21_json, True, "training", base_dir=stoic21_dir)
stoic21_cache_dir = cache_root + 'stoic21/'

LIDC_dir = root + 'LIDC/'
LIDC_json = "./jsons/LIDC.json"
LIDC_list = load_decathlon_datalist(LIDC_json, True, "training", base_dir=LIDC_dir)
LIDC_cache_dir = cache_root + 'LIDC/'

MELA_dir = root + 'MELA/'
MELA_json = "./jsons/MELA.json"
MELA_list = load_decathlon_datalist(MELA_json, True, "training", base_dir=MELA_dir)
MELA_cache_dir = cache_root + 'MELA/'


def get_loader_covid(args):

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode="constant"),

            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=args.pos,
                neg=args.neg,
                num_samples=args.sw_batch_size,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                   mode="constant"),
        ]
    )

    covid_train_ds = PersistentDataset(data=covid_train_list,
                                 transform=train_transform,
                                 pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                 cache_dir=covid_cache_dir)

    TCIAcovid19_ds = PersistentDataset(data=TCIAcovid19_list,
                               transform=train_transform,
                               pickle_protocol=pickle.HIGHEST_PROTOCOL,
                               cache_dir=TCIAcovid19_cache_dir)

    stoic21_ds = PersistentDataset(data=stoic21_list,
                                       transform=train_transform,
                                       pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                       cache_dir=stoic21_cache_dir)

    LIDC_ds = PersistentDataset(data=LIDC_list,
                                       transform=train_transform,
                                       pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                       cache_dir=LIDC_cache_dir)

    MELA_ds = PersistentDataset(data=MELA_list,
                                  transform=train_transform,
                                  pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                  cache_dir=MELA_cache_dir)
    covid_ls = []
    for _ in range(20):
        covid_ls.append(covid_train_ds)

    covid_train_ds = ConcatDataset(covid_ls)

    train_ds = ConcatDataset(
        [
            covid_train_ds,
            TCIAcovid19_ds,
            stoic21_ds,
            LIDC_ds,
            MELA_ds
        ])

    if args.task == 'onlylabeled':
        train_ds = covid_train_ds

    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
    )

    val_ds = PersistentDataset(data=covid_val_list,
                                 transform=val_transform,
                                 pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                 cache_dir=covid_cache_dir)
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=False
    )
    loader = [train_loader, val_loader]

    return loader

