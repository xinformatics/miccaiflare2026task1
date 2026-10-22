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
from copy import deepcopy
import numpy as np
import torch
import pickle
from monai import data, transforms
from monai.data import *
from monai.transforms import *
from torch.utils.data import DataLoader, ConcatDataset
from utils.data_trans import *


root = '/data/FreeTumor/'
cache_dir = '/data/FreeTumor/cache'

lits_dir = root + 'Dataset003_Liver/'
lits_json = "./jsons/five_fold/dataset_lits_split0.json"
lits_train_list = load_decathlon_datalist(lits_json, True, "training", base_dir=lits_dir)
lits_val_list = load_decathlon_datalist(lits_json, True, "validation", base_dir=lits_dir)

panc_dir = root + 'Dataset007_Pancreas/'
panc_json = './jsons/five_fold/dataset_panc_split0.json'
panc_train_list = load_decathlon_datalist(panc_json, True, "training", base_dir=panc_dir)
panc_val_list = load_decathlon_datalist(panc_json, True, "validation", base_dir=panc_dir)

kits_dir = root + 'Dataset220_KiTS2023/'
kits_json = './jsons/five_fold/dataset_kits_split0.json'
kits_train_list = load_decathlon_datalist(kits_json, True, "training", base_dir=kits_dir)
kits_val_list = load_decathlon_datalist(kits_json, True, "validation", base_dir=kits_dir)

BTCV_dir = root + "BTCV/"
BTCV_jsonlist = "./jsons/btcv.json"
BTCV_list = load_decathlon_datalist(BTCV_jsonlist, True, "training", base_dir=BTCV_dir)

flare_dir = root + "Flare22/"
flare_json = "./jsons/flare22.json"
flare_list = load_decathlon_datalist(flare_json, True, "training", base_dir=flare_dir)

Amos_dir = root + "Amos2022/"
Amos_json = "./jsons/amos.json"
Amos_list = load_decathlon_datalist(Amos_json, True, "training", base_dir=Amos_dir)

Word_dir = root + "WORD/"
Word_json = "./jsons/word.json"
Word_list = load_decathlon_datalist(Word_json, True, "training", base_dir=Word_dir)

flare23_dir = root + "Flare23/"
flare23_json = "./jsons/flare23.json"
flare23_list = load_decathlon_datalist(flare23_json, True, "training", base_dir=flare23_dir)

PANORAMA_dir = root + 'PANORAMA/'
PANORAMA_json = "./jsons/PANORAMA.json"
PANORAMA_list = load_decathlon_datalist(PANORAMA_json, True, "training", base_dir=PANORAMA_dir)

abdomen1k_dir = root + 'AbdomenCT-1K/'
abdomen1k_json = "./jsons/abdomen1k.json"
abdomen1k_list = load_decathlon_datalist(abdomen1k_json, True, "training", base_dir=abdomen1k_dir)

chaos_dir = root + 'CHAOS/'
chaos_json = "./jsons/chaos.json"
chaos_list = load_decathlon_datalist(chaos_json, True, "training", base_dir=chaos_dir)

TCIA_PANC_dir = root + 'Dataset082_TCIA_Pancreas-CT/'
TCIA_PANC_json = "./jsons/tcia_panc.json"
TCIA_PANC_list = load_decathlon_datalist(TCIA_PANC_json, True, "training", base_dir=TCIA_PANC_dir)

spleen_dir = root + 'Dataset009_Spleen/'
spleen_json = "./jsons/spleen.json"
spleen_list = load_decathlon_datalist(spleen_json, True, "training", base_dir=spleen_dir)

colon_dir = root + 'Dataset010_Colon/'
colon_json = "./jsons/colon.json"
colon_list = load_decathlon_datalist(colon_json, True, "training", base_dir=colon_dir)

atlas_dir = root + 'Dataset224_AbdomenAtlas1.0/'
atlas_json = "./jsons/atlas.json"
atlas_list = load_decathlon_datalist(atlas_json, True, "training", base_dir=atlas_dir)


def get_abdomen_ds(args, filter_labels=None):
    base_trans, random_trans = get_trans(args)

    if filter_labels is None:
        abdomen_transform = base_trans + random_trans
    else:
        abdomen_transform = base_trans + [filter_labels] + random_trans

    if args.use_persistent_dataset:
        BTCV_ds = PersistentDataset(data=BTCV_list,
                                    transform=abdomen_transform,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=cache_dir)
        flare_ds = PersistentDataset(data=flare_list,
                                     transform=abdomen_transform,
                                     pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                     cache_dir=cache_dir)
        Amos_ds = PersistentDataset(data=Amos_list,
                                    transform=abdomen_transform,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=cache_dir)
        WORD_ds = PersistentDataset(data=Word_list,
                                    transform=abdomen_transform,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=cache_dir)

        flare23_ds = PersistentDataset(data=flare23_list,
                                        transform=abdomen_transform,
                                        pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                        cache_dir=cache_dir)

        PANORAMA_ds = PersistentDataset(data=PANORAMA_list,
                                    transform=abdomen_transform,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=cache_dir)
        abdomen1k_ds = PersistentDataset(data=abdomen1k_list,
                                     transform=abdomen_transform,
                                     pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                     cache_dir=cache_dir)

        chaos_ds = PersistentDataset(data=chaos_list,
                                    transform=abdomen_transform,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=cache_dir)
        TCIA_PANC_ds = PersistentDataset(data=TCIA_PANC_list,
                                    transform=abdomen_transform,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=cache_dir)

        spleen_ds = PersistentDataset(data=spleen_list,
                                        transform=abdomen_transform,
                                        pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                        cache_dir=cache_dir)
        colon_ds = PersistentDataset(data=colon_list,
                                         transform=abdomen_transform,
                                         pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                         cache_dir=cache_dir)

        atlas_ds = PersistentDataset(data=atlas_list,
                                         transform=abdomen_transform,
                                         pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                         cache_dir=cache_dir)

    else:
        BTCV_ds = data.Dataset(data=BTCV_list, transform=abdomen_transform)
        flare_ds = data.Dataset(data=flare_list,transform=abdomen_transform)
        Amos_ds = data.Dataset(data=Amos_list, transform=abdomen_transform)
        WORD_ds = data.Dataset(data=Word_list, transform=abdomen_transform,)
        flare23_ds = data.Dataset(data=flare23_list,transform=abdomen_transform,)
        PANORAMA_ds = data.Dataset(data=PANORAMA_list, transform=abdomen_transform,)
        abdomen1k_ds = data.Dataset(data=abdomen1k_list, transform=abdomen_transform)
        chaos_ds = data.Dataset(data=chaos_list, transform=abdomen_transform)
        TCIA_PANC_ds = data.Dataset(data=TCIA_PANC_list,transform=abdomen_transform)
        spleen_ds = data.Dataset(data=spleen_list,transform=abdomen_transform)
        colon_ds = data.Dataset(data=colon_list,transform=abdomen_transform)
        atlas_ds = data.Dataset(data=atlas_list,transform=abdomen_transform)

    unlabeled_ds = ConcatDataset(
        [
         BTCV_ds, flare_ds, Amos_ds, WORD_ds,
         flare23_ds,
         chaos_ds, TCIA_PANC_ds,
         spleen_ds, colon_ds,
         PANORAMA_ds,
         abdomen1k_ds,
         atlas_ds,
         ])
    return unlabeled_ds


def get_loader_lits(args):
    base_trans, random_trans = get_trans(args)

    lits_train_trans = base_trans + random_trans
    lits_val_trans = base_trans

    # abdomen
    unlabeled_ds = get_abdomen_ds(args, filter_labels=Filter_to_liver(keys=["label"]))

    if args.use_persistent_dataset:
        print('use persistent')
        lits_train_ds = PersistentDataset(data=lits_train_list, transform=lits_train_trans,
                                          pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                          cache_dir=cache_dir)
        lits_val_ds = PersistentDataset(data=lits_val_list,
                                   transform=lits_val_trans,
                                   pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                   cache_dir=cache_dir)
    else:
        lits_train_ds = data.Dataset(data=lits_train_list, transform=lits_train_trans)
        lits_val_ds = data.Dataset(data=lits_val_list, transform=lits_val_trans)

    lits_ls = []
    for _ in range(10):
        lits_ls.append(lits_train_ds)
    lits_append_train_ds = ConcatDataset(lits_ls)

    train_ds = ConcatDataset(
        [lits_append_train_ds,
         unlabeled_ds
         ])

    if args.task == 'onlylabeled':
        train_ds = lits_train_ds

    train_sampler = Sampler(train_ds) if args.distributed else None

    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
    )
    val_sampler = Sampler(lits_val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(
        lits_val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=False
    )
    loader = [train_loader, val_loader]

    return loader


def get_loader_pancreas(args):
    base_trans, random_trans = get_trans(args)

    panc_train_trans = base_trans + random_trans
    panc_val_trans = base_trans

    # abdomen
    unlabeled_ds = get_abdomen_ds(args, filter_labels=Filter_to_panc(keys=["label"]))

    if args.use_persistent_dataset:
        print('use persistent')
        panc_train_ds = PersistentDataset(data=panc_train_list, transform=panc_train_trans,
                                          pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                          cache_dir=cache_dir)
        panc_val_ds = PersistentDataset(data=panc_val_list,
                                   transform=panc_val_trans,
                                   pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                   cache_dir=cache_dir)
    else:
        panc_train_ds = data.Dataset(data=panc_train_list, transform=panc_train_trans)
        panc_val_ds = data.Dataset(data=panc_val_list, transform=panc_val_trans)

    panc_ls = []
    for _ in range(10):
        panc_ls.append(panc_train_ds)
    panc_append_train_ds = ConcatDataset(panc_ls)

    train_ds = ConcatDataset(
        [panc_append_train_ds,
         unlabeled_ds
         ])

    if args.task == 'onlylabeled':
        train_ds = panc_train_ds

    train_sampler = Sampler(panc_train_ds) if args.distributed else None
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
    )

    val_sampler = Sampler(panc_val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(
        panc_val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=False
    )
    loader = [train_loader, val_loader]

    return loader


def get_loader_kits(args):
    base_trans, random_trans = get_trans(args)

    kits_train_trans = base_trans + [Filter_KiTs_Labels(keys="label")] + random_trans
    kits_val_trans = base_trans

    # abdomen
    unlabeled_ds = get_abdomen_ds(args, filter_labels=Filter_to_kidney(keys=["label"]))

    if args.use_persistent_dataset:
        print('use persistent')
        kits_train_ds = PersistentDataset(data=kits_train_list, transform=kits_train_trans,
                                          pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                          cache_dir=cache_dir)
        kits_val_ds = PersistentDataset(data=kits_val_list,
                                        transform=kits_val_trans,
                                        pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                        cache_dir=cache_dir)
    else:
        kits_train_ds = data.Dataset(data=kits_train_list, transform=kits_train_trans)
        kits_val_ds = data.Dataset(data=kits_val_list, transform=kits_val_trans)

    kits_ls = []
    for _ in range(10):
        kits_ls.append(kits_train_ds)
    kits_append_train_ds = ConcatDataset(kits_ls)

    train_ds = ConcatDataset(
        [kits_append_train_ds,
         unlabeled_ds
         ])

    if args.task == 'onlylabeled':
        train_ds = kits_train_ds

    train_sampler = Sampler(kits_train_ds) if args.distributed else None
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
    )

    val_sampler = Sampler(kits_val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(
        kits_val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=False
    )
    loader = [train_loader, val_loader]

    return loader


def get_loader_all(args):
    base_trans, random_trans = get_trans(args)

    lits_train_trans = base_trans + [Filter_LITS_alltraining_Labels(keys='label')] + random_trans
    lits_val_trans = base_trans + [Filter_LITS_alltraining_Labels(keys='label')]

    panc_train_trans = base_trans + [Filter_PANC_alltraining_Labels(keys='label')] + random_trans
    panc_val_trans = base_trans + [Filter_PANC_alltraining_Labels(keys='label')]

    kits_train_trans = base_trans + [Filter_KiTs_Labels(keys="label"),
                                     Filter_KITS_alltraining_Labels(keys='label')] + random_trans
    kits_val_trans = base_trans + [Filter_KiTs_Labels(keys="label"), Filter_KITS_alltraining_Labels(keys='label')]

    lits_train_ds = PersistentDataset(data=lits_train_list, transform=lits_train_trans,
                                      pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                      cache_dir=lits_cache_dir)
    lits_val_ds = PersistentDataset(data=lits_val_list, transform=lits_val_trans,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=lits_cache_dir)

    panc_train_ds = PersistentDataset(data=panc_train_list, transform=panc_train_trans,
                                      pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                      cache_dir=panc_cache_dir)
    panc_val_ds = PersistentDataset(data=panc_val_list,
                                    transform=panc_val_trans,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=panc_cache_dir)

    kits_train_ds = PersistentDataset(data=kits_train_list, transform=kits_train_trans,
                                      pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                      cache_dir=kits_cache_dir)
    kits_val_ds = PersistentDataset(data=kits_val_list,
                                    transform=kits_val_trans,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=kits_cache_dir)

    print('use persistent')

    train_ds = ConcatDataset([lits_train_ds, lits_train_ds, lits_train_ds, lits_train_ds, panc_train_ds, panc_train_ds, kits_train_ds])
    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
    )

    val_ds = ConcatDataset([lits_val_ds, panc_val_ds, kits_val_ds])
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=False
    )
    loader = [train_loader, val_loader]

    return loader

def get_loader_for_syn(args):
    base_trans, random_trans = get_trans(args)

    lits_train_trans = base_trans + [Filter_LITS_alltraining_Labels(keys='label')] + random_trans
    lits_val_trans = base_trans + [Filter_LITS_alltraining_Labels(keys='label')]

    panc_train_trans = base_trans + [Filter_PANC_alltraining_Labels(keys='label')] + random_trans
    panc_val_trans = base_trans + [Filter_PANC_alltraining_Labels(keys='label')]

    kits_train_trans = base_trans + [Filter_KiTs_Labels(keys="label"),
                                     Filter_KITS_alltraining_Labels(keys='label')] + random_trans
    kits_val_trans = base_trans + [Filter_KiTs_Labels(keys="label"), Filter_KITS_alltraining_Labels(keys='label')]

    lits_train_ds = PersistentDataset(data=lits_train_list, transform=lits_train_trans,
                                      pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                      cache_dir=cache_dir)
    lits_val_ds = PersistentDataset(data=lits_val_list, transform=lits_val_trans,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=cache_dir)

    panc_train_ds = PersistentDataset(data=panc_train_list, transform=panc_train_trans,
                                      pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                      cache_dir=cache_dir)
    panc_val_ds = PersistentDataset(data=panc_val_list,
                                    transform=panc_val_trans,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=cache_dir)

    kits_train_ds = PersistentDataset(data=kits_train_list, transform=kits_train_trans,
                                      pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                      cache_dir=cache_dir)
    kits_val_ds = PersistentDataset(data=kits_val_list,
                                    transform=kits_val_trans,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=cache_dir)

    unlabeled_ds = get_abdomen_ds(args)

    lits_ls = []
    for _ in range(200):
        lits_ls.append(lits_train_ds)
    lits_train_ds = ConcatDataset(lits_ls)

    panc_ls = []
    for _ in range(100):
        panc_ls.append(panc_train_ds)
    panc_train_ds = ConcatDataset(panc_ls)

    kits_ls = []
    for _ in range(100):
        kits_ls.append(kits_train_ds)
    kits_train_ds = ConcatDataset(kits_ls)

    train_ds = ConcatDataset([
                            lits_train_ds, panc_train_ds, kits_train_ds,
                              unlabeled_ds
                              ])

    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
    )

    val_ds = ConcatDataset([lits_val_ds, panc_val_ds, kits_val_ds])
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=False
    )
    loader = [train_loader, val_loader]

    return loader
