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

import os
import shutil
import time
from utils.utils import dice, resample_3d
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather
import shutil
from monai.data import decollate_batch
from utils.utils import *
import cv2
import torch.nn.functional as F
from PIL import Image


def train_epoch(model, loader, optimizerG, optimizerD, scaler, epoch, args):
    model.train()
    start_time = time.time()

    fake_avg = AverageMeter()
    seg_avg = AverageMeter()
    loss_D_avg = AverageMeter()

    dice_avg = AverageMeter()

    for idx, batch_data in enumerate(loader):

        image, label = batch_data["image"], batch_data["label"]
        image, label = image.cuda(), label.cuda()

        for param in model.parameters():
            param.grad = None

        # --- generator update ---#
        mode = 'losses_G'
        with autocast(enabled=args.amp):
            fake_loss, seg_loss, dice_acc = model(image, label, mode, args)
            lossG = fake_loss*0.5 + seg_loss

        if args.amp:
            scaler.scale(lossG).backward()
            scaler.step(optimizerG)
            scaler.update()
        else:
            lossG.backward()
            optimizerG.step()

        # --- discriminator update ---#
        mode = 'losses_D'
        with autocast(enabled=args.amp):
            lossD = model(image, label, mode, args)

        if args.amp:
            scaler.scale(lossD).backward()
            scaler.step(optimizerD)
            scaler.update()
        else:
            lossD.backward()
            optimizerD.step()

        fake_avg.update(fake_loss.item(), n=args.batch_size)
        seg_avg.update(seg_loss.item(), n=args.batch_size)
        dice_avg.update(dice_acc.item(), n=args.batch_size)

        loss_D_avg.update(lossD.item(), n=args.batch_size)

        # lrG = optimizerG.param_groups[0]["lr"]
        # lrD = optimizerD.param_groups[0]["lr"]

        if args.rank == 0:
            # print("Epoch:{}/{} {}/{}, fake_gen:{:.4f}, "
            #       "Time:{:.4f}".format(epoch, args.max_epochs, idx, len(loader),
            #                                                    fake_avg.avg,
            #                                                    time.time() - start_time))
            print("Epoch:{}/{} {}/{}, fake_gen:{:.4f}, seg:{:.4f}, fake_discrim:{:.4f}, Dice:{:.4f},  "
                  "Time:{:.4f}".format(epoch, args.max_epochs, idx, len(loader),
                                       fake_avg.avg, seg_avg.avg, loss_D_avg.avg, dice_avg.avg,
                                       time.time() - start_time))
        start_time = time.time()

    for param in model.parameters():
        param.grad = None

    return fake_avg.avg + loss_D_avg.avg


def val_epoch(model, loader, epoch, args):
    vis_path = args.logdir + '/vis_fake'
    if os.path.exists(vis_path):
        shutil.rmtree(vis_path)

    check_dir(vis_path)
    cmap = color_map()

    model.eval()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            mode = 'generate'
            image, label = batch_data["image"], batch_data["label"]

            ls = list(torch.unique(label[0]))
            if 2 not in ls:
                continue

            else:
                original_image = image.clone().cuda()
                image, label = image.cuda(), label.cuda()

                with autocast(enabled=args.amp):
                    _, _, x, y, z = image.size()

                    fake_image, fake_label, seg_logits, dice_acc, new_image, new_label = model(image, label, mode, args)

                    original_image = original_image.data.cpu().numpy()[0][0]
                    fake_image, fake_label = fake_image.data.cpu().numpy()[0][0], fake_label.data.cpu().numpy()[0][0]
                    new_image, new_label = new_image.data.cpu().numpy()[0][0], new_label.data.cpu().numpy()[0][0]

                ls = list(np.unique(fake_label))
                if (2 not in ls) or (3 not in ls):
                    continue

                for i in range(z):
                    ls_per = list(np.unique(fake_label[:, :, i]))

                    if 2 in ls_per or 3 in ls_per:
                        im = fake_image[:, :, i]
                        im = (255 * im).astype(np.uint8)
                        cv2.imwrite(vis_path + '/' + str(i) + '_im.png', im)

                        ori = original_image[:, :, i]
                        ori = (255 * ori).astype(np.uint8)
                        cv2.imwrite(vis_path + '/' + str(i) + '_ori.png', ori)

                        new = new_image[:, :, i]
                        new = (255 * new).astype(np.uint8)
                        cv2.imwrite(vis_path + '/' + str(i) + '_new.png', new)

                        la = fake_label[:, :, i]
                        la = Image.fromarray(la.astype(np.uint8), mode='P')
                        la.putpalette(cmap)
                        la.save(vis_path + '/' + str(i) + '_lab.png')

                        new_la = new_label[:, :, i]
                        new_la = Image.fromarray(new_la.astype(np.uint8), mode='P')
                        new_la.putpalette(cmap)
                        new_la.save(vis_path + '/' + str(i) + '_newlab.png')

                break

    torch.cuda.empty_cache()
    return


def save_checkpoint(model, epoch, args, filename="model.pt", optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
        model,
        train_loader,
        val_loader,
        optimizerG,
        optimizerD,
        args,
        start_epoch,
):
    scaler = None
    if args.amp:
        scaler = GradScaler()

    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizerG=optimizerG, optimizerD=optimizerD, scaler=scaler, epoch=epoch, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
            save_checkpoint(model, epoch, args, filename="model_final.pt")

        if (epoch + 1) % args.val_every == 0:
            val_epoch(
                model,
                train_loader,
                epoch=epoch,
                args=args,
            )

    print("Training Finished !")

    return val_acc_max
