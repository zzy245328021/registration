#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/4/2 21:18
# @Author  : Eric Ching
from config import LPBAConfig
import torch
from torch.nn import functional as F
from timeit import default_timer as timer
import os
from schedulers import PolyLR
from collections import defaultdict
from utils import mkdir, load_checkpoint, get_learning_rate, save_checkpoint
from utils import set_logger, AverageMeter, time_to_str, init_env, dict_to_csv
from losses import l2_smooth3d, ncc_loss,patch_ncc_loss,residual_complexity_loss,SSIM_3,gradient_loss
from metrics import parse_lable_subpopulation
from model.wnet import WNet3D


from data.lpba import get_loaders
import numpy as np
import nibabel as nib


def requires_grad(param):
    return param.requires_grad

def train_one_peoch(model, loader, optimizer, losses, metrics, epoch, since, clip_grident=True):
    model.train()
    n_batches = len(loader)
    meters = defaultdict(AverageMeter)
    for batch_id, (batch_fix, fix_label, batch_move, move_label) in enumerate(loader):
        batch_fix, batch_move = batch_fix.cuda(async=True), batch_move.cuda(async=True)
        batch_warp, batch_df_grid, batch_flow, batch_affine,batch_affine_grid = model(batch_fix, batch_move)

        affine_loss  = losses['patch_ncc'](batch_fix, batch_affine)
        smooth_loss =losses['smooth'](batch_flow)
        sim_loss = losses['patch_ncc'](batch_fix, batch_warp)

        loss = 1000.0*smooth_loss  + affine_loss + sim_loss
        meters['loss'].update(loss.item())
        meters['affine_loss'].update(affine_loss.item())
        meters['smooth'].update(smooth_loss.item()*1000)
        meters['sim_loss'].update(sim_loss.item())

        optimizer.zero_grad()
        loss.backward()
        if clip_grident:
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.)
        optimizer.step()
        str_list = ['%s: %8.5f' % (item[0], item[1].avg) for item in meters.items()]
        print('Epoch %d ' % epoch,
              'Batch %d|%d  ' % (batch_id, n_batches),
              ' || '.join(str_list),
              'Time: %s' % time_to_str((timer() - since), 'min'))
        

    return meters

def eval_one_epoch(model, loader, metrics, epoch, save_res=True):
    n_batches = len(loader)
    since = timer()
    model.eval()
    meters = defaultdict(AverageMeter)
    save_dir = os.path.join(cfg.log_dir, 'epoch'+str(epoch))
    with torch.no_grad():
        for batch_id, (batch_fix, fix_label, batch_move, move_label) in enumerate(loader):
            batch_fix, batch_move = batch_fix.cuda(async=True), batch_move.cuda(async=True)
            batch_warp, batch_df_grid, batch_flow, batch_affine,batch_affine_grid = model(batch_fix, batch_move)
            batch_move_label = move_label.cuda()
            batch_affine_label = F.grid_sample(batch_move_label, batch_affine_grid, mode='nearest')
            batch_warp_label = F.grid_sample(batch_affine_label, batch_df_grid, mode='nearest')
            warp_label = torch.squeeze(batch_warp_label.cpu())
            metric_fn = metrics['sub_dice']
            fix_label = torch.squeeze(fix_label)
            move_label = torch.squeeze(move_label)
            sub_dice = metric_fn(warp_label.numpy(), fix_label.numpy(), cfg.label)
            mean_dice = [d for d in sub_dice.values()]
            mean_dice = np.mean(mean_dice)
            meters['dice'].update(mean_dice)
            str_list = ['%s: %8.5f' % (item[0], item[1].avg) for item in meters.items()]
            print('Epoch %d ' % epoch,
                  'Batch %d|%d  ' % (batch_id, n_batches),
                  ' || '.join(str_list),
                  'Time: %s' % time_to_str((timer() - since), 'min'))
            if save_res:
                save_volume(batch_fix, os.path.join(save_dir, str(batch_id) + "_fix.nii.gz"))
                save_volume(batch_move, os.path.join(save_dir, str(batch_id) + "_move.nii.gz"))
                save_volume(batch_warp, os.path.join(save_dir, str(batch_id) + "_warp.nii.gz"))
                save_volume(batch_affine, os.path.join(save_dir, str(batch_id) + "_affine.nii.gz"))
                save_mask(fix_label.numpy(), os.path.join(save_dir, str(batch_id) + "_fix_label.nii.gz"))
                save_mask(move_label.numpy(), os.path.join(save_dir, str(batch_id) + "_move_label.nii.gz"))
                save_mask(warp_label.numpy(), os.path.join(save_dir, str(batch_id) + "_warp_label.nii.gz"))
                dict_to_csv(sub_dice, os.path.join(save_dir, str(batch_id) + "_sub_dic.csv"))
    return meters

def save_volume(batch_x, name):
    volume = batch_x.cpu().numpy()
    volume = np.squeeze(volume) * 255
    volume = volume.astype("uint8")
    volume = nib.Nifti1Image(volume, np.eye(4))
    nib.save(volume, name)

def save_mask(pred_mask, name):
    pred_mask = pred_mask.astype("uint8")
    volume = nib.Nifti1Image(pred_mask, np.eye(4))
    nib.save(volume, name)

def train_eval(model, loaders, optimizer, scheduler, losses, metrics):
    start = timer()
    train_meters = defaultdict(list)
    val_meters = defaultdict(list)
    for epoch in range(0, cfg.epoch):
        scheduler.step(epoch)
        mkdir(os.path.join(cfg.log_dir, 'epoch'+str(epoch)))
        cur_lr = get_learning_rate(optimizer)
        print('Learning rate is ', cur_lr)
        meter = train_one_peoch(model, loaders['train'], optimizer, losses, metrics, epoch, start)
        train_meters['loss'].append(meter['loss'].avg)
        meter = eval_one_epoch(model, loaders['val'], metrics, epoch)
        val_meters['dice'].append(meter['dice'].avg)
        file_name = os.path.join(cfg.log_dir, 'epoch'+str(epoch), "epoch_{}.pth".format(epoch))
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        if np.argmax(val_meters['dice']) == epoch:
            save_checkpoint(state, True, file_name)
        else:
            save_checkpoint(state, False, file_name)

def train_baseline():
    task_name = ""
    cfg.log_dir = os.path.join(cfg.log_dir, task_name)
    mkdir(cfg.log_dir)
    set_logger(os.path.join(cfg.log_dir, task_name + '.log'))
    loaders = get_loaders(cfg)
    model = WNet3D(use_dialte=True).cuda()
    lr = cfg.lr
    print('initial learning rate is ', lr)
    optimizer = torch.optim.Adam(filter(requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    losses = {'patch_ncc': patch_ncc_loss, 'smooth': l2_smooth3d}
    metrics = {'sub_dice': parse_lable_subpopulation}
    scheduler = PolyLR(optimizer, max_epoch=cfg.epoch)
    train_eval(model, loaders, optimizer, scheduler, losses, metrics=metrics)

if __name__ == '__main__':
    init_env('3')
    cfg = LPBAConfig()
    train_baseline()

