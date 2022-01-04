#!/usr/bin/env python3
''' Based off of the original Neural Baby Talk repo '''
import os
import yaml
import time
import torch
import numpy as np

import utils.opts as opts
import models.blocks as model

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None

def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)

def train(epoch, opt):
    model.train()

    data_iter = iter(dataloader)
    lm_loss_temp = 0
    bn_loss_temp = 0
    fg_loss_temp = 0
    cider_temp = 0
    rl_loss_temp = 0

    start = time.time()
    for step in range(len(dataloader) - 1):
        data = data_iter.next()
        img, iseq, gts_seq, num, proposals, bboxs, box_mask, img_id = data
        proposals = proposals[:, :max(int(max(num[: 1])), 1), :]
        bboxs = bboxs[:, :int(max(num[:, 2])), :]
        box_mask = box_mask[:, :max(int(max(num[:, 2])), 1), :]

        input_imgs.data.resize_(img.size()).copy_(img)
        input_seqs.data.resize_(iseq.size()).copy_(iseq)
        gt_seqs.data.resize_(gts_seq.size()).copy_(gts_seq)
        input_num.data.resize_(num.size()).copy_(num)
        input_ppls.data.resize_(proposals.size()).copy_(proposals)
        gt_bboxs.data.resize_(bboxs.size()).copy_(bboxs)
        mask_bboxs.data.resize_(box_mask.size()).copy_(box_mask)

        loss = 0
        if opt.self_critical:
            rl_loss, bn_loss, fg_loss, cider_score = model(input_imgs, input_seqs, gt_seqs, input_num, input_ppls, gt_bboxs, mask_bboxs, 'RL')
            cider_tmp += cider_score.sum().data[0] / cider_score.numel()
            loss += (rl_loss.sum() + bn_loss.sum() + fg_loss.sum() / rl_loss.numel())
            rl_loss_tmp += loss.data[0]
        else:
            lm_loss, bn_loss, fg_loss = model(input_imgs, input_seqs, gt_seqs, input_num, input_ppls, gt_bboxs, mask_bboxs, 'MLE')
            loss += (lm_loss.sum() + bn_loss.sum() + fg_loss.sum()) / lm_loss.numel()

            lm_loss_tmp += lm_loss.sum().data[0] / lm_loss.numel()
            bn_loss_tmp += bn_loss.sum().data[0] / lm_loss.numel()
            fg_loss_tmp += fg_loss.sum().data[0] / lm_loss.numel()

        model.zero_grad()
        loss.backward()
        torch.utils.clip_grad_norm(model.parameters(), opt.grad_clip)
        optimizer.step()

        if step % opt.disp_interval == 0 and step != 0:
            end = time.time()
            lm_loss_tmp /= opt.disp_interval
            bn_loss_tmp /= opt.disp_interval
            fg_loss_tmp /= opt.disp_interval
            rl_loss_tmp /= opt.disp_interval

            cider_tmp /= opt.disp_interval
            print("step {}/{} (epoch {}), lm_loss = {:.3f}, bn_loss = {:.3f}, rl_loss = {:.3f}, cider_score = {:.3f}, lr = {:.5f}, time/batch = {:.3f}" \
                .format(step, len(dataloader), epoch, lm_loss_temp, bn_loss_temp, fg_loss_temp, rl_loss_temp, cider_temp, opt.learning_rate, end - start))
            start = time.time()

            lm_loss_tmp = 0
            bn_loss_tmp = 0
            fg_loss_tmp = 0
            cider_tmp = 0
            rl_loss_tmp = 0

        if (iteration % opt.losses_log_every == 0):
            if tf is not None:
                add_summary_value(tf_summary_writer, 'train_loss', loss, iteration)
                add_summary_value(tf_summary_writer, 'learning_rate', opt.learning_rate, iteration)
                if opt.self_critical:
                    add_summary_value(tf_summary_writer, 'cider_score', cider_score.data[0], iteration)
                tf_summary_writer.flush()

            loss_history[iteration] = loss.data[0]
            lr_history[iteration] = opt.learning_rate

if __name__ == '__main__':
    opt = opts.parse_opt()
    if opt.path_opt is not None:
        with open(opt.path_opt, 'r') as handle:
            options_yaml = yaml.load(handle)
        utils.update_values(options_yaml, vars(opt))
    print(opt)
    cudnn.benchmark = True

    from misc.dataloader_coco import DataLoader

    if not os.path.exists(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)

    ''' Load Data '''
    dataset = DataLoader(opt, split='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, \
                                            shuffle=False, num_workers=opt.num_workers)

    dataset_val = DataLoader(opt, split=opt.val_split)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size, \
                                            shuffle=False, num_workers=opt.num_workers)

    input_imgs = torch.FloatTensor(1)
    input_seqs = torch.LongTensor(1)
    input_ppls = torch.FloatTensor(1)
    gt_bboxs = torch.FloatTensor(1)
    mask_bboxs = torch.ByteTensor(1)
    gt_seqs = torch.LongTensor(1)
    input_num = torch.LongTensor(1)

    if opt.cuda:
        input_imgs = input_imgs.cuda()
        input_seqs = input_seqs.cuda()
        gt_seqs = gt_seqs.cuda()
        input_num = input_num.cuda()
        input_ppls = input_ppls.cuda()
        gt_bboxs = gt_bboxs.cuda()
        mask_bboxs = mask_bboxs.cuda()

    input_imgs = torch.autograd.Variable(input_imgs)
    input_seqs = torch.autograd.Variable(input_seqs)
    gt_seqs = torch.autograd.Variable(gt_seqs)
    input_num = torch.autograd.Variable(input_num)
    input_ppls = torch.autograd.Variable(input_ppls)
    gt_bboxs = torch.autograd.Variable(gt_bboxs)
    mask_bboxs = torch.autograd.Variable(mask_bboxs)
        
    ''' Model '''
    train()