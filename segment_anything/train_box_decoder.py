# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Copied from DETR, changed import paths:
# https://github.com/facebookresearch/detr/blob/main/engine.py
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

from ..detr.util import misc as utils

def train_one_epoch(decoder: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    decoder.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        batch_outputs = []
        for vector in batch:
            rgb_vector = vector.get('rgb', torch.empty(0))
            multi_vector = vector.get('multi', torch.empty(0))
            image_vector = torch.cat((rgb_vector, multi_vector), dim=1).to(device)
            if image_vector.numel()==0:
                raise ValueError('Either RGB or Multi vector must be provided to model, but both are empty.')
            sparse_prompt = vector['prompt']['sparse'].to(device)
            position_prompt = vector['prompt']['position'].to(device)

            outputs = decoder(image_vector,
                              position_prompt,
                              sparse_prompt)
            batch_outputs.append(outputs)

        preds = {k : torch.cat([output[k] for output in batch_outputs]) for k in ['pred_boxes', 'pred_logits']}
        targets = [vector['annotation'] for vector in batch]

        loss_dict = criterion(preds, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        #metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}