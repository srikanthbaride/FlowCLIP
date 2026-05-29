# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu
#
# Extended with optical flow as a third modality (flow_branch).

import os
import clip
import torch
import torch.nn as nn
from datasets import Action_DATASETS
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
import numpy
from modules.Visual_Prompt import visual_prompt
from modules.Flow_Encoder import FlowCLIP          # ← flow modality
from utils.Augmentation import get_augmentation, get_flow_augmentation
from utils.Text_Prompt import *


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)


def validate(epoch, val_loader, classes, device, model, fusion_model,
             config, num_text_aug,
             model_flow=None, fusion_model_flow=None, alpha=None):
    """Compute top-1 / top-5 accuracy on *val_loader*.

    Args (original):
        epoch, val_loader, classes, device, model, fusion_model,
        config, num_text_aug — unchanged from the original interface.

    Args (flow branch — all default to None → RGB-only mode):
        model_flow (DataParallel[FlowCLIP]|None):
            Optical-flow encoder.  When provided and config.data.use_flow is
            True the flow embedding is fused with the RGB embedding before
            similarity is computed.
        fusion_model_flow (DataParallel[visual_prompt]|None):
            Temporal fusion module for the flow branch.
        alpha (nn.Parameter|None):
            Learnable scalar in [0, 1]: fused = α*rgb + (1−α)*flow.
    """
    use_flow = (model_flow is not None) and config.data.get('use_flow', False)

    model.eval()
    fusion_model.eval()
    if use_flow:
        model_flow.eval()
        fusion_model_flow.eval()

    num    = 0
    corr_1 = 0
    corr_5 = 0

    with torch.no_grad():
        text_inputs   = classes.to(device)
        text_features = model.encode_text(text_inputs)

        for iii, batch in enumerate(tqdm(val_loader)):
            # Unpack batch
            if use_flow:
                image, flows, class_id = batch
            else:
                image, class_id = batch

            image = image.view(
                (-1, config.data.num_segments, 3) + image.size()[-2:]
            )
            b, t, c, h, w = image.size()
            class_id = class_id.to(device)

            # ---- RGB branch ----
            image_input    = image.to(device).view(-1, c, h, w)
            image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = fusion_model(image_features)

            # ---- Flow branch ----
            if use_flow:
                # flows: (b, T*2, H, W) → (b*T, 2, H, W)
                flows_input    = flows.to(device).view(-1, 2, h, w)
                flow_features  = model_flow(flows_input).view(b, t, -1)
                flow_features  = fusion_model_flow(flow_features)

                alpha_v = alpha.clamp(0.0, 1.0).to(image_features.dtype)
                image_features = (
                    alpha_v * image_features
                    + (1.0 - alpha_v) * flow_features
                )

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features  /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T)
            similarity = similarity.view(b, num_text_aug, -1).softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)

            values_1, indices_1 = similarity.topk(1, dim=-1)
            values_5, indices_5 = similarity.topk(5, dim=-1)
            num += b
            for i in range(b):
                if indices_1[i] == class_id[i]:
                    corr_1 += 1
                if class_id[i] in indices_5[i]:
                    corr_5 += 1

    top1 = float(corr_1) / num * 100
    top5 = float(corr_5) / num * 100
    wandb.log({"top1": top1})
    wandb.log({"top5": top5})
    print('Epoch: [{}/{}]: Top1: {}, Top5: {}'.format(
        epoch, config.solver.epochs, top1, top5))
    return top1


def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    working_dir = os.path.join(
        './exp',
        config['network']['type'],
        config['network']['arch'],
        config['data']['dataset'],
        args.log_time,
    )
    wandb.init(
        project=config['network']['type'],
        name='{}_{}_{}_{}'.format(
            args.log_time,
            config['network']['type'],
            config['network']['arch'],
            config['data']['dataset'],
        ),
    )
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('test.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, clip_state_dict = clip.load(
        config.network.arch,
        device=device,
        jit=False,
        tsm=config.network.tsm,
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
    )  # Must set jit=False for training  ViT-B/32

    # ------------------------------------------------------------------ #
    # Determine whether optical flow is enabled                            #
    # ------------------------------------------------------------------ #
    use_flow = config.data.get('use_flow', False)

    # ------------------------------------------------------------------ #
    # Transforms                                                           #
    # ------------------------------------------------------------------ #
    transform_val = get_augmentation(False, config)

    if use_flow:
        flow_transform_val = get_flow_augmentation(False, config)
    else:
        flow_transform_val = None

    # ------------------------------------------------------------------ #
    # Models                                                               #
    # ------------------------------------------------------------------ #
    fusion_model = visual_prompt(
        config.network.sim_header, clip_state_dict, config.data.num_segments
    )
    model_text  = TextCLIP(model)
    model_image = ImageCLIP(model)

    model_text   = torch.nn.DataParallel(model_text).cuda()
    model_image  = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()

    wandb.watch(model)
    wandb.watch(fusion_model)

    # ---- Flow branch ----
    if use_flow:
        model_flow = FlowCLIP(model)
        fusion_model_flow = visual_prompt(
            config.network.sim_header, clip_state_dict, config.data.num_segments
        )
        alpha = nn.Parameter(torch.tensor(0.5))

        model_flow        = torch.nn.DataParallel(model_flow).cuda()
        fusion_model_flow = torch.nn.DataParallel(fusion_model_flow).cuda()
    else:
        model_flow = fusion_model_flow = alpha = None

    # ------------------------------------------------------------------ #
    # Dataset                                                              #
    # ------------------------------------------------------------------ #
    val_data = Action_DATASETS(
        config.data.val_list,
        config.data.label_list,
        num_segments=config.data.num_segments,
        image_tmpl=config.data.image_tmpl,
        transform=transform_val,
        random_shift=config.random_shift,
        use_flow=use_flow,
        flow_root=config.data.get('flow_root', None),
        flow_tmpl=config.data.get('flow_tmpl', 'flow_{}_{:05d}.jpg'),
        flow_transform=flow_transform_val,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.data.batch_size,
        num_workers=config.data.workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    # ------------------------------------------------------------------ #
    # FP16 conversion                                                      #
    # ------------------------------------------------------------------ #
    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(model_text)
        clip.model.convert_weights(model_image)
        if use_flow:
            clip.model.convert_weights(model_flow)

    # ------------------------------------------------------------------ #
    # Checkpoint loading                                                   #
    # ------------------------------------------------------------------ #
    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            if use_flow:
                if 'flow_model_state_dict' in checkpoint:
                    model_flow.load_state_dict(checkpoint['flow_model_state_dict'])
                if 'fusion_flow_state_dict' in checkpoint:
                    fusion_model_flow.load_state_dict(checkpoint['fusion_flow_state_dict'])
                if 'alpha' in checkpoint:
                    alpha.data.fill_(checkpoint['alpha'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    classes, num_text_aug, text_dict = text_prompt(val_data)

    best_prec1 = 0.0
    prec1 = validate(
        start_epoch, val_loader, classes, device, model, fusion_model,
        config, num_text_aug,
        model_flow=model_flow,
        fusion_model_flow=fusion_model_flow,
        alpha=alpha,
    )


if __name__ == '__main__':
    main()
