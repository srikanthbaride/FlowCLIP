# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu
#
# Extended with optical flow as a third modality (flow_branch).

import os
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
from modules.Visual_Prompt import visual_prompt
from modules.Flow_Encoder import FlowCLIP          # ← flow modality
from utils.KLLoss import KLLoss
from test import validate
from utils.Augmentation import *                   # includes get_flow_augmentation
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import *
from utils.Text_Prompt import *
from utils.saving import *


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
    shutil.copy('train.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, clip_state_dict = clip.load(
        config.network.arch,
        device=device,
        jit=False,
        tsm=config.network.tsm,
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init,
        joint=config.network.joint,
    )  # Must set jit=False for training  ViT-B/32

    # ------------------------------------------------------------------ #
    # Determine whether optical flow is enabled                            #
    # ------------------------------------------------------------------ #
    use_flow = config.data.get('use_flow', False)

    # ------------------------------------------------------------------ #
    # Augmentation / transforms                                            #
    # ------------------------------------------------------------------ #
    transform_train = get_augmentation(True, config)
    transform_val   = get_augmentation(False, config)

    if config.data.randaug.N > 0:
        transform_train = randAugment(transform_train, config)

    print('train transforms: {}'.format(transform_train.transforms))
    print('val transforms:   {}'.format(transform_val.transforms))

    if use_flow:
        flow_transform_train = get_flow_augmentation(True, config)
        flow_transform_val   = get_flow_augmentation(False, config)
    else:
        flow_transform_train = None
        flow_transform_val   = None

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

    # ---- Flow branch (only when use_flow=True) ----
    if use_flow:
        model_flow = FlowCLIP(model)              # shares the CLIP backbone
        fusion_model_flow = visual_prompt(
            config.network.sim_header, clip_state_dict, config.data.num_segments
        )
        # Learnable fusion scalar α: fused = α * rgb + (1-α) * flow
        alpha = nn.Parameter(torch.tensor(0.5))

        model_flow        = torch.nn.DataParallel(model_flow).cuda()
        fusion_model_flow = torch.nn.DataParallel(fusion_model_flow).cuda()

        wandb.watch(model_flow)
        wandb.watch(fusion_model_flow)
    else:
        model_flow = fusion_model_flow = alpha = None

    # ------------------------------------------------------------------ #
    # Datasets                                                             #
    # ------------------------------------------------------------------ #
    train_data = Action_DATASETS(
        config.data.train_list,
        config.data.label_list,
        num_segments=config.data.num_segments,
        image_tmpl=config.data.image_tmpl,
        random_shift=config.data.random_shift,
        transform=transform_train,
        use_flow=use_flow,
        flow_root=config.data.get('flow_root', None),
        flow_tmpl=config.data.get('flow_tmpl', 'flow_{}_{:05d}.jpg'),
        flow_transform=flow_transform_train,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=config.data.batch_size,
        num_workers=config.data.workers,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    val_data = Action_DATASETS(
        config.data.val_list,
        config.data.label_list,
        random_shift=False,
        num_segments=config.data.num_segments,
        image_tmpl=config.data.image_tmpl,
        transform=transform_val,
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
        pin_memory=False,
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
            # Converts both the shared CLIP backbone AND flow_proj to fp16
            clip.model.convert_weights(model_flow)

    # ------------------------------------------------------------------ #
    # Loss functions                                                       #
    # ------------------------------------------------------------------ #
    loss_img = KLLoss()
    loss_txt = KLLoss()

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

    if config.resume:
        if os.path.isfile(config.resume):
            print(("=> loading checkpoint '{}'".format(config.resume)))
            checkpoint = torch.load(config.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            start_epoch = checkpoint['epoch']
            if use_flow:
                if 'flow_model_state_dict' in checkpoint:
                    model_flow.load_state_dict(checkpoint['flow_model_state_dict'])
                if 'fusion_flow_state_dict' in checkpoint:
                    fusion_model_flow.load_state_dict(checkpoint['fusion_flow_state_dict'])
                if 'alpha' in checkpoint:
                    alpha.data.fill_(checkpoint['alpha'])
            print(("=> loaded checkpoint '{}' (epoch {})".format(
                config.evaluate, start_epoch)))
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.resume)))

    # ------------------------------------------------------------------ #
    # Text prompts & optimiser                                             #
    # ------------------------------------------------------------------ #
    classes, num_text_aug, text_dict = text_prompt(train_data)

    optimizer = _optimizer(config, model, fusion_model)

    if use_flow:
        # Add flow-specific parameters as extra param groups so they are
        # tracked by the same optimiser / lr-scheduler.
        optimizer.add_param_group({
            'params': model_flow.module.flow_proj.parameters(),
            'lr': config.solver.lr * config.solver.f_ratio,
        })
        optimizer.add_param_group({
            'params': fusion_model_flow.parameters(),
            'lr': config.solver.lr * config.solver.f_ratio,
        })
        optimizer.add_param_group({
            'params': [alpha],
            'lr': config.solver.lr * config.solver.f_ratio,
        })

    lr_scheduler = _lr_scheduler(config, optimizer)

    # ------------------------------------------------------------------ #
    # Evaluate-only mode                                                   #
    # ------------------------------------------------------------------ #
    best_prec1 = 0.0
    if config.solver.evaluate:
        prec1 = validate(
            start_epoch, val_loader, classes, device, model, fusion_model,
            config, num_text_aug,
            model_flow=model_flow,
            fusion_model_flow=fusion_model_flow,
            alpha=alpha,
        )
        return

    for k, v in model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    for epoch in range(start_epoch, config.solver.epochs):
        model_image.train()
        model_text.train()
        fusion_model.train()
        if use_flow:
            model_flow.train()
            fusion_model_flow.train()

        for kkk, batch in enumerate(tqdm(train_loader)):
            # Unpack batch — format depends on whether flow is enabled
            if use_flow:
                images, flows, list_id = batch
            else:
                images, list_id = batch

            if config.solver.type != 'monitor':
                if (kkk + 1) == 1 or (kkk + 1) % 10 == 0:
                    lr_scheduler.step(epoch + kkk / len(train_loader))
            optimizer.zero_grad()

            # ---- RGB branch ----
            images = images.view(
                (-1, config.data.num_segments, 3) + images.size()[-2:]
            )
            b, t, c, h, w = images.size()

            text_id = numpy.random.randint(num_text_aug, size=len(list_id))
            texts = torch.stack(
                [text_dict[j][i, :] for i, j in zip(list_id, text_id)]
            )

            images = images.to(device).view(-1, c, h, w)
            texts  = texts.to(device)

            image_embedding = model_image(images)
            image_embedding = image_embedding.view(b, t, -1)
            image_embedding = fusion_model(image_embedding)

            # ---- Flow branch (fused with RGB when use_flow=True) ----
            if use_flow:
                # flows: (b, T*2, H, W) → (b*T, 2, H, W)
                flows_input = flows.to(device).view(-1, 2, h, w)
                flow_embedding = model_flow(flows_input)          # (b*T, D)
                flow_embedding = flow_embedding.view(b, t, -1)    # (b, T, D)
                flow_embedding = fusion_model_flow(flow_embedding) # (b, D)

                # Weighted fusion with learnable α (clamped to [0, 1])
                alpha_v = alpha.clamp(0.0, 1.0).to(image_embedding.dtype)
                fused_embedding = (
                    alpha_v * image_embedding
                    + (1.0 - alpha_v) * flow_embedding
                )
                fused_embedding = fused_embedding / fused_embedding.norm(
                    dim=-1, keepdim=True
                )
                final_embedding = fused_embedding
            else:
                final_embedding = image_embedding

            # ---- Text branch ----
            text_embedding = model_text(texts)
            if config.network.fix_text:
                text_embedding.detach_()

            # ---- Loss ----
            logit_scale = model.logit_scale.exp()
            logits_per_image, logits_per_text = create_logits(
                final_embedding, text_embedding, logit_scale
            )

            ground_truth = torch.tensor(
                gen_label(list_id), dtype=final_embedding.dtype, device=device
            )
            loss_imgs  = loss_img(logits_per_image, ground_truth)
            loss_texts = loss_txt(logits_per_text, ground_truth)
            total_loss = (loss_imgs + loss_texts) / 2

            wandb.log({"train_total_loss": total_loss})
            wandb.log({"train_loss_imgs": loss_imgs})
            wandb.log({"train_loss_texts": loss_texts})
            wandb.log({"lr": optimizer.param_groups[0]['lr']})
            if use_flow:
                wandb.log({"alpha": alpha.item()})

            total_loss.backward()

            # ---- Optimiser step (with AMP fp32/fp16 dance) ----
            if device == "cpu":
                optimizer.step()
            else:
                # Convert shared CLIP params → fp32 for stable optimiser step
                convert_models_to_fp32(model)
                if use_flow:
                    # flow_proj is NOT part of 'model', handle separately
                    convert_models_to_fp32(model_flow.module.flow_proj)
                optimizer.step()
                # Convert back to fp16
                clip.model.convert_weights(model)
                if use_flow:
                    clip.model.convert_weights(model_flow)

            # Clamp α to valid [0, 1] range after each step
            if use_flow:
                with torch.no_grad():
                    alpha.clamp_(0.0, 1.0)

        # ---------------------------------------------------------------- #
        # Validation & checkpoint saving                                    #
        # ---------------------------------------------------------------- #
        if epoch % config.logging.eval_freq == 0:
            prec1 = validate(
                epoch, val_loader, classes, device, model, fusion_model,
                config, num_text_aug,
                model_flow=model_flow,
                fusion_model_flow=fusion_model_flow,
                alpha=alpha,
            )

        is_best    = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('Testing: {}/{}'.format(prec1, best_prec1))
        print('Saving:')
        filename = "{}/last_model.pt".format(working_dir)

        epoch_saving(
            epoch, model, fusion_model, optimizer, filename,
            model_flow=model_flow,
            fusion_model_flow=fusion_model_flow,
            alpha=alpha,
        )
        if is_best:
            best_saving(
                working_dir, epoch, model, fusion_model, optimizer,
                model_flow=model_flow,
                fusion_model_flow=fusion_model_flow,
                alpha=alpha,
            )


if __name__ == '__main__':
    main()
