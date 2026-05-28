# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch


def epoch_saving(epoch, model, fusion_model, optimizer, filename,
                 model_flow=None, fusion_model_flow=None, alpha=None):
    """Save a training checkpoint.

    Args:
        epoch, model, fusion_model, optimizer: original arguments (unchanged).
        model_flow (nn.Module|None):     DataParallel-wrapped FlowCLIP.
        fusion_model_flow (nn.Module|None): DataParallel-wrapped visual_prompt
                                            for the flow branch.
        alpha (nn.Parameter|None):       Learnable fusion scalar α ∈ [0, 1].
    """
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'fusion_model_state_dict': fusion_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if model_flow is not None:
        save_dict['flow_model_state_dict'] = model_flow.state_dict()
    if fusion_model_flow is not None:
        save_dict['fusion_flow_state_dict'] = fusion_model_flow.state_dict()
    if alpha is not None:
        save_dict['alpha'] = alpha.item()

    torch.save(save_dict, filename)


def best_saving(working_dir, epoch, model, fusion_model, optimizer,
                model_flow=None, fusion_model_flow=None, alpha=None):
    """Save the best-so-far checkpoint.

    Args:
        working_dir, epoch, model, fusion_model, optimizer: original arguments.
        model_flow, fusion_model_flow, alpha: flow-branch state (optional).
    """
    best_name = '{}/model_best.pt'.format(working_dir)
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'fusion_model_state_dict': fusion_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if model_flow is not None:
        save_dict['flow_model_state_dict'] = model_flow.state_dict()
    if fusion_model_flow is not None:
        save_dict['fusion_flow_state_dict'] = fusion_model_flow.state_dict()
    if alpha is not None:
        save_dict['alpha'] = alpha.item()

    torch.save(save_dict, best_name)
