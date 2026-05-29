# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

from datasets.transforms_ss import *
from RandAugment import RandAugment

class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

def get_augmentation(training, config):
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    scale_size = config.data.input_size * 256 // 224
    if training:

        unique = torchvision.transforms.Compose([GroupMultiScaleCrop(config.data.input_size, [1, .875, .75, .66]),
                                                 GroupRandomHorizontalFlip(is_sth='some' in config.data.dataset),
                                                 GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4,
                                                                        saturation=0.2, hue=0.1),
                                                 GroupRandomGrayscale(p=0.2),
                                                 GroupGaussianBlur(p=0.0),
                                                 GroupSolarization(p=0.0)]
                                                )
    else:
        unique = torchvision.transforms.Compose([GroupScale(scale_size),
                                                 GroupCenterCrop(config.data.input_size)])

    common = torchvision.transforms.Compose([Stack(roll=False),
                                             ToTorchFormatTensor(div=True),
                                             GroupNormalize(input_mean,
                                                            input_std)])
    return torchvision.transforms.Compose([unique, common])

def randAugment(transform_train, config):
    print('Using RandAugment!')
    transform_train.transforms.insert(0, GroupTransform(RandAugment(config.data.randaug.N, config.data.randaug.M)))
    return transform_train


def get_flow_augmentation(training, config):
    """Transform pipeline for grayscale optical flow images.

    Mirrors the spatial transforms used for RGB (same crop / resize sizes)
    but skips colour-jitter / grayscale / blur augmentations that are
    meaningless for flow.  Normalises each channel to mean=0.5, std=0.5
    (maps the [0, 1] flow range to approximately [−1, 1]).

    The transform expects a list of PIL 'L'-mode (grayscale) images ordered
    as [flow_x_0, flow_y_0, flow_x_1, flow_y_1, ...] for T segments and
    returns a FloatTensor of shape (T*2, H, W).

    NOTE: Horizontal flip is included to match RGB augmentation but does NOT
    negate the flow_x channel.  For fully correct flip semantics, disable
    ``GroupRandomHorizontalFlip`` or invert flow_x images on the fly.
    """
    scale_size = config.data.input_size * 256 // 224

    if training:
        spatial = torchvision.transforms.Compose([
            GroupMultiScaleCrop(config.data.input_size, [1, .875, .75, .66]),
            GroupRandomHorizontalFlip(is_sth='some' in config.data.dataset),
        ])
    else:
        spatial = torchvision.transforms.Compose([
            GroupScale(scale_size),
            GroupCenterCrop(config.data.input_size),
        ])

    common = torchvision.transforms.Compose([
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize([0.5], [0.5]),   # grayscale normalisation
    ])
    return torchvision.transforms.Compose([spatial, common])
