# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch.utils.data as data
import os
import os.path
import numpy as np
from numpy.random import randint
import pdb
import io
import time
import pandas as pd
import torchvision
import random
from PIL import Image, ImageOps
import cv2
import numbers
import math
import torch
from RandAugment import RandAugment

class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]
    
class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()

class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                print(len(img_group))
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                print(len(img_group))
                rst = np.concatenate(img_group, axis=2)
                return rst

    
class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class Action_DATASETS(data.Dataset):
    def __init__(self, list_file, labels_file,
                 num_segments=1, new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False, index_bias=1,
                 # --- optical flow parameters (all default to off) ---
                 use_flow=False,
                 flow_root=None,
                 flow_tmpl='flow_{}_{:05d}.jpg',
                 flow_transform=None):
        """
        Args:
            use_flow (bool):      Load optical flow alongside RGB frames.
            flow_root (str|None): Root directory that contains per-video flow
                                  sub-directories named the same as the RGB
                                  video directories.  When None, flow frames
                                  are looked up in the same directory as RGB
                                  frames (record.path).
            flow_tmpl (str):      Filename template for flow files.  Must
                                  accept two format arguments: the component
                                  string ('x' or 'y') and the 1-based frame
                                  index, e.g. 'flow_{}_{:05d}.jpg'.
            flow_transform:       Transform applied to the list of grayscale
                                  PIL flow images (flow_x, flow_y alternating
                                  for each selected frame).  Should produce a
                                  FloatTensor of shape (T*2, H, W).
        """
        self.list_file = list_file
        self.num_segments = num_segments
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop = False
        self.index_bias = index_bias
        self.labels_file = labels_file

        # Flow modality
        self.use_flow = use_flow
        self.flow_root = flow_root
        self.flow_tmpl = flow_tmpl
        self.flow_transform = flow_transform

        if self.index_bias is None:
            if self.image_tmpl == "frame{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1
        self._parse_list()
        self.initialized = False

    def _load_image(self, directory, idx):
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]

    def _load_flow(self, directory, idx):
        """Load optical flow_x and flow_y for frame *idx* as grayscale PIL images.

        Returns:
            [flow_x_img, flow_y_img] — list of two PIL 'L'-mode images.
        """
        flow_x = Image.open(
            os.path.join(directory, self.flow_tmpl.format('x', idx))
        ).convert('L')
        flow_y = Image.open(
            os.path.join(directory, self.flow_tmpl.format('y', idx))
        ).convert('L')
        return [flow_x, flow_y]

    def _get_flow_dir(self, record):
        """Resolve the flow frame directory for a given video record.

        When ``flow_root`` is set, the flow directory is constructed as
        ``flow_root / <video_dir_name>``, where *video_dir_name* is the
        last component of ``record.path``.  This assumes that ``flow_root``
        contains per-video sub-directories with the same names as the RGB
        directories.

        When ``flow_root`` is None, flow frames are expected to live in
        the same directory as the RGB frames (``record.path``).
        """
        if self.flow_root is not None:
            video_dir = os.path.basename(record.path.rstrip(os.sep))
            return os.path.join(self.flow_root, video_dir)
        return record.path
    @property
    def total_length(self):
        return self.num_segments * self.seg_length
    
    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()
    
    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        if record.num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(
                    self.total_length) + randint(record.num_frames // 2),
                    record.num_frames) + self.index_bias
            offsets = np.concatenate((
                np.arange(record.num_frames),
                randint(record.num_frames,
                        size=self.total_length - record.num_frames)))
            return np.sort(offsets) + self.index_bias
        offsets = list()
        ticks = [i * record.num_frames // self.num_segments
                 for i in range(self.num_segments + 1)]

        for i in range(self.num_segments):
            tick_len = ticks[i + 1] - ticks[i]
            tick = ticks[i]
            if tick_len >= self.seg_length:
                tick += randint(tick_len - self.seg_length + 1)
            offsets.extend([j for j in range(tick, tick + self.seg_length)])
        return np.array(offsets) + self.index_bias

    def _get_val_indices(self, record):
        if self.num_segments == 1:
            return np.array([record.num_frames //2], dtype=np.int) + self.index_bias
        
        if record.num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(self.total_length), record.num_frames) + self.index_bias
            return np.array([i * record.num_frames // self.total_length
                             for i in range(self.total_length)], dtype=np.int) + self.index_bias
        offset = (record.num_frames / self.num_segments - self.seg_length) / 2.0
        return np.array([i * record.num_frames / self.num_segments + offset + j
                         for i in range(self.num_segments)
                         for j in range(self.seg_length)], dtype=np.int) + self.index_bias

    def __getitem__(self, index):
        record = self.video_list[index]
        segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        return self.get(record, segment_indices)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

    def get(self, record, indices):
        # ---- RGB frames ------------------------------------------------
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            try:
                seg_imgs = self._load_image(record.path, p)
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(seg_imgs)
        process_data = self.transform(images)

        if not self.use_flow:
            return process_data, record.label

        # ---- Optical flow frames ---------------------------------------
        # For each selected frame index p, load (flow_x_p, flow_y_p).
        # The resulting list alternates: [fx_0, fy_0, fx_1, fy_1, ...].
        # After flow_transform this becomes a (T*2, H, W) FloatTensor where
        # consecutive pairs are (flow_x, flow_y) for each temporal segment.
        flow_dir = self._get_flow_dir(record)
        flows = list()
        for seg_ind in indices:
            p = int(seg_ind)
            try:
                flow_imgs = self._load_flow(flow_dir, p)
            except OSError:
                print('ERROR: Could not read flow frames for index {} '
                      'in "{}"'.format(p, flow_dir))
                raise
            flows.extend(flow_imgs)  # appends [flow_x, flow_y] per frame

        process_flow = self.flow_transform(flows)
        return process_data, process_flow, record.label

    def __len__(self):
        return len(self.video_list)
