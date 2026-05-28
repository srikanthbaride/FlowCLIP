# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# Optical flow pre-computation utility.

import cv2
import numpy as np
import os
from PIL import Image


def compute_optical_flow(video_dir, output_dir, image_tmpl='img_{:05d}.jpg'):
    """Compute dense optical flow between consecutive frames and save to disk.

    Uses Farneback's dense optical flow algorithm (OpenCV).  Flow components
    are normalised to [0, 255] and saved as grayscale JPEGs following the
    naming convention expected by Action_DATASETS._load_flow:

        flow_x_{idx:05d}.jpg   — horizontal component (u)
        flow_y_{idx:05d}.jpg   — vertical component   (v)

    Index starts at 1 so that it aligns with the 1-based ``index_bias``
    default in Action_DATASETS.

    Args:
        video_dir (str):   Directory containing RGB frame JPEGs sorted by name.
        output_dir (str):  Destination directory for flow JPEG files.
        image_tmpl (str):  (unused; frames are discovered by directory listing)
    """
    os.makedirs(output_dir, exist_ok=True)

    frame_files = sorted(
        f for f in os.listdir(video_dir) if f.lower().endswith('.jpg')
    )

    if len(frame_files) < 2:
        print(f'[flow_utils] Not enough frames in {video_dir} to compute flow.')
        return

    for i in range(len(frame_files) - 1):
        frame1 = cv2.imread(
            os.path.join(video_dir, frame_files[i]), cv2.IMREAD_GRAYSCALE
        )
        frame2 = cv2.imread(
            os.path.join(video_dir, frame_files[i + 1]), cv2.IMREAD_GRAYSCALE
        )

        if frame1 is None or frame2 is None:
            print(f'[flow_utils] Could not read frame pair ({frame_files[i]}, '
                  f'{frame_files[i+1]}), skipping.')
            continue

        flow = cv2.calcOpticalFlowFarneback(
            frame1, frame2, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # Normalise each component to [0, 255] independently
        flow_x = cv2.normalize(
            flow[..., 0], None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        flow_y = cv2.normalize(
            flow[..., 1], None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        # 1-based index to match dataset index_bias=1
        idx = i + 1
        Image.fromarray(flow_x).save(
            os.path.join(output_dir, f'flow_x_{idx:05d}.jpg')
        )
        Image.fromarray(flow_y).save(
            os.path.join(output_dir, f'flow_y_{idx:05d}.jpg')
        )

    print(f'[flow_utils] Saved {len(frame_files) - 1} flow pairs to {output_dir}')
