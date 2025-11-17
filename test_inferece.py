import os

from pathlib import Path

os.chdir("..")
from copy import deepcopy

import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure

from src.loftr import LoFTR, full_default_cfg, opt_cfg_3090, reparameter

# You can choose numerical precision in ['fp32', 'mp', 'fp16']. 'fp16' for best efficiency
# precision = "mp"  # Enjoy near-lossless precision with Mixed Precision (MP) / FP16 computation if you have a modern GPU (recommended NVIDIA architecture >= SM_70).
precision = "fp32"  # Enjoy near-lossless precision with Mixed Precision (MP) / FP16 computation if you have a modern GPU (recommended NVIDIA architecture >= SM_70).
config = deepcopy(opt_cfg_3090)
# config = deepcopy(full_default_cfg)
matcher = LoFTR(config=config)

# matcher.load_state_dict(
#     torch.load(
#         "/home/pizzaroll04/dev/TRLoFTR/weights/eloftr_outdoor.ckpt", weights_only=False
#     )["state_dict"]
# )
matcher.load_state_dict(
    torch.load(
        "/home/pizzaroll04/dev/TRLoFTR/logs/tb_logs/outdoor-ds-128-bs=8-/version_5/checkpoints/last.ckpt"
    )["state_dict"]
)
matcher = reparameter(matcher)  # no reparameterization will lead to low performance

if precision == "fp16":
    matcher = matcher.half()

matcher = matcher.eval().cuda()

# Load example images
img0_pth = "/mnt/data/datasets/4seasons/neighborhood_4_train/recording_2020-12-22_11-54-24_stereo_images_undistorted/recording_2020-12-22_11-54-24/undistorted_images/cam0/1608634464756219648.png"
img1_pth = "/mnt/data/datasets/4seasons/neighborhood_5_train/recording_2021-02-25_13-25-15_stereo_images_undistorted/recording_2021-02-25_13-25-15/undistorted_images/cam0/1614255915935133952.png"
img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
img0_raw = cv2.resize(
    img0_raw, (img0_raw.shape[1] // 32 * 32, img0_raw.shape[0] // 32 * 32)
)  # input size shuold be divisible by 32
img1_raw = cv2.resize(
    img1_raw, (img1_raw.shape[1] // 32 * 32, img1_raw.shape[0] // 32 * 32)
)

if precision == "fp16":
    img0 = torch.from_numpy(img0_raw)[None][None].half().cuda() / 255.0
    img1 = torch.from_numpy(img1_raw)[None][None].half().cuda() / 255.0
else:
    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.0
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.0
cv2.imshow("ooga0", img0_raw)
cv2.imshow("ooga1", img1_raw)
cv2.waitKey(0)
batch = {"image0": img0, "image1": img1}

# Inference with EfficientLoFTR and get prediction
with torch.no_grad():
    if precision == "mp":
        with torch.autocast(enabled=True, device_type="cuda"):
            matcher(batch)
    else:
        matcher(batch)
    mkpts0 = batch["mkpts0_f"].cpu().numpy()
    mkpts1 = batch["mkpts1_f"].cpu().numpy()
    mconf = batch["mconf"].cpu().numpy()

print(mkpts0)
print(mconf)
print(mconf.max())
mconf = (mconf - min(20.0, mconf.min())) / (
    max(30.0, mconf.max()) - min(20.0, mconf.min())
)

color = cm.jet(mconf)
text = [
    "LoFTR",
    "Matches: {}".format(len(mkpts0)),
]
fig = make_matching_figure(
    img0_raw,
    img1_raw,
    mkpts0,
    mkpts1,
    color,
    text=text,
    path="/home/pizzaroll04/dev/TRLoFTR/out",
)
