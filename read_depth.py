import h5py
import numpy as np
import matplotlib.pyplot as plt


with h5py.File(
    "/home/pizzaroll04/dev/TRLoFTR/data/megadepth/train/phoenix/S6/zl548/MegaDepth_v1/0000/dense0/depths/5978745_520771b1cb_o.h5"
) as f:
    print(f"Keys: {list(f.keys())}")
    depth = f["depth"][:]

print(f"shape: {depth.shape}")
print(f"dtype: {depth.dtype}")
print(f"min: {np.min(depth)} \tmax: {np.max(depth)}")


# depth[depth <= 0] = np.nan

plt.figure(figsize=(8, 6))
plt.imshow(depth, cmap="plasma", vmin=0, vmax=np.nanpercentile(depth, 95))
plt.colorbar(label="Depth (meters)")
plt.title("MegaDepth Depth Map")
plt.show()
