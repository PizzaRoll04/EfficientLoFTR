import numpy as np
from pathlib import Path

if __name__ == "__main__":
    dir_val = Path(
        "/home/pizzaroll04/dev/TRLoFTR/data/megadepth/index/scene_info_val_1500"
    )
    paths_npz = [f for f in dir_val.glob("*.npz")]
    scene_infos = [np.load(f, allow_pickle=True) for f in paths_npz]
    print(list(scene_infos[0].keys()))

    print(scene_infos[0]["pair_infos"][0])
