import pytorch_lightning as pl
import argparse
import pprint
from loguru import logger as loguru_logger

from src.config.default import get_cfg_defaults
from src.utils.profiler import build_profiler

from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_loftr import PL_LoFTR

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- existing app args (unchanged) ---
    parser.add_argument("data_cfg_path", type=str, help="data config path")
    parser.add_argument("main_cfg_path", type=str, help="main config path")
    parser.add_argument("--ckpt_path", type=str, default="weights/indoor_ds.ckpt")
    parser.add_argument("--dump_dir", type=str, default=None)
    parser.add_argument("--profiler_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--thr", type=float, default=None)
    parser.add_argument("--pixel_thr", type=float, default=None)
    parser.add_argument("--ransac", type=str, default=None)
    parser.add_argument("--scannetX", type=int, default=None)
    parser.add_argument("--scannetY", type=int, default=None)
    parser.add_argument("--megasize", type=int, default=None)
    parser.add_argument("--npe", action="store_true", default=False)
    parser.add_argument("--fp32", action="store_true", default=False)
    parser.add_argument("--ransac_times", type=int, default=None)
    parser.add_argument("--rmbd", type=int, default=None)
    parser.add_argument("--deter", action="store_true", default=False)
    parser.add_argument("--half", action="store_true", default=False)
    parser.add_argument("--flash", action="store_true", default=False)

    # --- PL2 trainer args ---
    t = parser.add_argument_group("trainer")
    t.add_argument("--accelerator", default="gpu")
    t.add_argument("--devices", default=1)  # int or "-1" for all
    t.add_argument("--num_nodes", type=int, default=1)  # PL2 still supports this
    t.add_argument("--precision", default="32-true")
    # t.add_argument("--max_epochs", type=int, default=None)
    # t.add_argument("--max_steps", type=int, default=None)
    t.add_argument("--log_every_n_steps", type=int, default=50)
    t.add_argument("--default_root_dir", type=str, default=".")
    t.add_argument("--limit_val_batches", type=float, default=1.0)
    t.add_argument("--num_sanity_val_steps", type=int, default=0)
    t.add_argument("--detect_anomaly", action="store_true")
    t.add_argument("--strategy", type=str, default="auto")
    t.add_argument("--benchmark", action="store_true")  # maps to cudnn.benchmark

    # --- legacy PL1 flags for compatibility (translate later) ---
    parser.add_argument("--gpus", default=None)  # e.g., "-1" or "0,1"

    args = parser.parse_args()

    # Translate legacy flags
    if args.gpus is not None:
        if str(args.gpus).strip() == "-1":
            args.devices = -1
        else:
            # allow "0,1,3" etc.  For PL2 you can pass a list, but keep it simple: count.
            gpu_list = [g for g in str(args.gpus).split(",") if g != ""]
            try:
                args.devices = len(gpu_list)
            except Exception:
                args.devices = 1

    return args


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True


if __name__ == "__main__":
    # parse arguments
    args = parse_args()
    pprint.pprint(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    if args.deter:
        torch.backends.cudnn.deterministic = True
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility

    # tune when testing
    if args.thr is not None:
        config.LOFTR.MATCH_COARSE.THR = args.thr

    if args.scannetX is not None and args.scannetY is not None:
        config.DATASET.SCAN_IMG_RESIZEX = args.scannetX
        config.DATASET.SCAN_IMG_RESIZEY = args.scannetY
    if args.megasize is not None:
        config.DATASET.MGDPT_IMG_RESIZE = args.megasize

    if args.npe:
        if config.LOFTR.COARSE.ROPE:
            assert config.DATASET.NPE_NAME is not None
        if config.DATASET.NPE_NAME is not None:
            if config.DATASET.NPE_NAME == "megadepth":
                config.LOFTR.COARSE.NPE = [
                    832,
                    832,
                    config.DATASET.MGDPT_IMG_RESIZE,
                    config.DATASET.MGDPT_IMG_RESIZE,
                ]  # [832, 832, 1152, 1152]
            elif config.DATASET.NPE_NAME == "scannet":
                config.LOFTR.COARSE.NPE = [
                    832,
                    832,
                    config.DATASET.SCAN_IMG_RESIZEX,
                    config.DATASET.SCAN_IMG_RESIZEX,
                ]  # [832, 832, 640, 640]
    else:
        config.LOFTR.COARSE.NPE = [832, 832, 832, 832]

    if args.ransac_times is not None:
        config.LOFTR.EVAL_TIMES = args.ransac_times

    if args.rmbd is not None:
        config.LOFTR.MATCH_COARSE.BORDER_RM = args.rmbd

    if args.pixel_thr is not None:
        config.TRAINER.RANSAC_PIXEL_THR = args.pixel_thr

    if args.ransac is not None:
        config.TRAINER.POSE_ESTIMATION_METHOD = args.ransac
        if args.ransac == "LO-RANSAC" and config.TRAINER.RANSAC_PIXEL_THR == 0.5:
            config.TRAINER.RANSAC_PIXEL_THR = 2.0

    if args.fp32:
        config.LOFTR.MP = False

    if args.half:
        config.LOFTR.HALF = True
        config.DATASET.FP16 = True
    else:
        config.LOFTR.HALF = False
        config.DATASET.FP16 = False

    if args.flash:
        config.LOFTR.COARSE.NO_FLASH = False

    loguru_logger.info(f"Args and config initialized!")

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_LoFTR(
        config,
        pretrained_ckpt=args.ckpt_path,
        profiler=profiler,
        dump_dir=args.dump_dir,
    )
    loguru_logger.info(f"LoFTR-lightning initialized!")

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"DataModule initialized!")

    # # lightning trainer
    # trainer = pl.Trainer.from_argparse_args(
    #     args, replace_sampler_ddp=False, logger=False
    # )
    # lightning trainer (PL2 style)
    precision = "32-true" if args.fp32 else args.precision
    print("Precision: ", precision)
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=precision,
        # max_epochs=args.max_epochs,
        # max_steps=args.max_steps,
        log_every_n_steps=args.log_every_n_steps,
        default_root_dir=args.default_root_dir,
        detect_anomaly=args.detect_anomaly,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=args.num_sanity_val_steps,
        strategy=args.strategy,
        enable_checkpointing=False,
        logger=False,
        benchmark=args.benchmark,  # enables torch.backends.cudnn.benchmark
    )

    loguru_logger.info(f"Start testing!")
    trainer.test(model, datamodule=data_module, verbose=False)
