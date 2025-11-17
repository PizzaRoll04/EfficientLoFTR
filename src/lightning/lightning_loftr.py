import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl

from src.loftr import LoFTR
from src.loftr.utils.supervision import (
    compute_supervision_coarse,
    compute_supervision_fine,
)
from src.losses.loftr_loss import LoFTRLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics,
)
from src.utils.plotting import make_matching_figures
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler

from torch.profiler import profile


def reparameter(matcher):
    module = matcher.backbone.layer0
    if hasattr(module, "switch_to_deploy"):
        module.switch_to_deploy()
    for modules in [
        matcher.backbone.layer1,
        matcher.backbone.layer2,
        matcher.backbone.layer3,
    ]:
        for module in modules:
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
    for modules in [
        matcher.fine_preprocess.layer2_outconv2,
        matcher.fine_preprocess.layer1_outconv2,
    ]:
        for module in modules:
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
    return matcher


class PL_LoFTR(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.test_step_outputs = []
        self.config = config  # full config
        _config = lower_config(self.config)
        self.loftr_cfg = lower_config(_config["loftr"])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(
            config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1
        )

        # Matcher: LoFTR
        self.matcher = LoFTR(config=_config["loftr"], profiler=self.profiler)
        self.loss = LoFTRLoss(_config)

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(
                pretrained_ckpt, map_location="cpu", weights_only=False
            )["state_dict"]
            msg = self.matcher.load_state_dict(state_dict, strict=False)
            logger.info(f"Load '{pretrained_ckpt}' as pretrained checkpoint")

        # Testing
        self.warmup = False
        self.reparameter = False
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.total_ms = 0

    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_closure,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == "linear":
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + (self.trainer.global_step / warmup_step) * abs(
                    self.config.TRAINER.TRUE_LR - base_lr
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            elif self.config.TRAINER.WARMUP_TYPE == "constant":
                # keep lr at base ratio until warmup ends
                for pg in optimizer.param_groups:
                    pg["lr"] = (
                        self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                    )
            else:
                raise ValueError(
                    f"Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}"
                )

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            with torch.autocast(enabled=False, device_type="cuda"):
                compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("LoFTR"):
            with torch.autocast(enabled=self.config.LOFTR.MP, device_type="cuda"):
                self.matcher(batch)

        with self.profiler.profile("Compute fine supervision"):
            with torch.autocast(enabled=False, device_type="cuda"):
                compute_supervision_fine(batch, self.config, self.logger)

        with self.profiler.profile("Compute losses"):
            with torch.autocast(enabled=self.config.LOFTR.MP, device_type="cuda"):
                self.loss(batch)

    def _compute_metrics(self, batch):
        compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
        compute_pose_errors(
            batch, self.config
        )  # compute R_errs, t_errs, pose_errs for each pair

        rel_pair_names = list(zip(*batch["pair_names"]))
        bs = batch["image0"].size(0)
        metrics = {
            # to filter duplicate pairs caused by DistributedSampler
            "identifiers": ["#".join(rel_pair_names[b]) for b in range(bs)],
            "epi_errs": [
                (batch["epi_errs"].reshape(-1, 1))[batch["m_bids"] == b]
                .reshape(-1)
                .cpu()
                .numpy()
                for b in range(bs)
            ],
            "R_errs": batch["R_errs"],
            "t_errs": batch["t_errs"],
            "inliers": batch["inliers"],
            "num_matches": [batch["mconf"].shape[0]],  # batch size = 1 only
        }
        ret_dict = {"metrics": metrics}
        return ret_dict, rel_pair_names

    def training_step(self, batch, batch_idx):
        # ensure buffers exist
        if not hasattr(self, "_train_losses"):
            self._train_losses = []

        self._trainval_inference(batch)

        # logging (batch-level)
        if (
            self.trainer.global_rank == 0
            and self.global_step % self.trainer.log_every_n_steps == 0
        ):
            # scalars
            for k, v in batch["loss_scalars"].items():
                self.logger.experiment.add_scalar(f"train/{k}", v, self.global_step)

            # figures
            if self.config.TRAINER.ENABLE_PLOTTING:
                compute_symmetrical_epipolar_errors(batch)
                figures = make_matching_figures(
                    batch, self.config, self.config.TRAINER.PLOT_MODE
                )
                for k, v in figures.items():
                    self.logger.experiment.add_figure(
                        f"train_match/{k}", v, self.global_step
                    )
                plt.close("all")

        loss = batch["loss"]
        # stash for epoch aggregation (Lightning ≥ 2.0: don't rely on training_epoch_end outputs)
        self._train_losses.append(loss.detach())
        return loss

    def on_train_epoch_end(self):
        # Lightning ≥ 2.0 replacement for training_epoch_end
        if hasattr(self, "_train_losses") and len(self._train_losses) > 0:
            avg_loss = torch.stack(self._train_losses).mean()
            if self.trainer.global_rank == 0:
                self.logger.experiment.add_scalar(
                    "train/avg_loss_on_epoch", avg_loss, global_step=self.current_epoch
                )
            self._train_losses.clear()

    def on_validation_epoch_start(self):
        # reset per-epoch accumulators and enable validate mode for matcher
        self.matcher.fine_matching.validate = True
        from collections import defaultdict as _dd

        self._val_accum = _dd(list)  # maps dataloader_idx -> list of step dicts

    def on_validation_epoch_end(self):
        # Lightning ≥ 2.0 replacement for validation_epoch_end
        from collections import defaultdict

        multi_val_metrics = defaultdict(list)

        # determine epoch index for logging (handle sanity checking)
        cur_epoch = self.current_epoch
        # PL 2.x uses `sanity_checking`; fall back to old name if present
        if getattr(self.trainer, "sanity_checking", False) or getattr(
            self.trainer, "running_sanity_check", False
        ):
            cur_epoch = -1

        # iterate over val loaders (dataloader_idx)
        for valset_idx, outputs in sorted(self._val_accum.items()):
            if len(outputs) == 0:
                continue

            # 1) loss_scalars: dict of list, on cpu
            _loss_scalars = [o["loss_scalars"] for o in outputs]
            loss_scalars = {
                k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars]))
                for k in _loss_scalars[0]
            }

            # 2) metrics: dict of list, numpy
            _metrics = [o["metrics"] for o in outputs]
            metrics = {
                k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics])))
                for k in _metrics[0]
            }

            # aggregate (all ranks compute; only rank 0 writes TB)
            val_metrics_4tb = aggregate_metrics(
                metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config
            )
            for thr in [5, 10, 20]:
                multi_val_metrics[f"auc@{thr}"].append(val_metrics_4tb[f"auc@{thr}"])

            # 3) figures
            _figures = [o["figures"] for o in outputs]
            figures = {
                k: flattenList(gather(flattenList([_me[k] for _me in _figures])))
                for k in _figures[0]
            }

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.add_scalar(
                        f"val_{valset_idx}/avg_{k}", mean_v, global_step=cur_epoch
                    )

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(
                        f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch
                    )

                for k, v in figures.items():
                    for plot_idx, fig in enumerate(v):
                        self.logger.experiment.add_figure(
                            f"val_match_{valset_idx}/{k}/pair-{plot_idx}",
                            fig,
                            cur_epoch,
                            close=True,
                        )
            plt.close("all")

        # log on all ranks for ModelCheckpoint monitoring
        for thr in [5, 10, 20]:
            vals = multi_val_metrics[f"auc@{thr}"]
            if vals:
                # put metric on the same device as the DDP backend (NCCL → GPU)
                v = torch.tensor(float(np.mean(vals)), device=self.device)
                self.log(
                    f"auc@{thr}",
                    v,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        # turn off validate flag and clear accumulators
        self.matcher.fine_matching.validate = False
        self._val_accum.clear()

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        self._trainval_inference(batch)
        ret_dict, _ = self._compute_metrics(batch)

        val_plot_interval = max(
            self.trainer.num_val_batches[dataloader_idx] // self.n_vals_plot, 1
        )
        figures = {self.config.TRAINER.PLOT_MODE: []}
        if batch_idx % val_plot_interval == 0:
            figures = make_matching_figures(
                batch, self.config, mode=self.config.TRAINER.PLOT_MODE
            )

        # Accumulate for epoch-end aggregation (don’t rely on validation_epoch_end outputs arg)
        self._val_accum[dataloader_idx].append(
            {
                **ret_dict,
                "loss_scalars": batch["loss_scalars"],
                "figures": figures,
            }
        )
        plt.close("all")
        # Returning is optional; we aggregate via _val_accum
        return

    def test_step(self, batch, batch_idx):
        if (self.config.LOFTR.BACKBONE_TYPE == "RepVGG") and not self.reparameter:
            self.matcher = reparameter(self.matcher)
            if self.config.LOFTR.HALF:
                self.matcher = self.matcher.eval().half()
            self.reparameter = True

        if not self.warmup:
            if self.config.LOFTR.HALF:
                for i in range(50):
                    self.matcher(batch)
            else:
                with torch.autocast(enabled=self.config.LOFTR.MP, device_type="cuda"):
                    for i in range(50):
                        self.matcher(batch)
            self.warmup = True
            torch.cuda.synchronize()

        if self.config.LOFTR.HALF:
            self.start_event.record()
            self.matcher(batch)
            self.end_event.record()
            torch.cuda.synchronize()
            self.total_ms += self.start_event.elapsed_time(self.end_event)
        else:
            with torch.autocast(enabled=self.config.LOFTR.MP, device_type="cuda"):
                self.start_event.record()
                self.matcher(batch)
                self.end_event.record()
                torch.cuda.synchronize()
                self.total_ms += self.start_event.elapsed_time(self.end_event)

        ret_dict, rel_pair_names = self._compute_metrics(batch)
        self.test_step_outputs.append(ret_dict)

        return ret_dict

    def on_test_epoch_end(self):
        # metrics: dict of list, numpy
        outputs = self.test_step_outputs
        if len(outputs) == 0:
            return

        _metrics = [o["metrics"] for o in outputs]
        metrics = {
            k: flattenList(gather(flattenList([_me[k] for _me in _metrics])))
            for k in _metrics[0]
        }

        if self.trainer.is_global_zero:
            n_pairs = max(1, len(outputs))
            print(
                "Averaged Matching time over {} pairs: {:.2f} ms".format(
                    n_pairs, self.total_ms / n_pairs
                )
            )
            val_metrics_4tb = aggregate_metrics(
                metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config
            )
            logger.info("\n" + pprint.pformat(val_metrics_4tb))

        # clear buffer to free memory
        self.test_step_outputs.clear()
