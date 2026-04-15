# trainer.py
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

from multimodal.distributed import (
    barrier,
    get_rank,
    get_world_size,
    infer_torchrun_env,
    init_distributed,
    is_main_process,
    reduce_mean_dict,
)
from multimodal.tasks import BaseTask


@dataclass
class DDPConfig:
    """DistributedDataParallel and process-group options (used when DDP is enabled)."""

    backend: str = "nccl"
    find_unused_parameters: bool = False
    static_graph: bool = False
    sync_bn: bool = False


@dataclass
class TrainerConfig:
    max_epochs: int
    grad_accum_steps: int = 1
    mixed_precision: bool = True
    log_every: int = 100
    clip_grad_norm: float | None = None
    device: str = "cuda"
    #: If set, save ``torch.save`` checkpoints to this path whenever validation loss
    #: hits a new minimum. Requires ``val_loader`` in :meth:`Trainer.train`.
    checkpoint_path: str | None = None
    #: Metric key in aggregated validation metrics to minimize (e.g. ``"cls/loss"``).
    #: If ``None``, sums all values whose keys end with ``"/loss"`` (multi-task total).
    checkpoint_monitor_key: str | None = None
    #: If True, set ``requires_grad=False`` on every submodule in ``model.encoders``.
    #: When True, :attr:`freeze_encoder_ids` is ignored.
    freeze_all_encoders: bool = False
    #: Freeze only these **encoder tower** keys (must exist in ``model.encoders``). For
    #: :class:`~multimodal.model.ContrastiveModel`, use encoder ids (e.g. ``"vision"``), not
    #: dataloader batch keys like ``"image_aug"``. Ignored if :attr:`freeze_all_encoders`
    #: is True.
    freeze_encoder_ids: tuple[str, ...] = ()
    #: If ``None``, enable DDP when ``torchrun`` sets ``WORLD_SIZE > 1``.
    #: If ``True``, require multi-process launch; if ``False``, never wrap with DDP.
    distributed: bool | None = None
    ddp: DDPConfig = field(default_factory=DDPConfig)
    #: If True, show a tqdm progress bar (rank 0 only under DDP).
    progress_bar: bool = True
    #: Decimal precision when printing metric dicts.
    metric_precision: int = 4


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        tasks: list[BaseTask],
        optimizer: torch.optim.Optimizer,
        config: TrainerConfig,
    ) -> None:
        self.tasks = tasks
        self.optimizer = optimizer
        self.config = config

        env = infer_torchrun_env()
        if config.distributed is False:
            self._use_ddp = False
        elif config.distributed is True:
            if env is None or env.world_size <= 1:
                raise ValueError(
                    "TrainerConfig.distributed=True requires multi-process launch "
                    "(e.g. torchrun with WORLD_SIZE > 1)."
                )
            self._use_ddp = True
        else:
            self._use_ddp = env is not None and env.world_size > 1

        if self._use_ddp:
            init_distributed(backend=config.ddp.backend)
            self._rank = get_rank()
            self._world_size = get_world_size()
            self._local_rank = env.local_rank if env is not None else 0
        else:
            self._rank = 0
            self._world_size = 1
            self._local_rank = 0

        self._device = self._resolve_device(config.device, env)
        self._autocast_device_type = self._device.type

        self._raw_model = model
        self._raw_model.to(self._device)

        self._apply_encoder_freezing()

        if self._use_ddp and config.ddp.sync_bn:
            self._raw_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self._raw_model)

        if self._use_ddp:
            if self._device.type == "cuda":
                dev_ids = [self._local_rank]
                out_dev = self._local_rank
            else:
                dev_ids = None
                out_dev = None
            self.model = DistributedDataParallel(
                self._raw_model,
                device_ids=dev_ids,
                output_device=out_dev,
                find_unused_parameters=config.ddp.find_unused_parameters,
                static_graph=config.ddp.static_graph,
            )
        else:
            self.model = self._raw_model

        self.scaler = GradScaler(enabled=config.mixed_precision)

        self._best_val_loss: float = float("inf")
        self._warned_train_sampler: bool = False
        self._warned_val_sampler: bool = False
        self._warned_tqdm_missing: bool = False

    def _resolve_device(self, device_str: str, env) -> torch.device:
        if not self._use_ddp:
            return torch.device(device_str)
        if device_str.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(self._local_rank)
            return torch.device(f"cuda:{self._local_rank}")
        return torch.device(device_str)

    def _unwrap_model(self) -> nn.Module:
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module
        return self._raw_model

    def _apply_encoder_freezing(self) -> None:
        cfg = self.config
        enc = self._raw_model.encoders

        if cfg.freeze_all_encoders:
            for mod in enc.values():  # ty:ignore[call-non-callable]
                for p in mod.parameters():
                    p.requires_grad = False
            return

        for name in cfg.freeze_encoder_ids:
            if name not in enc:  # ty:ignore[unsupported-operator]
                raise KeyError(
                    f"freeze_encoder_ids: unknown encoder id {name!r}; "
                    f"available: {sorted(enc.keys())}"  # ty:ignore[unresolved-attribute, call-non-callable]
                )
            for p in enc[name].parameters():  # ty:ignore[unresolved-attribute, invalid-argument-type, not-subscriptable]
                p.requires_grad = False

    def _maybe_warn_non_distributed_sampler(self, loader, *, train: bool) -> None:
        if not self._use_ddp or not is_main_process():
            return
        from torch.utils.data.distributed import DistributedSampler

        sampler = getattr(loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            return
        if train and not self._warned_train_sampler:
            warnings.warn(
                "Distributed training is active but the train DataLoader does not use "
                "DistributedSampler; each rank will iterate the same data unless you wrap "
                "the loader with multimodal.distributed.wrap_loader_with_distributed_sampler.",
                UserWarning,
                stacklevel=2,
            )
            self._warned_train_sampler = True
        if not train and not self._warned_val_sampler:
            warnings.warn(
                "Distributed training is active but the validation DataLoader does not use "
                "DistributedSampler; each rank will evaluate the same subset unless you wrap "
                "the loader with multimodal.distributed.wrap_loader_with_distributed_sampler.",
                UserWarning,
                stacklevel=2,
            )
            self._warned_val_sampler = True

    def _set_sampler_epoch(self, loader, epoch: int) -> None:
        from torch.utils.data.distributed import DistributedSampler

        sampler = getattr(loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

    # -----------------------------------------------------
    # Public training loop
    # -----------------------------------------------------
    def train(self, train_loader, val_loader=None):
        if self.config.checkpoint_path and val_loader is None:
            raise ValueError(
                "checkpoint_path is set but val_loader is None; "
                "provide a validation loader to track validation loss."
            )

        for epoch in range(self.config.max_epochs):
            self._set_sampler_epoch(train_loader, epoch)
            self._maybe_warn_non_distributed_sampler(train_loader, train=True)

            self.model.train()
            if is_main_process():
                print(f"\n===== Epoch {epoch+1}/{self.config.max_epochs} =====")

            train_metrics = self._run_one_epoch(train_loader, train=True)
            train_metrics = reduce_mean_dict(train_metrics, self._device)
            if is_main_process():
                print(f"Train: {self._format_metrics(train_metrics)}")

            if val_loader is not None:
                self._set_sampler_epoch(val_loader, epoch)
                self._maybe_warn_non_distributed_sampler(val_loader, train=False)

                self.model.eval()
                with torch.no_grad():
                    val_metrics = self._run_one_epoch(val_loader, train=False)
                val_metrics = reduce_mean_dict(val_metrics, self._device)
                if is_main_process():
                    print(f"Val:   {self._format_metrics(val_metrics)}")

                if self.config.checkpoint_path:
                    val_loss = self._validation_loss_scalar(val_metrics)
                    if val_loss < self._best_val_loss:
                        self._best_val_loss = val_loss
                        self._save_checkpoint(
                            self.config.checkpoint_path,
                            epoch=epoch,
                            val_metrics=val_metrics,
                            val_loss=val_loss,
                        )
                        if is_main_process():
                            print(
                                f"Saved best checkpoint (val_loss={val_loss:.6f}) "
                                f"-> {self.config.checkpoint_path}"
                            )
                        barrier()

    # -----------------------------------------------------
    # One epoch pass (training or validation)
    # -----------------------------------------------------
    def _run_one_epoch(self, loader, train: bool):
        running_metrics = {}
        step = 0

        self.optimizer.zero_grad(set_to_none=True)

        it = loader
        pbar = None
        if self.config.progress_bar and is_main_process():
            try:
                from tqdm.auto import tqdm

                pbar = tqdm(
                    total=len(loader) if hasattr(loader, "__len__") else None,
                    desc="train" if train else "val",
                    leave=False,
                    dynamic_ncols=True,
                )
                it = loader
            except Exception:
                if not self._warned_tqdm_missing:
                    warnings.warn(
                        "progress_bar=True but tqdm is not available; "
                        "install tqdm or set TrainerConfig.progress_bar=False.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self._warned_tqdm_missing = True

        for batch in it:
            batch = self._move_to_device(batch)

            if train:
                metrics = self._train_step(batch, step)
            else:
                metrics = self._val_step(batch)

            # aggregate metrics
            for k, v in metrics.items():
                running_metrics.setdefault(k, []).append(v)

            step += 1
            if pbar is not None:
                pbar.update(1)
                if self.config.log_every and (step % self.config.log_every == 0):
                    postfix = self._tqdm_postfix_loss_only(metrics)
                    if postfix is not None:
                        pbar.set_postfix(postfix)

        if pbar is not None:
            pbar.close()

        # mean-reduce the metrics
        final_metrics = {
            k: sum(vs) / len(vs)
            for k, vs in running_metrics.items()
        }
        return final_metrics

    def _tqdm_postfix_loss_only(
        self, metrics: dict[str, float]
    ) -> dict[str, float] | None:
        """Postfix dict for tqdm: only ``loss`` (batch total over tasks), if present."""
        if "loss" not in metrics:
            return None
        prec = int(self.config.metric_precision)
        try:
            return {"loss": float(round(float(metrics["loss"]), prec))}
        except Exception:
            return None

    def _format_metrics(self, metrics: dict[str, float]) -> str:
        prec = int(self.config.metric_precision)
        items = []
        for k in sorted(metrics.keys()):
            v = metrics[k]
            try:
                items.append(f"{k}={float(v):.{prec}f}")
            except Exception:
                items.append(f"{k}={v}")
        return "{" + ", ".join(items) + "}"

    # -----------------------------------------------------
    # Training step
    # -----------------------------------------------------
    def _train_step(self, batch, step: int):
        use_amp = self.config.mixed_precision

        with autocast(device_type=self._autocast_device_type, enabled=use_amp):
            preds, embs = self.model(batch)

            p0 = next(self.model.parameters())
            total_loss = torch.zeros((), device=p0.device, dtype=p0.dtype)
            metrics: dict[str, float] = {}

            for task in self.tasks:
                task_loss, task_metrics = task.compute_loss(preds, embs, batch)
                total_loss = total_loss + task_loss
                metrics.update(task_metrics)

            metrics["loss"] = float(total_loss.detach().item())

        # backward + grad accumulation
        accum_loss = total_loss / self.config.grad_accum_steps
        self.scaler.scale(accum_loss).backward()

        if (step + 1) % self.config.grad_accum_steps == 0:
            if self.config.clip_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.clip_grad_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        return metrics

    # -----------------------------------------------------
    # Validation step
    # -----------------------------------------------------
    def _val_step(self, batch):
        preds, embs = self.model(batch)
        metrics_out: dict[str, float] = {}
        p0 = next(self.model.parameters())
        total_loss = torch.zeros((), device=p0.device, dtype=p0.dtype)

        for task in self.tasks:
            task_loss, task_metrics = task.compute_loss(preds, embs, batch)
            total_loss = total_loss + task_loss
            metrics_out.update(task_metrics)

        metrics_out["loss"] = float(total_loss.detach().item())
        return metrics_out

    # -----------------------------------------------------
    # Helper to move nested dict batches to device
    # -----------------------------------------------------
    def _move_to_device(self, batch):
        device = self._device

        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        if isinstance(batch, dict):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        if isinstance(batch, list):
            return [self._move_to_device(v) for v in batch]
        return batch

    def _validation_loss_scalar(self, val_metrics: dict[str, float]) -> float:
        """Single scalar to minimize for best-checkpoint selection."""
        key = self.config.checkpoint_monitor_key
        if key is not None:
            if key not in val_metrics:
                raise KeyError(
                    f"checkpoint_monitor_key {key!r} not in val metrics: {sorted(val_metrics)}"
                )
            return float(val_metrics[key])

        loss_keys = [k for k in val_metrics if k.endswith("/loss")]
        if not loss_keys:
            raise ValueError(
                "No keys ending with '/loss' in val metrics; "
                "set checkpoint_monitor_key on TrainerConfig to the metric to minimize."
            )
        return float(sum(val_metrics[k] for k in loss_keys))

    def _save_checkpoint(
        self,
        path: str,
        *,
        epoch: int,
        val_metrics: dict[str, float],
        val_loss: float,
    ) -> None:
        if not is_main_process():
            return

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "epoch": epoch,
            "best_val_loss": val_loss,
            "val_metrics": val_metrics,
            "model_state_dict": self._unwrap_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }
        torch.save(payload, out)
