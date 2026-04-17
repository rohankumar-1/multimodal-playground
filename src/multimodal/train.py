# trainer.py
from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator, Sequence
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
    """Construction-time options: device, DDP, AMP, optimization step shape, encoder freezing.

    Per-run settings (epochs, logging, checkpoints, early stopping) belong on
    :meth:`Trainer.train` keyword arguments.
    """

    grad_accum_steps: int = 1
    mixed_precision: bool = True
    clip_grad_norm: float | None = None
    device: str = "cuda"
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


def iter_training_parameters(
    model: nn.Module,
    tasks: Iterable[BaseTask],
    *extra_modules: nn.Module,
) -> Iterator[nn.Parameter]:
    """Parameters to optimize: ``model`` plus any stateful per-task loss modules, deduplicated.

    Use this when building the optimizer so critics and other :class:`~torch.nn.Module`
    losses (see :meth:`BaseTask.trainable_loss_modules`) are trained alongside the model::

        opt = torch.optim.Adam(iter_training_parameters(model, tasks), lr=1e-3)

    Parameters shared between ``model`` and a task loss appear once (dedupe by ``id``).
    Pass optional ``*extra_modules`` for trainable modules not declared on any task.
    """
    seen: set[int] = set()

    def _yield_new(from_params: Iterable[nn.Parameter]) -> Iterator[nn.Parameter]:
        for p in from_params:
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                yield p

    yield from _yield_new(model.parameters())
    for task in tasks:
        for mod in task.trainable_loss_modules():
            yield from _yield_new(mod.parameters())
    for mod in extra_modules:
        yield from _yield_new(mod.parameters())


class Trainer:
    """Runs optimization given ``model``, ``tasks``, ``optimizer``, and :class:`TrainerConfig`.

    Call :meth:`train` with loaders and keyword arguments for **this run** (epochs, logging,
    checkpoints, early stopping). :class:`TrainerConfig` holds device, DDP, AMP, grad
    accumulation, clipping, and encoder freezing—things fixed when the trainer is built.
    """

    def __init__(
        self,
        model: nn.Module,
        tasks: Sequence[BaseTask],
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
        self._epochs_without_improvement: int = 0
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
    def train(
        self,
        train_loader,
        val_loader=None,
        *,
        max_epochs: int,
        log_every: int = 100,
        progress_bar: bool = True,
        metric_precision: int = 4,
        checkpoint_path: str | None = None,
        checkpoint_monitor_key: str | None = None,
        patience: int | None = None,
    ) -> None:
        """Run training for ``max_epochs`` (and optional validation each epoch).

        Args:
            train_loader: Training data iterator.
            val_loader: If set, run validation after each train epoch.
            max_epochs: Number of train/val cycles.
            log_every: Steps between tqdm postfix updates (train/val bar).
            progress_bar: Rank-0 tqdm on each epoch (requires ``tqdm`` if True).
            metric_precision: Float formatting for printed metrics.
            checkpoint_path: Save ``torch.save`` payload when validation improves (needs ``val_loader``).
            checkpoint_monitor_key: Val metric key to minimize; if ``None``, sum of ``*/loss`` keys.
            patience: Early-stop after this many epochs without val improvement; needs ``val_loader``.
        """
        if checkpoint_path and val_loader is None:
            raise ValueError(
                "checkpoint_path is set but val_loader is None; "
                "provide a validation loader to track validation loss."
            )
        if patience is not None:
            if patience < 1:
                raise ValueError("patience must be >= 1 when set.")
            if val_loader is None:
                raise ValueError(
                    "patience is set but val_loader is None; "
                    "provide a validation loader for early stopping."
                )

        self._epochs_without_improvement = 0

        for epoch in range(max_epochs):
            self._set_sampler_epoch(train_loader, epoch)
            self._maybe_warn_non_distributed_sampler(train_loader, train=True)

            self.model.train()
            if is_main_process():
                print(f"\n===== Epoch {epoch+1}/{max_epochs} =====")

            train_metrics = self._run_one_epoch(
                train_loader,
                train=True,
                log_every=log_every,
                progress_bar=progress_bar,
                metric_precision=metric_precision,
            )
            train_metrics = reduce_mean_dict(train_metrics, self._device)
            if is_main_process():
                print(f"Train: {self._format_metrics(train_metrics, metric_precision)}")

            if val_loader is not None:
                self._set_sampler_epoch(val_loader, epoch)
                self._maybe_warn_non_distributed_sampler(val_loader, train=False)

                self.model.eval()
                with torch.no_grad():
                    val_metrics = self._run_one_epoch(
                        val_loader,
                        train=False,
                        log_every=log_every,
                        progress_bar=progress_bar,
                        metric_precision=metric_precision,
                    )
                val_metrics = reduce_mean_dict(val_metrics, self._device)
                if is_main_process():
                    print(f"Val:   {self._format_metrics(val_metrics, metric_precision)}")

                val_loss = self._validation_loss_scalar(
                    val_metrics, checkpoint_monitor_key
                )
                improved = val_loss < self._best_val_loss
                if improved:
                    self._best_val_loss = val_loss
                    self._epochs_without_improvement = 0
                else:
                    self._epochs_without_improvement += 1

                if checkpoint_path and improved:
                    self._save_checkpoint(
                        checkpoint_path,
                        epoch=epoch,
                        val_metrics=val_metrics,
                        val_loss=val_loss,
                    )
                    if is_main_process():
                        print(
                            f"Saved best checkpoint (val_loss={val_loss:.6f}) "
                            f"-> {checkpoint_path}"
                        )
                    barrier()

                if patience is not None:
                    if self._epochs_without_improvement >= patience:
                        if is_main_process():
                            print(
                                f"Early stopping: no validation improvement for "
                                f"{patience} epoch(s) "
                                f"(best val_loss={self._best_val_loss:.6f})."
                            )
                        break

    # -----------------------------------------------------
    # One epoch pass (training or validation)
    # -----------------------------------------------------
    def _run_one_epoch(
        self,
        loader,
        train: bool,
        *,
        log_every: int,
        progress_bar: bool,
        metric_precision: int,
    ):
        running_metrics = {}
        step = 0
        loss_running_sum = 0.0
        loss_running_count = 0

        self.optimizer.zero_grad(set_to_none=True)

        it = loader
        pbar = None
        if progress_bar and is_main_process():
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
                        "install tqdm or pass progress_bar=False to train().",
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

            if "loss" in metrics:
                loss_running_sum += float(metrics["loss"])
                loss_running_count += 1

            step += 1
            if pbar is not None:
                pbar.update(1)
                if log_every and (step % log_every == 0):
                    postfix = self._tqdm_postfix_running_loss(
                        loss_running_sum, loss_running_count, metric_precision
                    )
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

    def _tqdm_postfix_running_loss(
        self, loss_sum: float, loss_count: int, metric_precision: int
    ) -> dict[str, float] | None:
        """Postfix for tqdm: mean ``loss`` over all batches seen so far this epoch."""
        if loss_count <= 0:
            return None
        prec = int(metric_precision)
        try:
            avg = loss_sum / loss_count
            return {"loss": float(round(avg, prec))}
        except Exception:
            return None

    def _format_metrics(self, metrics: dict[str, float], metric_precision: int) -> str:
        prec = int(metric_precision)
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
                task_loss = task.compute_loss(preds, embs, batch)
                total_loss = total_loss + task_loss
                metrics[f"{task.name}/loss"] = task.logged_loss_scalar(task_loss)

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
            task_loss = task.compute_loss(preds, embs, batch)
            total_loss = total_loss + task_loss
            metrics_out[f"{task.name}/loss"] = task.logged_loss_scalar(task_loss)
            metrics_out.update(task.compute_metrics(preds, embs, batch))

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

    def _validation_loss_scalar(
        self,
        val_metrics: dict[str, float],
        checkpoint_monitor_key: str | None,
    ) -> float:
        """Single scalar to minimize for best-checkpoint selection."""
        if checkpoint_monitor_key is not None:
            if checkpoint_monitor_key not in val_metrics:
                raise KeyError(
                    f"checkpoint_monitor_key {checkpoint_monitor_key!r} not in val metrics: "
                    f"{sorted(val_metrics)}"
                )
            return float(val_metrics[checkpoint_monitor_key])

        loss_keys = [k for k in val_metrics if k.endswith("/loss")]
        if not loss_keys:
            raise ValueError(
                "No keys ending with '/loss' in val metrics; "
                "pass checkpoint_monitor_key=... to train() for the metric to minimize."
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
