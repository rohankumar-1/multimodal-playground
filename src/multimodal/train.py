# trainer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast

from multimodal.model import MultimodalModel
from multimodal.tasks import BaseTask


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
    #: If True, set ``requires_grad=False`` on every encoder in :attr:`MultimodalModel.encoders`.
    #: When True, :attr:`freeze_encoder_modalities` is ignored.
    freeze_all_encoders: bool = False
    #: Freeze only these encoder keys (must exist on the model). Ignored if
    #: :attr:`freeze_all_encoders` is True.
    freeze_encoder_modalities: tuple[str, ...] = ()


class Trainer:
    def __init__(
        self,
        model: MultimodalModel,
        tasks: list[BaseTask],
        optimizer: torch.optim.Optimizer,
        config: TrainerConfig,
    ) -> None:
        self.model = model
        self.tasks = tasks
        self.optimizer = optimizer
        self.config = config

        self.scaler = GradScaler(enabled=config.mixed_precision)

        self.model.to(config.device)

        self._apply_encoder_freezing()

        self._best_val_loss: float = float("inf")

    def _apply_encoder_freezing(self) -> None:
        cfg = self.config
        enc = self.model.encoders

        if cfg.freeze_all_encoders:
            for mod in enc.values():
                for p in mod.parameters():
                    p.requires_grad = False
            return

        for name in cfg.freeze_encoder_modalities:
            if name not in enc:
                raise KeyError(
                    f"freeze_encoder_modalities: unknown modality {name!r}; "
                    f"available: {sorted(enc.keys())}"
                )
            for p in enc[name].parameters():
                p.requires_grad = False

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
            self.model.train()
            print(f"\n===== Epoch {epoch+1}/{self.config.max_epochs} =====")

            train_metrics = self._run_one_epoch(train_loader, train=True)
            print(f"Train: {train_metrics}")

            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    val_metrics = self._run_one_epoch(val_loader, train=False)
                    print(f"Val:   {val_metrics}")

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
                        print(
                            f"Saved best checkpoint (val_loss={val_loss:.6f}) "
                            f"-> {self.config.checkpoint_path}"
                        )

    # -----------------------------------------------------
    # One epoch pass (training or validation)
    # -----------------------------------------------------
    def _run_one_epoch(self, loader, train: bool):
        running_metrics = {}
        step = 0

        self.optimizer.zero_grad(set_to_none=True)

        for batch in loader:
            batch = self._move_to_device(batch)

            if train:
                metrics = self._train_step(batch, step)
            else:
                metrics = self._val_step(batch)

            # aggregate metrics
            for k, v in metrics.items():
                running_metrics.setdefault(k, []).append(v)

            step += 1

        # mean-reduce the metrics
        final_metrics = {
            k: sum(vs) / len(vs)
            for k, vs in running_metrics.items()
        }
        return final_metrics

    # -----------------------------------------------------
    # Training step
    # -----------------------------------------------------
    def _train_step(self, batch, step: int):
        use_amp = self.config.mixed_precision

        with autocast(device_type=self.config.device, enabled=use_amp):
            preds, embs = self.model(batch)

            p0 = next(self.model.parameters())
            total_loss = torch.zeros((), device=p0.device, dtype=p0.dtype)
            metrics: dict[str, float] = {}

            for task in self.tasks:
                task_loss, task_metrics = task.compute_loss(preds, embs, batch)
                total_loss = total_loss + task_loss
                metrics.update(task_metrics)

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

        for task in self.tasks:
            _, task_metrics = task.compute_loss(preds, embs, batch)
            metrics_out.update(task_metrics)

        return metrics_out

    # -----------------------------------------------------
    # Helper to move nested dict batches to device
    # -----------------------------------------------------
    def _move_to_device(self, batch):
        device = self.config.device

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
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "epoch": epoch,
            "best_val_loss": val_loss,
            "val_metrics": val_metrics,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }
        torch.save(payload, out)

