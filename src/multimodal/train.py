# trainer.py
from dataclasses import dataclass

import torch
from torch.amp import GradScaler, autocast

from multimodal.tasks import BaseTask


@dataclass
class TrainerConfig:
    max_epochs: int
    grad_accum_steps: int = 1
    mixed_precision: bool = True
    log_every: int = 100
    clip_grad_norm: float|None = None
    device: str = "cuda"


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tasks: list[BaseTask],
        optimizer: torch.optim.Optimizer,
        config: TrainerConfig
    ):
        self.model = model
        self.tasks = tasks
        self.optimizer = optimizer
        self.config = config

        self.scaler = GradScaler(enabled=config.mixed_precision)

        self.model.to(config.device)

    # -----------------------------------------------------
    # Public training loop
    # -----------------------------------------------------
    def train(self, train_loader, val_loader=None):
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

