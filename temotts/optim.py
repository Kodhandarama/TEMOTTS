"""Adam + Noam schedule (paper Sec. 4.1: beta1 0.9, beta2 0.999, eps 10e-9, 4000 warm-up, lr 10e-4).

Noam in ESPnet `WarmupLR` form, so `peak_lr` is reached at step == warmup:
    lr(step) = peak_lr * warmup^0.5 * min(step^-0.5, step * warmup^-1.5)
"""
import torch


def build_optimizer(params, cfg):
    opt = torch.optim.Adam(params, lr=cfg["peak_lr"], betas=tuple(cfg["betas"]), eps=cfg["eps"],
                           weight_decay=cfg.get("weight_decay", 0.0))
    warmup = cfg["warmup_steps"]

    def noam(step):
        step = step + 1
        return warmup ** 0.5 * min(step ** -0.5, step * warmup ** -1.5)

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, noam)
