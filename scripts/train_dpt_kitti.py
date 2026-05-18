"""Train DPT head on KITTI improved GT (DINOv3-L backbone frozen).

Usage:
    PYTHONPATH=/home/izi2sgh/MYDATA/quanjie/liren/dinov3_baseline:\
/home/izi2sgh/MYDATA/quanjie/liren/dinov3_baseline/dinov3 \
        python my_baseline/scripts/train_dpt_kitti.py \
            --config my_baseline/configs/dpt_kitti_vitl.yaml

Designed to be self-contained: does not use dinov3.eval.depth.run_train.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

# Self-pathing so the script runs from anywhere
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "dinov3"))

from my_baseline.datasets.kitti_raw import KITTIRawDataset  # noqa: E402
from my_baseline.eval.kitti_eval import KITTIEvaluator  # noqa: E402
from my_baseline.models.dpt_model import DPTDepthModel, DPTModelConfig  # noqa: E402
from dinov3.eval.depth.loss import SigLoss  # noqa: E402


# =============================================================================
# Helpers
# =============================================================================
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg_model: Dict[str, Any]) -> DPTDepthModel:
    mc = DPTModelConfig(
        hf_model_id=cfg_model["hf_model_id"],
        embed_dim=cfg_model["embed_dim"],
        backbone_out_layers=tuple(cfg_model["backbone_out_layers"]),
        dpt_channels=cfg_model["dpt_channels"],
        post_process_channels=tuple(cfg_model["post_process_channels"]),
        readout_type=cfg_model["readout_type"],
        n_hidden_channels=cfg_model["n_hidden_channels"],
        max_depth=cfg_model["max_depth"],
        min_depth=cfg_model["min_depth"],
        detach_backbone_features=cfg_model.get("detach_backbone_features", True),
    )
    return DPTDepthModel(mc)


def build_datasets(cfg_data: Dict[str, Any]):
    train_set = KITTIRawDataset(
        data_path=cfg_data["data_path"],
        anno_path=cfg_data["anno_path"],
        split_file=cfg_data["train_split"],
        height=cfg_data["height"],
        width=cfg_data["width"],
        frame_idxs=tuple(cfg_data["frame_idxs"]),
        is_train=True,
        gt_source=cfg_data["gt_source_train"],
        img_ext=cfg_data["img_ext"],
        color_jitter=cfg_data["color_jitter"],
    )
    val_set = KITTIRawDataset(
        data_path=cfg_data["data_path"],
        anno_path=cfg_data["anno_path"],
        split_file=cfg_data["val_split"],
        height=cfg_data["height"],
        width=cfg_data["width"],
        frame_idxs=(0,),
        is_train=False,
        gt_source=cfg_data["gt_source_val"],
        img_ext=cfg_data["img_ext"],
        color_jitter=False,
    )
    return train_set, val_set


def make_warmup_onecycle(optimizer: torch.optim.Optimizer, cfg_sched: Dict[str, Any]):
    """Linear warmup to peak LR, then cosine decay to peak/final_div_factor."""
    total = int(cfg_sched["total_iter"])
    warm = int(cfg_sched["warmup_iters"])
    final_div = float(cfg_sched["final_div_factor"])

    def lr_lambda(step: int):
        if step < warm:
            return step / max(1, warm)
        # cosine from 1.0 -> 1/final_div over [warm, total]
        progress = (step - warm) / max(1, total - warm)
        progress = min(1.0, max(0.0, progress))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        end_scale = 1.0 / final_div
        return end_scale + (1.0 - end_scale) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Eval loop
# =============================================================================
@torch.no_grad()
def run_eval(model: DPTDepthModel, val_set, cfg: Dict[str, Any], device: str, max_batches: int | None = None):
    model.eval()
    loader = DataLoader(
        val_set,
        batch_size=cfg["eval"]["batch_size"],
        num_workers=cfg["eval"]["num_workers"],
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    aligns = ["median"]
    if cfg["eval"].get("also_report_least_squares", False):
        aligns.append("least_squares")
    evs = {
        a: KITTIEvaluator(
            split=cfg["eval"]["split"],
            align=a,
            min_depth=cfg["model"]["min_depth"],
            max_depth=cfg["model"]["max_depth"],
            ckpt_name=cfg.get("run_name", "run"),
        )
        for a in aligns
    }

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        if "depth_gt" not in batch:
            continue
        x = batch[("color", 0, 0)].to(device, non_blocking=True)
        depth_pred = model(x)  # (B,1,h,w) at model resolution
        # Resize pred to GT resolution happens inside the evaluator (PIL bilinear).
        # We just hand it the pred at model resolution.
        for b in range(x.shape[0]):
            evs[aligns[0]].update(depth_pred[b:b+1], batch["depth_gt"][b:b+1])
            for a in aligns[1:]:
                evs[a].update(depth_pred[b:b+1], batch["depth_gt"][b:b+1])

    results: Dict[str, Dict[str, float]] = {}
    for a in aligns:
        try:
            results[a] = evs[a].compute()
            evs[a].print_table(prefix=f"[eval/{a}] ")
        except RuntimeError:
            print(f"[eval/{a}] no samples")
    return results


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="run only a handful of iters then exit; for sanity")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    run_name = cfg.get("run_name", "run")
    save_dir = cfg["train"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            _ = torch.empty(1, device="cuda")
        except Exception as e:
            print(f"[train] CUDA probe failed ({e.__class__.__name__}); CPU mode")
            device = "cpu"
    print(f"[train] device={device}, run={run_name}, save_dir={save_dir}")

    # ---- model ---------------------------------------------------------
    print("[train] building model ...")
    model = build_model(cfg["model"]).to(device)
    model.train()
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    print(f"[train] trainable params: {n_trainable/1e6:.2f}M (head only)")

    # ---- data ----------------------------------------------------------
    print("[train] building datasets ...")
    train_set, val_set = build_datasets(cfg["data"])
    print(f"[train] train_set={len(train_set)}  val_set={len(val_set)}")
    pin_mem = device == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        shuffle=True,
        pin_memory=pin_mem,
        drop_last=True,
        persistent_workers=cfg["train"]["num_workers"] > 0,
    )

    # ---- optim / sched / loss ------------------------------------------
    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=opt_cfg["lr"],
        betas=(opt_cfg["beta1"], opt_cfg["beta2"]),
        weight_decay=opt_cfg["weight_decay"],
    )
    scheduler = make_warmup_onecycle(optimizer, cfg["scheduler"])
    total_iter = int(cfg["scheduler"]["total_iter"])
    grad_clip = float(cfg["train"]["grad_clip"])
    amp_enabled = bool(cfg["train"]["amp"]) and device == "cuda"
    scaler = torch.amp.GradScaler(device, enabled=amp_enabled) if amp_enabled else None

    loss_fn = SigLoss(
        warm_up=cfg["loss"]["warm_up"],
        warm_iter=cfg["loss"]["warm_iter"],
    ).to(device)

    # ---- train loop ----------------------------------------------------
    log_every = int(cfg["train"]["log_every"])
    eval_every = int(cfg["train"]["eval_every"])
    ckpt_every = int(cfg["train"]["ckpt_every"])
    min_d = float(cfg["model"]["min_depth"])
    max_d = float(cfg["model"]["max_depth"])
    smoke = args.smoke

    step = 0
    t0 = time.time()
    epoch = 0
    history = []

    while step < total_iter:
        epoch += 1
        for batch in train_loader:
            if step >= total_iter:
                break
            x = batch[("color_aug", 0, 0)].to(device, non_blocking=True)
            gt = batch.get("depth_gt", None)
            if gt is None:
                # filenames that miss improved GT slip through; skip
                continue
            gt = gt.to(device, non_blocking=True)
            # gt is at *full* KITTI resolution; resize to model output res.
            target = F.interpolate(gt, size=x.shape[-2:], mode="nearest")
            valid = (target > min_d) & (target < max_d)
            if not valid.any():
                continue

            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(x)
                    loss = loss_fn(pred, target, valid)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x)
                loss = loss_fn(pred, target, valid)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
                optimizer.step()

            scheduler.step()
            step += 1

            if step % log_every == 0 or step == 1:
                lr = optimizer.param_groups[0]["lr"]
                dt = time.time() - t0
                ips = step / max(dt, 1e-6)
                print(f"[train] iter {step:>6}/{total_iter}  ep={epoch}  "
                      f"loss={loss.item():.4f}  lr={lr:.2e}  "
                      f"{ips:.2f} it/s  pred=[{pred.min().item():.2f},{pred.max().item():.2f}]")
                history.append({"step": step, "loss": float(loss.item()), "lr": float(lr)})

            if smoke and step >= 5:
                print("[train] smoke mode: done after 5 iters")
                _save_ckpt(model, optimizer, scheduler, step, save_dir, name="smoke")
                with open(os.path.join(save_dir, "history.json"), "w") as f:
                    json.dump(history, f, indent=2)
                return

            if step % eval_every == 0:
                res = run_eval(model, val_set, cfg, device)
                _dump_eval(res, save_dir, step)
                model.train()

            if step % ckpt_every == 0:
                _save_ckpt(model, optimizer, scheduler, step, save_dir)

    # final
    print("[train] training complete; running final eval ...")
    res = run_eval(model, val_set, cfg, device)
    _dump_eval(res, save_dir, step, name="final")
    _save_ckpt(model, optimizer, scheduler, step, save_dir, name="final")
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("[train] all done.")


def _save_ckpt(model, optimizer, scheduler, step, save_dir, name: str | None = None):
    fname = f"ckpt_{name or step}.pt"
    path = os.path.join(save_dir, fname)
    payload = {
        "step": step,
        "head_state_dict": model.head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(payload, path)
    print(f"[train] saved {path}")


def _dump_eval(res: Dict[str, Dict[str, float]], save_dir: str, step: int, name: str | None = None):
    out = os.path.join(save_dir, f"eval_{name or step}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[train] eval dumped to {out}")


if __name__ == "__main__":
    main()
