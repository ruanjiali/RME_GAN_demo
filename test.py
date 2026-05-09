# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 10:52:13 2022

@author: Achintha
"""

from __future__ import print_function, division
import argparse
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import time
from lib import loaders, modules


def _to_db(x, value_scale, value_offset, db_min, db_max):
    x = x * value_scale + value_offset
    if db_min is None or db_max is None:
        return x
    return (x / 256.0) * (db_max - db_min) + db_min


def _ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def _torch_load_compat(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_generator_state_dict(obj):
    if isinstance(obj, dict) and "netG" in obj:
        return obj["netG"]
    return obj


def _predict_batch(mode, model, inputs, device):
    inputs = inputs.to(device)

    if mode == "model":
        x = inputs[:, :3, :, :]
        outputs, _ = model(x)
        return outputs

    if mode == "baseline_prior":
        return inputs[:, 3:4, :, :]

    if mode == "baseline_samples":
        return inputs[:, 2:3, :, :]

    raise ValueError(f"Unknown mode: {mode}")


def evaluate_with_mode(mode, model, dataset, device, batch_size, num_workers, save_dir, save_images, max_batches, value_scale, value_offset, db_min, db_max, threshold_db):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    abs_err_sum = 0.0
    sq_err_sum = 0.0
    nmse_num = 0.0
    nmse_den = 0.0
    count = 0
    within_thr_count = 0
    total_px = 0
    abs_err_samples = []
    saved = []

    infer_s = 0.0
    with torch.inference_mode():
        if model is not None:
            model.to(device)
            model.eval()

        for bi, (inputs, targets) in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break

            targets = targets.to(device)

            t0 = time.perf_counter()
            outputs = _predict_batch(mode=mode, model=model, inputs=inputs, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            infer_s += time.perf_counter() - t0

            out_db = _to_db(outputs, value_scale, value_offset, db_min, db_max)
            tgt_db = _to_db(targets, value_scale, value_offset, db_min, db_max)

            diff = (out_db - tgt_db).reshape(out_db.shape[0], -1)
            abs_err = diff.abs()
            sq_err = diff.square()

            abs_err_sum += abs_err.sum().item()
            sq_err_sum += sq_err.sum().item()
            count += diff.numel()

            mse_batch = ((outputs - targets) ** 2).mean().item()
            nmse_num += mse_batch * outputs.numel()
            nmse_den += ((targets - torch.zeros_like(targets)) ** 2).mean().item() * targets.numel()

            if threshold_db is not None:
                within_thr_count += (abs_err <= threshold_db).sum().item()
                total_px += abs_err.numel()

            abs_err_samples.append(abs_err.detach().cpu().flatten())

            if save_images:
                _ensure_dir(save_dir)
                for j in range(min(outputs.shape[0], 4)):
                    paths = {
                        "in_buildings": f"in_buildings_b{bi}_i{j}.png",
                        "in_tx": f"in_tx_b{bi}_i{j}.png",
                        "in_samples": f"in_samples_b{bi}_i{j}.png",
                        "in_prior": f"in_prior_b{bi}_i{j}.png",
                        "pred": f"pred_b{bi}_i{j}.png",
                        "gt": f"gt_b{bi}_i{j}.png",
                    }
                    save_image(inputs[j, 0:1].to(torch.float32), os.path.join(save_dir, paths["in_buildings"]), nrow=1, normalize=True)
                    save_image(inputs[j, 1:2].to(torch.float32), os.path.join(save_dir, paths["in_tx"]), nrow=1, normalize=True)
                    save_image(inputs[j, 2:3].to(torch.float32) / 255.0, os.path.join(save_dir, paths["in_samples"]), nrow=1, normalize=True)
                    save_image(inputs[j, 3:4].to(torch.float32) / 255.0, os.path.join(save_dir, paths["in_prior"]), nrow=1, normalize=True)
                    save_image(outputs[j].to(torch.float32) / 255.0, os.path.join(save_dir, paths["pred"]), nrow=1, normalize=True)
                    save_image(targets[j].to(torch.float32) / 255.0, os.path.join(save_dir, paths["gt"]), nrow=1, normalize=True)
                    saved.append({"batch": bi, "index": j, "paths": paths})

    mae = abs_err_sum / max(count, 1)
    rmse = float(np.sqrt(sq_err_sum / max(count, 1)))
    nmse = float(nmse_num / max(nmse_den, 1e-12))

    abs_err_all = torch.cat(abs_err_samples, dim=0) if abs_err_samples else torch.tensor([])
    p90 = float(torch.quantile(abs_err_all, 0.90).item()) if abs_err_all.numel() else float("nan")
    p95 = float(torch.quantile(abs_err_all, 0.95).item()) if abs_err_all.numel() else float("nan")

    within = None
    if threshold_db is not None and total_px > 0:
        within = within_thr_count / total_px

    return {
        "mae": mae,
        "rmse": rmse,
        "p90": p90,
        "p95": p95,
        "nmse": nmse,
        "infer_seconds": infer_s,
        "pixels": count,
        "within_threshold_ratio": within,
        "saved_images": saved,
    }


def _write_metrics_json(out_dir, payload):
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--dataset", default="radiounet", choices=["radiounet", "radiomapseer_polygon"])
    parser.add_argument("--setup", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--mode", default="model", choices=["model", "baseline_prior", "baseline_samples"])
    parser.add_argument("--weights", default=None)
    parser.add_argument("--batch_size", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--out_dir", default="eval_out")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--value_scale", type=float, default=1.0)
    parser.add_argument("--value_offset", type=float, default=0.0)
    parser.add_argument("--db_min", type=float, default=None)
    parser.add_argument("--db_max", type=float, default=None)
    parser.add_argument("--threshold_db", type=float, default=7.0)
    args = parser.parse_args()

    if args.dataset == "radiounet":
        if args.setup == 1:
            ds = loaders.RadioUNet_s(phase="test", fix_samples=655, num_samples_low=10, num_samples_high=300, dir_dataset=args.dataset_dir)
        elif args.setup == 2:
            ds = loaders.RadioUNet_s(phase="test", fix_samples=1, num_samples_low=655, num_samples_high=655*10, dir_dataset=args.dataset_dir)
        else:
            ds = loaders.RadioUNet_s(phase="test", fix_samples=0, num_samples_low=655, num_samples_high=655*10, dir_dataset=args.dataset_dir)
    else:
        if args.setup == 1:
            ds = loaders.RadioMapSeerPolygon(phase="test", fix_samples=655, num_samples_low=10, num_samples_high=300, dir_dataset=args.dataset_dir)
        elif args.setup == 2:
            ds = loaders.RadioMapSeerPolygon(phase="test", fix_samples=1, num_samples_low=655, num_samples_high=655*10, dir_dataset=args.dataset_dir)
        else:
            ds = loaders.RadioMapSeerPolygon(phase="test", fix_samples=0, num_samples_low=655, num_samples_high=655*10, dir_dataset=args.dataset_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = None
    if args.mode == "model":
        if args.weights is None:
            raise SystemExit("--weights is required when --mode=model")
        model = modules.RadioWNet(phase="firstU")
        loaded = _torch_load_compat(args.weights)
        state = _extract_generator_state_dict(loaded)
        model.load_state_dict(state)

    metrics = evaluate_with_mode(
        mode=args.mode,
        model=model,
        dataset=ds,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_dir=args.out_dir,
        save_images=args.save_images,
        max_batches=args.max_batches,
        value_scale=args.value_scale,
        value_offset=args.value_offset,
        db_min=args.db_min,
        db_max=args.db_max,
        threshold_db=args.threshold_db,
    )

    print("device:", device)
    print("mode:", args.mode)
    for k in ["mae", "rmse", "p90", "p95", "nmse", "infer_seconds", "pixels", "within_threshold_ratio"]:
        print(f"{k}: {metrics[k]}")

    payload = {
        "dataset": args.dataset,
        "setup": args.setup,
        "mode": args.mode,
        "db_min": args.db_min,
        "db_max": args.db_max,
        "threshold_db": args.threshold_db,
        "metrics": metrics,
    }
    _write_metrics_json(args.out_dir, payload)


if __name__ == "__main__":
    main()

# with torch.no_grad():
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model = ResnetGenerator(input_nc=2,output_nc=1,ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6)
#     model.load_state_dict(torch.load('RadioWNet_c_DPM_Thr2_CGAN/Trained_Model_G.pt'))
#     model.to(device)
#     model.eval()
#     criterion = nn.MSELoss()
#     loss = []
#     for inps,gts in torch.utils.data.DataLoader(Radio_test,batch_size=15,shuffle=False,num_workers=2):
#         inps,gts = inps.to(device),gts.to(device)
#         R = model(inps)
#         l = criterion(R,gts)
#         loss += [l.item()]
#     print('Test Loss: ', np.mean(np.array(loss)))
