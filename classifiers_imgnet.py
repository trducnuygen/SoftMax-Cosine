# -*- coding: utf-8 -*-

import argparse
# import math
import os
import random
import shutil
import time
# import warnings

import numpy as np
# import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
# import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from tqdm import tqdm
from src.data import get_dataloader

import getModel as gM
import writeLogAcc as wA

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


p = argparse.ArgumentParser(description="SoftMax-Cosine Rescaling pipeline for ImgNet")

# shared
p.add_argument("-d", "--data",       type=str, required=True,
                help="Root directory of the ImageNet dataset")
p.add_argument("--workers",          type=int, default=4,
                help="DataLoader worker processes (default: 4)")
p.add_argument("--seed",             type=int, default=42)
p.add_argument("--batch_size",       type=int, default=256)
p.add_argument("--rates",            type=float, nargs="+", # when two rates are to be the same
                default=[0.1],
                help="Compression rates to evaluate (default: 0.1 0.3 0.5)")
# p.add_argument("-r1", "--rate1", type=float, nargs="+", 
#                default=[0.1], help='Pruning rate for correctly classified samples')
# p.add_argument("-r2", "--rate2", type=float, nargs="+", 
#                default=[0.1], help='Pruning rate for incorrectly classified samples')
p.add_argument("-q", "--n_bins", type=int, default=10, help='num bins for pruning')
p.add_argument("--prune_out",        type=str, default="reduced_result",
                help="Root directory for pruned index files")
p.add_argument("--epochs_clf",       type=int, default=100,
                help="Training epochs for each backbone classifier")
p.add_argument("--lr",               type=float, default=0.1)
p.add_argument("--momentum",         type=float, default=0.9)
p.add_argument("--weight_decay",     type=float, default=1e-4)
p.add_argument("--clf_out",          type=str, default="clf_out",
                help="Root directory for classifier checkpoints")
p.add_argument("--print_freq",       type=int, default=100,
                help="Print frequency during training/validation")
# p.add_argument("-pm", "--prune_mode", type=str, default="normal", choices=["normal", "cosine_all"], help='Mode for pruning: normal or cosine_all')


# Backbones to benchmark.  Comment out any you don't need.
BACKBONES = [
   "shufflenetv1",
    "shufflenetv2",
    "mobilenetv1",
    "mobilenetv2",
    "mobilenetv3",
    "GoogLeNet",
]


def adjust_lr_clf(optimizer, epoch, base_lr):
    """Step decay: ×0.1 every 30 epochs."""
    lr = base_lr * (0.1 ** (epoch // 30))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def train_one_epoch_clf(loader, model, criterion, optimizer, epoch,
                        print_freq):
    batch_t = AverageMeter(); data_t = AverageMeter()
    losses  = AverageMeter(); top1   = AverageMeter(); top5 = AverageMeter()
    model.train()
    end = time.time()
    for i, (inp, target) in enumerate(loader):
        data_t.update(time.time() - end)
        if DEVICE.type == "cuda":
            inp = inp.to(DEVICE, non_blocking=True)
        target = target.to(DEVICE, non_blocking=True)
        out  = model(inp)
        loss = criterion(out, target)
        p1, p5 = accuracy(out, target, topk=(1, 5))
        losses.update(loss.item(), inp.size(0))
        top1.update(p1[0], inp.size(0)); top5.update(p5[0], inp.size(0))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        batch_t.update(time.time() - end); end = time.time()
        if i % print_freq == 0:
            print(f"Epoch [{epoch}][{i}/{len(loader)}]  "
                  f"Loss {losses.val:.4f} ({losses.avg:.4f})  "
                  f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})  "
                  f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})")
    return losses.avg, top1.avg, top5.avg


def validate_clf(loader, model, criterion, print_freq):
    batch_t = AverageMeter()
    losses  = AverageMeter(); top1 = AverageMeter(); top5 = AverageMeter()
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (inp, target) in enumerate(loader):
            if DEVICE.type == "cuda":
                inp = inp.to(DEVICE, non_blocking=True)
                target = target.to(DEVICE, non_blocking=True)
            out  = model(inp)
            loss = criterion(out, target)
            p1, p5 = accuracy(out, target, topk=(1, 5))
            losses.update(loss.item(), inp.size(0))
            top1.update(p1[0], inp.size(0)); top5.update(p5[0], inp.size(0))
            batch_t.update(time.time() - end); end = time.time()
            if i % print_freq == 0:
                print(f"Test [{i}/{len(loader)}]  "
                      f"Loss {losses.val:.4f} ({losses.avg:.4f})  "
                      f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})  "
                      f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})")
    print(f" * Prec@1 {top1.avg:.3f}  Prec@5 {top5.avg:.3f}")
    return top1.avg, top5.avg



def run_classifiers(args, pruned_map):
    """
    For every (backbone, rate) combination, train a fresh classifier on the
    pruned subset (train split) and evaluate on the full val split.
    Checkpoints land in args.clf_out/<backbone>_SoftMax_r-{rate}_bins-{n_bins}/.
    """
    print("\n" + "="*70)
    print("STEP 3 — Classifier training")


    n_classes = 1000

    for backbone in BACKBONES:
        # for bins in args.n_bins:
        for rate in args.rates:
            tag              = f"r-{rate}-q-{args.n_bins}-imgnet-{args.prune_mode}"
            # tag              = f"r1-{args.rate1[0]}_r2-{rate}-imgnet"
            # tag              = f"bins{bins}-r1-{args.rate1[0]}_r2-{args.rate2[0]}-imgnet"
            model_name       = f"{backbone}_SoftMax_{tag}"
            ckpt_dir         = os.path.join(args.clf_out, model_name)
            best_ckpt        = os.path.join(ckpt_dir, "model_best.pth.tar")
            log_file         = os.path.join(ckpt_dir, model_name + ".txt")

            print(f"\n[Step 3] ── {model_name} ──")

            if os.path.exists(best_ckpt):
                print(f"[Step 3] model_best.pth.tar already exists — skipping.")
                continue

            os.makedirs(ckpt_dir, exist_ok=True)

            # ── Build pruned train dataset from the saved path list ────────────
            paths_file = pruned_map[rate]
            with open(paths_file) as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith("Total")]

            # selected_paths.txt now contains split-prefixed relative paths,
            # e.g. "train/abbey/00000001.jpg" and "val/abbey/00000001.jpg".
            # we prune both train and val splits 
            selected_train_abs = set()
            selected_val_abs = set()
            traindir = os.path.join(args.data, 'train')
            valdir = os.path.join(args.data, 'val')

            for line in lines:
                if "train" in line:         
                    selected_train_abs.add(line)
                elif "val" in line:
                    selected_val_abs.add(line)
            print(f"[Step 3] Selected {len(selected_train_abs)} train and "
                  f"{len(selected_val_abs)} val samples for rate={rate}.")
            print(f"example of selected train paths: {list(selected_train_abs)[:5]}")
            print(f"example of selected val paths: {list(selected_val_abs)[:5]}")


            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            _, train_set = get_dataloader(traindir, args.batch_size, shuffle=False,
                                        num_workers=args.workers,
                                        return_dataset=True, transform=train_transform)
            _, val_set   = get_dataloader(valdir,   args.batch_size, shuffle=False,
                                        num_workers=args.workers,
                                        return_dataset=True, transform=train_transform)
            print("Example of original train paths:", [s[0] for s in train_set.imgs[:5]])

            # Prune the training set
            train_set.imgs    = [s for s in train_set.imgs    if s[0] in selected_train_abs]
            train_set.samples = train_set.imgs

            print("Example of original train paths:", [s[0] for s in train_set.imgs[:5]])

            print(f"[Step 3] Pruned train size: {len(train_set)}")
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)


            print("Example of original val paths:", [s[0] for s in val_set.imgs[:5]])

            val_set.imgs    = [s for s in val_set.imgs    if s[0] in selected_val_abs]
            val_set.samples = val_set.imgs
            print(f"[Step 3] Pruned val size: {len(val_set)}")
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)

            

            # ── Model ─────────────────────────────────────────────────────────
            model = gM.get_model(model_name, num_class=n_classes)
            if DEVICE.type == "cuda":
                model = model.to(DEVICE)
            else:
                model = torch.nn.DataParallel(model).cuda()

            print(f"[Step 3] Parameters: "
                  f"{sum(p.numel() for p in model.parameters()):,}")
            
            criterion = nn.CrossEntropyLoss().to(DEVICE)
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)

            # ── Resume classifier checkpoint if interrupted ────────────────────
            start_epoch = 0
            ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth.tar")
            if os.path.exists(ckpt_path):
                print(f"[Step 3] Resuming {model_name} from {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location=DEVICE)
                start_epoch  = ckpt["epoch"]          # already epoch+1 when saved
                best_prec1   = ckpt["best_prec1"]
                model.load_state_dict(ckpt["state_dict"])
                optimizer.load_state_dict(ckpt["optimizer"])
                print(f"[Step 3] Resumed at epoch {start_epoch}, best_prec1={best_prec1:.2f}%")

            cudnn.benchmark = True
            Loss_plot = {}; trp1 = {}; trp5 = {}
            vp1_plot  = {}; vp5_plot = {}
            best_prec1 = 0.0; epoch_max = None

            for epoch in range(start_epoch, args.epochs_clf):
                t0 = time.time()
                adjust_lr_clf(optimizer, epoch, args.lr)

                loss_v, p1_v, p5_v = train_one_epoch_clf(
                    train_loader, model, criterion, optimizer, epoch, args.print_freq)
                Loss_plot[epoch] = loss_v
                trp1[epoch] = p1_v; trp5[epoch] = p5_v

                val_p1, val_p5 = validate_clf(
                    val_loader, model, criterion, args.print_freq)
                vp1_plot[epoch] = val_p1; vp5_plot[epoch] = val_p5

                is_best = val_p1 > best_prec1
                if is_best:
                    epoch_max = epoch
                best_prec1 = max(val_p1, best_prec1)

                save_checkpoint({
                    "epoch": epoch + 1, "arch": model_name,
                    "state_dict": model.state_dict(),
                    "best_prec1": best_prec1,
                    "optimizer": optimizer.state_dict(),
                }, is_best, directory=ckpt_dir)

                data_save(os.path.join(ckpt_dir, "Loss_plot.txt"),   Loss_plot)
                data_save(os.path.join(ckpt_dir, "train_prec1.txt"), trp1)
                data_save(os.path.join(ckpt_dir, "train_prec5.txt"), trp5)
                data_save(os.path.join(ckpt_dir, "val_prec1.txt"),   vp1_plot)
                data_save(os.path.join(ckpt_dir, "val_prec5.txt"),   vp5_plot)

                line = (f"Epoch {epoch}/{args.epochs_clf} "
                        f"loss={loss_v:.5f} train_p1={p1_v:.2f}% "
                        f"val_p1={val_p1:.2f}% best={best_prec1:.2f}%@{epoch_max}")
                wA.writeLogAcc(log_file, line)
                print(f"[Step 3] Wall time: {(time.time()-t0)/3600:.3f}h")
                print("-" * 70)

def prune_maps(args):
    pruned_map = {}
    # for bins in args.n_bins:
    for rate in args.rates: # for now, rate1 varies, rate2 fixed
        # output_dir = os.path.join(args.prune_out, f"r1-{rate}-r2-{args.rate2[0]}-ImageNet")
        # output_dir = os.path.join(args.prune_out, f"sm-r-{rate}-q-{args.n_bins}-ImageNet") # for clustering-based pruning
        output_dir = os.path.join(args.prune_out, f"bins{args.n_bins}-r1-{rate}-r2-{rate}-ImageNet-{args.prune_mode}") 
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "selected_paths.txt")
        pruned_map[rate] = output_file
    return pruned_map

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    global args
    args = p.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # ── Step 3 ────────────────────────────────────────────────────────────────
    pruned_map = prune_maps(args)
    run_classifiers(args, pruned_map)

    print("\n✓ Pipeline complete.")

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, directory, filename="checkpoint.pth.tar"):
    os.makedirs(directory, exist_ok=True)
    fpath = os.path.join(directory, filename)
    torch.save(state, fpath)
    if is_best:
        shutil.copyfile(fpath, os.path.join(directory, "model_best.pth.tar"))


def data_save(root, file_dict):
    """Append new epoch entries to a plain-text log file."""
    if not os.path.exists(root):
        open(root, "w").close()
    with open(root, "r") as f:
        lines = f.readlines()
    last_epoch = int(lines[-1].split()[0]) if lines else -1
    with open(root, "a") as f:
        for ep, val in file_dict.items():
            if ep > last_epoch:
                f.write(f"{ep} {val}\n")


if __name__ == "__main__":
    main()
