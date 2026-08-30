"""Training loop for CNN baseline (or any image-level classifier with a train/val DataLoader)."""
import time
import torch
from sklearn.metrics import roc_auc_score
from pathology.utils.checkpoint import save_latest, save_best, load_latest_if_exists


def run_epoch(model, loader, criterion, optimizer, device, train_mode=True):
    model.train() if train_mode else model.eval()
    total_loss = 0
    all_labels, all_probs = [], []
    with torch.set_grad_enabled(train_mode):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train_mode:
                optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            probs = torch.softmax(out, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    auroc = roc_auc_score(all_labels, all_probs)
    return avg_loss, auroc


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                 epochs, device, ckpt_dir, verbose=True):
    import os
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_path = f"{ckpt_dir}/latest.pt"
    best_path = f"{ckpt_dir}/best.pt"

    start_epoch, best_val_auroc = load_latest_if_exists(latest_path, model, optimizer, scheduler, device)

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss, train_auroc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_auroc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step(val_loss)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} train_auroc={train_auroc:.4f} "
                  f"| val_loss={val_loss:.4f} val_auroc={val_auroc:.4f} | {time.time()-t0:.1f}s")

        save_latest(latest_path, epoch, model, optimizer, scheduler, val_auroc, best_val_auroc)
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            save_best(best_path, epoch, model, val_auroc)
            if verbose:
                print(f"  New best (val_auroc={val_auroc:.4f})")

    return best_val_auroc
