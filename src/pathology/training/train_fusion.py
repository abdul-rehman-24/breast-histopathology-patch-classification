"""Training loop for fusion/foundation-only/attention models using cached features (fast, no image I/O)."""
import time
import torch
from sklearn.metrics import roc_auc_score
from pathology.utils.checkpoint import save_latest, save_best, load_latest_if_exists


def run_fused_epoch(model, loader, criterion, optimizer, device, train_mode=True):
    """For FusionModel: batches yield (cnn_feat, foundation_feat, label)."""
    model.train() if train_mode else model.eval()
    total_loss = 0
    all_labels, all_probs = [], []
    with torch.set_grad_enabled(train_mode):
        for cnn_f, found_f, labels in loader:
            cnn_f, found_f, labels = cnn_f.to(device), found_f.to(device), labels.to(device)
            if train_mode:
                optimizer.zero_grad()
            out = model(cnn_f, found_f)
            loss = criterion(out, labels)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * cnn_f.size(0)
            probs = torch.softmax(out, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, roc_auc_score(all_labels, all_probs)


def run_single_feature_epoch(model, loader, criterion, optimizer, device, train_mode=True):
    """For FoundationOnlyModel: batches yield (feature, label)."""
    model.train() if train_mode else model.eval()
    total_loss = 0
    all_labels, all_probs = [], []
    with torch.set_grad_enabled(train_mode):
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            if train_mode:
                optimizer.zero_grad()
            out = model(feats)
            loss = criterion(out, labels)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * feats.size(0)
            probs = torch.softmax(out, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, roc_auc_score(all_labels, all_probs)


def run_group_epoch(model, group_data, criterion, optimizer, device, train_mode=True):
    """For AttentionAggregator: group_data is {group_id: (instances_tensor, label_int)}."""
    model.train() if train_mode else model.eval()
    total_loss = 0
    all_labels, all_probs = [], []
    with torch.set_grad_enabled(train_mode):
        for gid, (instances, label) in group_data.items():
            instances = instances.to(device)
            label_t = torch.tensor([label]).to(device)
            if train_mode:
                optimizer.zero_grad()
            out, _ = model(instances)
            loss = criterion(out, label_t)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            prob = torch.softmax(out, dim=1)[0, 1].item()
            all_labels.append(label)
            all_probs.append(prob)
    avg_loss = total_loss / len(group_data)
    return avg_loss, roc_auc_score(all_labels, all_probs)


def train_generic(model, train_data, val_data, epoch_fn, criterion, optimizer, scheduler,
                   epochs, device, ckpt_dir, verbose=True):
    """Shared training driver — epoch_fn is one of the run_*_epoch functions above."""
    import os
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_path = f"{ckpt_dir}/latest.pt"
    best_path = f"{ckpt_dir}/best.pt"

    start_epoch, best_val_auroc = load_latest_if_exists(latest_path, model, optimizer, scheduler, device)

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss, train_auroc = epoch_fn(model, train_data, criterion, optimizer, device, True)
        val_loss, val_auroc = epoch_fn(model, val_data, criterion, optimizer, device, False)
        scheduler.step(val_loss)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} | train_auroc={train_auroc:.4f} | "
                  f"val_loss={val_loss:.4f} val_auroc={val_auroc:.4f} | {time.time()-t0:.1f}s")

        save_latest(latest_path, epoch, model, optimizer, scheduler, val_auroc, best_val_auroc)
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            save_best(best_path, epoch, model, val_auroc)
            if verbose:
                print(f"  New best (val_auroc={val_auroc:.4f})")

    return best_val_auroc
