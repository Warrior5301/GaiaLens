import argparse, os, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from features import extract_path
from model import GSDNet


def group_id(fname):
    m = re.match(r"ucm_s\d+_(\d+)", fname)
    if m:
        return "ucm_" + m.group(1)
    m = re.match(r"eurosat_(\d+)", fname)
    if m:
        return "eurosat_" + m.group(1)
    return fname


def build_features(df, data_dir, cache):
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if list(z["files"]) == list(df.filename):
            print("loaded cached features", cache)
            return z["X"]
    X = np.stack([extract_path(os.path.join(data_dir, f)) for f in df.filename])
    if cache:
        np.savez(cache, X=X, files=np.array(df.filename))
    return X


def main(data_dir, csv, out_dir=".", epochs=50, lr=1e-3, bs=64, seed=0,
         shuffle_labels=False, holdout_gsd=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv, sep=None, engine="python")  # handles comma or tab
    df.columns = [c.strip() for c in df.columns]
    df["group"] = df.filename.map(group_id)
    print(df.groupby("gsd_meters").size().to_string(), "\n")

    X = build_features(df, data_dir, os.path.join(out_dir, "features_cache.npz"))
    y = np.log(df.gsd_meters.values.astype(np.float32))

    # --- split by group (all scales of one image stay together) ---
    groups = np.array(sorted(df.group.unique()), dtype=object)
    rng.shuffle(groups)
    val_groups = set(groups[: int(0.2 * len(groups))])
    is_val = df.group.isin(val_groups).values
    is_train = ~is_val
    if holdout_gsd is not None:
        held = np.isclose(df.gsd_meters.values, holdout_gsd)
        is_train &= ~held
        is_val = held  # validate ONLY on the unseen GSD
        print(f"holding out GSD {holdout_gsd}: train={is_train.sum()} val={is_val.sum()}")
    else:
        print(f"train={is_train.sum()} val={is_val.sum()}")

    if shuffle_labels:
        y = y.copy(); tr = np.where(is_train)[0]; y[tr] = y[rng.permutation(tr)]
        print("!! labels shuffled in train (sanity check mode)")

    mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt, yt = torch.tensor(Xn[is_train]).to(dev), torch.tensor(y[is_train]).to(dev)
    Xv, yv = torch.tensor(Xn[is_val]).to(dev), torch.tensor(y[is_val]).to(dev)

    model = GSDNet(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=0.5)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
        if ep % 5 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(Xv), yv).item()
            print(f"epoch {ep:3d}  train_loss {loss.item():.4f}  val_loss {vl:.4f}")

    # --- evaluation ---
    model.eval()
    with torch.no_grad():
        pred = torch.exp(model(Xv)).cpu().numpy()
    true = np.exp(yv.cpu().numpy())
    rel = np.abs(pred - true) / true
    print(f"\nOVERALL  MAE {np.abs(pred - true).mean():.3f} m   median rel err {np.median(rel) * 100:.1f}%")
    print("\nper-GSD:  true | n | mean pred | MAE | median rel err")
    for g in np.unique(np.round(true, 4)):
        m = np.isclose(true, g, atol=1e-3)
        print(f"  {g:5.2f} | {m.sum():4d} | {pred[m].mean():6.2f} | {np.abs(pred[m] - g).mean():5.2f} | {np.median(rel[m]) * 100:5.1f}%")

    # predicted-vs-true table: nearest known GSD class
    classes = np.unique(np.round(np.exp(y), 4))
    nearest = lambda a: classes[np.abs(np.log(a)[:, None] - np.log(classes)[None]).argmin(1)]
    pc, tc = nearest(pred), nearest(true)
    print("\nconfusion (rows=true, cols=pred nearest class):", classes)
    for c in np.unique(tc):
        print(f"  {c:5.2f} ->", [int(((tc == c) & (pc == k)).sum()) for k in classes])

    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd, "n_in": X.shape[1]},
               os.path.join(out_dir, "gsd_model.pt"))
    print("\nsaved", os.path.join(out_dir, "gsd_model.pt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--shuffle_labels", action="store_true")
    ap.add_argument("--holdout_gsd", type=float, default=None)
    a = ap.parse_args()
    main(a.data_dir, a.csv, a.out_dir, a.epochs, shuffle_labels=a.shuffle_labels,
         holdout_gsd=a.holdout_gsd)
