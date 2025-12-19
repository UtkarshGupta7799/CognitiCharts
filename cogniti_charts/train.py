import argparse, os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from .data import load_prices, label_patterns, train_test_split_time
from .features import build_features
from .utils import window_stack, load_config, CLASS_NAMES
from . import models_tf, models_torch


# ------------------------------
# Utilities
# ------------------------------
def build_xy(df: pd.DataFrame, lookback: int):
    feats = build_features(df)
    X = window_stack(feats.values, lookback)          # [N, L, F]
    y = df["label"].values[lookback-1:]               # align to end of window
    return X.astype("float32"), y.astype("int64")


# Optional: handy oversampler (kept for later toggling; disabled by default)
def oversample_minority(X: np.ndarray, y: np.ndarray):
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    idxs = []
    for c, cnt in zip(classes, counts):
        c_idx = np.where(y == c)[0]
        if cnt < max_count:
            reps = np.random.choice(c_idx, size=max_count - cnt, replace=True)
            idxs.append(np.concatenate([c_idx, reps]))
        else:
            idxs.append(c_idx)
    sel = np.concatenate(idxs)
    np.random.shuffle(sel)
    return X[sel], y[sel]


# Optional: Focal Loss (works well with imbalance)
def sparse_categorical_focal_loss(gamma=2.0, alpha=None):
    import tensorflow as tf
    # capture alpha once at factory time to avoid inner-scope reassignment
    alpha_vec = None
    if alpha is not None:
        # alpha can be: scalar (broadcast) or list/np.array of per-class weights [C]
        alpha_vec = tf.convert_to_tensor(alpha, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        # y_true: (B,), y_pred: (B, C) with softmax probabilities expected
        y_true = tf.cast(y_true, tf.int32)
        num_classes = tf.shape(y_pred)[1]
        y_true_oh = tf.one_hot(y_true, depth=num_classes, dtype=tf.float32)

        # numerical stability
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)

        # standard CE
        ce = -tf.reduce_sum(y_true_oh * tf.math.log(y_pred), axis=1)

        # p_t = prob of the true class
        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=1)

        # modulating factor (1 - p_t)^gamma
        mod = tf.pow(1.0 - p_t, gamma)

        if alpha_vec is not None:
            # support scalar alpha (broadcast) or vector alpha per class
            if alpha_vec.shape.rank == 0:
                alpha_t = alpha_vec
            else:
                # pick alpha for the true class
                alpha_t = tf.reduce_sum(y_true_oh * alpha_vec, axis=1)
            return tf.reduce_mean(alpha_t * mod * ce)

        return tf.reduce_mean(mod * ce)

    return loss_fn


# ------------------------------
# Trainers
# ------------------------------
def train_tf(Xtr, ytr, Xval, yval, class_weight, use_focal=False):
    import tensorflow as tf
    os.makedirs("models", exist_ok=True)

    model = models_tf.make_tf_model(
        seq_len=Xtr.shape[1],
        num_features=Xtr.shape[2],
        num_classes=len(CLASS_NAMES)
    )

    # compile (lower LR; focal optional)
    loss_fn = sparse_categorical_focal_loss(gamma=2.0, alpha=None) if use_focal else "sparse_categorical_crossentropy"
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        "models/best_tf.keras", monitor="val_loss", save_best_only=True
    )

    model.fit(
        Xtr, ytr,
        validation_data=(Xval, yval),
        epochs=30,
        batch_size=64,
        callbacks=[es, ckpt],
        verbose=2,
        class_weight=class_weight,  # key for imbalance
    )

    # final save
    model.save("models/tf_model.keras")
    return model


def train_torch(Xtr, ytr, Xval, yval):
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = models_torch.TorchCNN1D(num_features=Xtr.shape[2], num_classes=len(CLASS_NAMES)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    # class weights (balanced)
    classes, counts = np.unique(ytr, return_counts=True)
    weights = (counts.max() / counts).astype(np.float32)
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)

    # per-sample weights for sampling
    sample_w = np.zeros_like(ytr, dtype=np.float32)
    for c, w in zip(classes, weights):
        sample_w[ytr == c] = w
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    va_ds = TensorDataset(torch.from_numpy(Xval), torch.from_numpy(yval))
    tr_dl = DataLoader(tr_ds, batch_size=64, sampler=sampler)
    va_dl = DataLoader(va_ds, batch_size=128)

    loss_fn = nn.CrossEntropyLoss(weight=class_weight)

    best_acc, best_state = 0.0, None
    for epoch in range(25):
        net.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = net(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        # val
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = net(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += len(yb)
        acc = correct / total if total else 0.0
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in net.state_dict().items()}

    if best_state:
        net.load_state_dict(best_state)

    os.makedirs("models", exist_ok=True)
    joblib.dump(net.state_dict(), "models/torch_model.pt")
    return net.cpu()


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="sample_data/sample_prices.csv")
    ap.add_argument("--framework", choices=["tf", "torch"], default="tf")
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--use_focal", action="store_true", help="Use focal loss (TF only)")
    ap.add_argument("--use_oversample", action="store_true", help="Enable oversampling (optional)")
    args = ap.parse_args()

    cfg = load_config()
    df = load_prices(args.csv)
    df = label_patterns(df, lookback=args.lookback)
    tr, va, te = train_test_split_time(df, test_ratio=0.2, val_ratio=0.1)

    # windows → tensors
    Xtr, ytr = build_xy(tr, args.lookback)
    Xva, yva = build_xy(pd.concat([tr.iloc[-args.lookback+1:], va]), args.lookback)
    Xte, yte = build_xy(pd.concat([va.iloc[-args.lookback+1:], te]), args.lookback)

    # compute class weights from ORIGINAL ytr
    classes = np.unique(ytr)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=ytr)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

    # normalize + clip class weights (keeps gradients sane)
    mean_w = np.mean(list(class_weight.values()))
    for k in class_weight:
        class_weight[k] = class_weight[k] / mean_w
        class_weight[k] = float(np.clip(class_weight[k], 0.6, 2.0))
    print("Class weights (normalized+clipped):", class_weight)

    # OPTIONAL oversampling (default OFF)
    if args.use_oversample:
        Xtr, ytr = oversample_minority(Xtr, ytr)

    if args.framework == "tf":
        model = train_tf(Xtr, ytr, Xva, yva, class_weight=class_weight, use_focal=args.use_focal)
        # Evaluate on test
        import tensorflow as tf
        yhat = model.predict(Xte, verbose=0).argmax(1)
    else:
        net = train_torch(Xtr, ytr, Xva, yva)
        import torch
        from .models_torch import TorchCNN1D
        net_eval = TorchCNN1D(num_features=Xte.shape[2], num_classes=len(CLASS_NAMES))
        net_eval.load_state_dict(net.state_dict())
        net_eval.eval()
        with torch.no_grad():
            yhat = net_eval(torch.from_numpy(Xte)).argmax(1).numpy()

    print(classification_report(yte, yhat, target_names=CLASS_NAMES))


if __name__ == "__main__":
    main()
