import shap, numpy as np, pandas as pd, matplotlib.pyplot as plt
from .features import build_features
from .data import load_prices
from .utils import window_stack, CLASS_NAMES

def shap_for_tf(model, X, max_samples=256):
    # Use KernelExplainer as a framework-agnostic fallback
    bg = shap.kmeans(X.reshape(X.shape[0], -1), 25)
    explainer = shap.KernelExplainer(lambda z: model.predict(z.reshape(-1, X.shape[1], X.shape[2]), verbose=0),
                                     bg)
    m = min(max_samples, X.shape[0])
    shap_vals = explainer.shap_values(X[:m].reshape(m, -1), nsamples=100)
    return shap_vals

def shap_for_torch(net, X, max_samples=256):
    import torch
    net.eval()
    def f(z):
        with torch.no_grad():
            logits = net(torch.from_numpy(z.reshape(-1, X.shape[1], X.shape[2])))
            return torch.softmax(logits, dim=1).numpy()
    bg = shap.kmeans(X.reshape(X.shape[0], -1), 25)
    explainer = shap.KernelExplainer(f, bg)
    m = min(max_samples, X.shape[0])
    shap_vals = explainer.shap_values(X[:m].reshape(m, -1), nsamples=100)
    return shap_vals

def save_summary_plot(shap_vals, X, path="shap_summary.png"):
    plt.figure()
    shap.summary_plot(shap_vals, X.reshape(X.shape[0], -1), show=False)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path
