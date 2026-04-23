import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# =========================
# RDCM CORE
# =========================
def run_rdcm_timeseries(T=200, alpha=0.08, beta=0.9, k=3, noise_scale=1.2):

    E = np.zeros(T)
    for t in range(1, T):
        if t < T//2:
            E[t] = E[t-1] + np.random.randn() * noise_scale
        else:
            E[t] = E[t-1] + 0.3 + np.random.randn() * noise_scale * 0.5

    pred = np.zeros(T)
    for t in range(1, T):
        pred[t] = beta * pred[t-1] + (1 - beta) * E[t-1]

    Delta = np.abs(E - pred)

    centers = np.linspace(np.min(Delta), np.max(Delta), k)
    labels = np.zeros(T)

    for t in range(T):
        d = Delta[t]
        idx = np.argmin(np.abs(centers - d))
        labels[t] = idx
        centers[idx] += 0.05 * (d - centers[idx])

    D = np.zeros(T)
    weights = np.ones(k)

    for t in range(1, T):
        c = int(labels[t])
        weights[c] += 1
        w = weights[c] / np.sum(weights)
        D[t] = (1 - alpha) * D[t-1] + alpha * (w * centers[c])

    return D


# =========================
# Convergence metric
# =========================
def compute_convergence(D):
    return np.mean(np.abs(np.diff(D[-30:])))


# =========================
# Phase計算
# =========================
def compute_phase(alpha_vals, noise_vals):

    heatmap = np.zeros((len(alpha_vals), len(noise_vals)))

    for i, alpha in enumerate(alpha_vals):
        for j, noise in enumerate(noise_vals):

            vals = []
            for _ in range(3):
                D = run_rdcm_timeseries(alpha=alpha, noise_scale=noise)
                vals.append(compute_convergence(D))

            heatmap[i, j] = np.mean(vals)

    return heatmap


# =========================
# 軽スムージング
# =========================
def smooth(data, k=3):
    pad = k // 2
    padded = np.pad(data, pad, mode='edge')
    out = np.zeros_like(data)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            out[i, j] = np.mean(padded[i:i+k, j:j+k])

    return out


# =========================
# Phaseプロット（最終版）
# =========================
def plot_phase_final():

    alpha_vals = np.linspace(0.02, 0.5, 40)
    noise_vals = np.linspace(0.5, 3.0, 40)

    heatmap = compute_phase(alpha_vals, noise_vals)
    heatmap = smooth(heatmap, 3)

    X, Y = np.meshgrid(noise_vals, alpha_vals)

    plt.figure(figsize=(10, 6))

    # ヒートマップ
    im = plt.imshow(
        heatmap,
        origin='lower',
        aspect='auto',
        extent=[noise_vals[0], noise_vals[-1], alpha_vals[0], alpha_vals[-1]],
        cmap='viridis'
    )

    plt.colorbar(label="Convergence Metric (lower = stable)")

    # 等高線（構造）
    levels = np.linspace(np.min(heatmap), np.max(heatmap), 6)
    plt.contour(
        X, Y, heatmap,
        levels=levels,
        colors='white',
        linewidths=1.2,
        alpha=0.7
    )

    # 境界（弱めた）
    boundary = np.percentile(heatmap, 35)

    plt.contour(
        X, Y, heatmap,
        levels=[boundary],
        colors='red',
        linewidths=1.6,
        alpha=0.8
    )

    # ラベル
    plt.xlabel("Noise Scale")
    plt.ylabel("Alpha (Update Rate)")

    # ✔ 名前統一
    plt.title("Conditional Convergence in RDCM")

    plt.text(0.9, 0.07, "Stable Region", color='white', fontsize=12, weight='bold')
    plt.text(2.0, 0.42, "Unstable Region", color='white', fontsize=12, weight='bold')

    plt.tight_layout()
    plt.show()


# =========================
# RUN
# =========================
if __name__ == "__main__":
    plot_phase_final()