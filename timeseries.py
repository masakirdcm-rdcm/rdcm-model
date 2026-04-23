import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# =========================
# RDCM CORE（Time Series用）
# =========================
def run_rdcm_timeseries(T=200, alpha=0.08, beta=0.9, k=3, noise_scale=1.2):

    # Experience E(t)
    E = np.zeros(T)
    for t in range(1, T):
        if t < T//2:
            E[t] = E[t-1] + np.random.randn() * noise_scale
        else:
            E[t] = E[t-1] + 0.3 + np.random.randn() * noise_scale * 0.5

    # Prediction
    pred = np.zeros(T)
    for t in range(1, T):
        pred[t] = beta * pred[t-1] + (1 - beta) * E[t-1]

    # Differentiation
    Delta = np.abs(E - pred)

    # Online clustering
    centers = np.linspace(np.min(Delta), np.max(Delta), k)
    labels = np.zeros(T)

    for t in range(T):
        d = Delta[t]
        idx = np.argmin(np.abs(centers - d))
        labels[t] = idx
        centers[idx] += 0.05 * (d - centers[idx])

    # Recursive integration
    D = np.zeros(T)
    weights = np.ones(k)

    for t in range(1, T):
        c = int(labels[t])
        weights[c] += 1
        w = weights[c] / np.sum(weights)
        D[t] = (1 - alpha) * D[t-1] + alpha * (w * centers[c])

    return E, D


# =========================
# PLOT TIME SERIES
# =========================
def plot_timeseries():

    E, D = run_rdcm_timeseries()

    # 線形スケーリング（最小限）
    D_scaled = (D - np.min(D)) / (np.max(D) - np.min(D))
    D_scaled = D_scaled * (np.max(E) - np.min(E)) + np.min(E)

    # ローカル分散
    window = 10
    rolling_mean = np.zeros(len(D))
    rolling_std = np.zeros(len(D))

    for i in range(len(D)):
        seg = D_scaled[max(0, i-window):i+1]
        rolling_mean[i] = np.mean(seg)
        rolling_std[i] = np.std(seg)

    plt.figure(figsize=(12, 7))

    plt.plot(E, color='gray', alpha=0.5, linewidth=1.2, label="Experience")
    plt.plot(D_scaled, color='black', linewidth=2.5, label="Dominant Direction")

    plt.fill_between(
        range(len(D)),
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        color='black',
        alpha=0.15
    )

    plt.title("Emergence of Stable Direction in RDCM")
    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================
# RUN
# =========================
if __name__ == "__main__":
    plot_timeseries()