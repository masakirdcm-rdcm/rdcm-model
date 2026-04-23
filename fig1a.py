import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# -------------------------
# Parameters
# -------------------------
N = 10
decay = 0.65
angle = np.deg2rad(30)

# -------------------------
# Initialization
# -------------------------
points = [(0.0, 0.0)]
vectors = []

# -------------------------
# Recursive generation
# -------------------------
for i in range(1, N):
    prev = np.array(points[-1])

    base_vec = np.array([np.cos(angle), np.sin(angle)])
    vec = base_vec * (decay ** (i-1))

    new_point = prev + vec

    points.append(tuple(new_point))
    vectors.append(vec)

points = np.array(points)

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Scale structure (circles)
for i, p in enumerate(points):
    if i == 0:
        r = 1.6
    else:
        r = np.linalg.norm(vectors[i-1])

    circle = plt.Circle(
        p, r,
        fill=False,
        color='black',
        alpha=0.35,
        linewidth=1.2
    )
    ax.add_patch(circle)

# Recursive updates (vectors)
for i in range(1, len(points)):
    ax.arrow(
        points[i-1, 0], points[i-1, 1],
        points[i, 0] - points[i-1, 0],
        points[i, 1] - points[i-1, 1],
        head_width=0.03,
        length_includes_head=True,
        color='tab:blue',
        linewidth=1.8
    )

# States (points)
ax.scatter(
    points[:, 0],
    points[:, 1],
    color='tab:red',
    zorder=3,
    label="State"
)

# Step indices
for i, p in enumerate(points):
    ax.text(p[0] + 0.02, p[1] + 0.02, f"{i}", fontsize=9)

# Legend
ax.plot([], [], color='tab:blue', label="Recursive update")
ax.plot([], [], color='black', label="Scale structure")
ax.legend(frameon=False, loc='upper left')

# Style
ax.set_title("Recursive Integration Structure in RDCM")
ax.set_aspect('equal')
ax.grid(True)

plt.tight_layout()
plt.show()