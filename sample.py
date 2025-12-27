import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Parameters
# -----------------------------
np.random.seed(3)

STEP_SIZE = 0.35
MAX_STEPS = 500
SUN_ANGLE = 45  # global reference (visual)

NEST = np.array([0.0, 0.0])
FOOD = np.array([9.0, 6.0])
FOOD_RADIUS = 0.8

# -----------------------------
# Helper Functions
# -----------------------------
def random_direction():
    angle = np.random.uniform(0, 2 * np.pi)
    return np.array([np.cos(angle), np.sin(angle)])

# -----------------------------
# Simulation State
# -----------------------------
position = NEST.copy()
home_vector = np.zeros(2)

exploration_path = []
return_path = []

phase = "explore"

# -----------------------------
# Plot Setup
# -----------------------------
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_aspect("equal")
ax.grid(True)
ax.set_title("Bio-inspired strategies for multimodal navigation of autonomous vehicles in warehouse settings")

ax.plot(NEST[0], NEST[1], "go", markersize=10, label="Nest")
ax.plot(FOOD[0], FOOD[1], "o", color="orange", markersize=10, label="Food")

agent_dot, = ax.plot([], [], "ko", markersize=6, label="Ant")
explore_line, = ax.plot([], [], "b-", linewidth=2, label="Exploration")
return_line, = ax.plot([], [], "r-", linewidth=2, label="Homing")

# Sun compass arrow (visual cue)
sun_vec = np.array([
    np.cos(np.deg2rad(SUN_ANGLE)),
    np.sin(np.deg2rad(SUN_ANGLE))
])
ax.arrow(-12, -12, sun_vec[0]*3, sun_vec[1]*3,
         width=0.15, color="gold")

ax.legend()

# -----------------------------
# Animation Update
# -----------------------------
def update(frame):
    global position, phase, home_vector

    if phase == "explore":
        # 🔹 Biased exploration (key fix)
        to_food = FOOD - position
        to_food = to_food / np.linalg.norm(to_food)

        rand_dir = random_direction()
        direction = 0.7 * rand_dir + 0.3 * to_food
        direction = direction / np.linalg.norm(direction)

        move = STEP_SIZE * direction
        position[:] += move
        home_vector += move
        exploration_path.append(position.copy())

        if np.linalg.norm(position - FOOD) < FOOD_RADIUS:
            phase = "return"

    elif phase == "return":
        if np.linalg.norm(home_vector) > 0.05:
            direction = -home_vector / np.linalg.norm(home_vector)
            move = STEP_SIZE * direction
            position[:] += move
            home_vector += move
            return_path.append(position.copy())

    # Update visuals (Python 3.14 safe)
    agent_dot.set_data([position[0]], [position[1]])

    if exploration_path:
        p = np.array(exploration_path)
        explore_line.set_data(p[:, 0], p[:, 1])

    if return_path:
        r = np.array(return_path)
        return_line.set_data(r[:, 0], r[:, 1])

    return agent_dot, explore_line, return_line

# -----------------------------
# Run
# -----------------------------
ani = FuncAnimation(fig, update, frames=500, interval=60)
plt.show()
