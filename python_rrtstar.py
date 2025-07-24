#!/usr/bin/env python3
import heapq, random
from collections import namedtuple
from itertools import count

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap, BoundaryNorm
import json

# ───────────────────────────────────────────────────────── parameters ──
GRID_W, GRID_H   = 80, 80
WALL_PROB        = 0.3
VIS_DELAY_MS     = 0          # ms between frames
DIAGONALS        = False      # only 4‑neighbour moves
STEP_PIXELS      = 5          # tree extension step (grid units)
REWIRE_RADIUS    = 8          # radius for RRT★ rewiring (grid units)
MAX_ITERS        = 10_000
GOAL_SAMPLE_RATE = 0.05       # probability of sampling the goal

# ────────────────────────────────────────────────────── helpers ───────
def dist(p, q):
    return max(abs(p[0]-q[0]), abs(p[1]-q[1])) if DIAGONALS \
           else abs(p[0]-q[0]) + abs(p[1]-q[1])

steps4  = [(-1,0),(1,0),(0,-1),(0,1)]
steps8  = steps4 + [(-1,-1),(-1,1),(1,-1),(1,1)]

def steer(p, q, step=STEP_PIXELS):
    """Move from p towards q by ≤step (grid coords)."""
    if dist(p, q) <= step:
        return q
    dx, dy = q[0]-p[0], q[1]-p[1]
    if dx == dy == 0:
        return p
    length = max(abs(dx), abs(dy)) if DIAGONALS else abs(dx)+abs(dy)
    ux, uy = dx/length, dy/length
    return (int(round(p[0] + ux*step)), int(round(p[1] + uy*step)))

def collision_free(p, q, grid):
    """Bresenham‑like check along segment p→q."""
    x0, y0 = p; x1, y1 = q
    dx, dy = abs(x1-x0), abs(y1-y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx-dy
    while True:
        if grid[y0, x0] == 1:
            return False
        if (x0, y0) == (x1, y1):
            break
        e2 = 2*err
        if e2 > -dy:
            err -= dy; x0 += sx
        if e2 <  dx:
            err += dx; y0 += sy
    return True

# ────────────────────────────────────────────── maze load ─────────────
with open("maze.json", "r") as f:
    maze = np.array(json.load(f))

maze[0,:] = maze[-1,:] = maze[:,0] = maze[:,-1] = 1
maze[1,1] = maze[-2,-2] = 0
START = (1,1)
GOAL  = (GRID_W-2, GRID_H-2)

# ─────────────────────── RRT★ node & storage ─────────────────────────
RRTNode = namedtuple("RRTNode", "x y parent cost")  # cost = g‑score

nodes       = [RRTNode(*START, parent=None, cost=0)]
node_lookup = {START: 0}   # (x,y) → index in nodes list

# ─────────────────────────────── UI helpers ───────────────────────────
frames = []
def record_frame(new_node=None, rewired_edges=None, final_path=None):
    # 0 free, 1 wall
    grid = maze.copy()
    # explored tree nodes
    for n in nodes:
        grid[n.y, n.x] = 2
    # optionally highlight the most recently added node
    if new_node:
        grid[new_node.y, new_node.x] = 3
    # final path
    if final_path:
        for a, b in zip(final_path[:-1], final_path[1:]):
            draw_segment(a, b, grid, value=4)
    grid[START[1],START[0]] = 5
    grid[GOAL[1], GOAL[0]]  = 6
    frames.append(grid)

def draw_segment(p, q, grid, value=4):
    """Colour every cell along segment p→q (inclusive)."""
    x0, y0 = p; x1, y1 = q
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        grid[y0, x0] = value
        if (x0, y0) == (x1, y1):
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x0 += sx
        if e2 <  dx:
            err += dx; y0 += sy


record_frame()  # first frame

# ───────────────────────────── RRT★ loop ──────────────────────────────
rng   = random.Random(42)
iters = 0
reached_goal = None

while iters < MAX_ITERS:
    iters += 1

    # --- sample goal occasionally to bias tree ---
    if rng.random() < GOAL_SAMPLE_RATE:
        sample = GOAL
    else:
        sx, sy = rng.randrange(1, GRID_W-1), rng.randrange(1, GRID_H-1)
        if maze[sy, sx] == 1:
            continue
        sample = (sx, sy)

    # --- nearest existing node ---
    nearest_idx = min(range(len(nodes)), key=lambda i: dist((nodes[i].x,nodes[i].y), sample))
    nearest = nodes[nearest_idx]

    # --- steer towards sample ---
    new_pt = steer((nearest.x, nearest.y), sample, STEP_PIXELS)
    if maze[new_pt[1], new_pt[0]] == 1:
        continue
    if not collision_free((nearest.x, nearest.y), new_pt, maze):
        continue

    # --- choose parent & cost ---
    # find neighbours for rewiring
    neighbour_ids = [i for i,n in enumerate(nodes)
                     if dist((n.x,n.y), new_pt) <= REWIRE_RADIUS
                     and collision_free((n.x,n.y), new_pt, maze)]
    # pick min-cost parent
    best_parent_id = nearest_idx
    best_cost      = nearest.cost + dist((nearest.x,nearest.y), new_pt)
    for i in neighbour_ids:
        cand_cost = nodes[i].cost + dist((nodes[i].x,nodes[i].y), new_pt)
        if cand_cost < best_cost:
            best_cost, best_parent_id = cand_cost, i

    new_node = RRTNode(*new_pt, parent=best_parent_id, cost=best_cost)
    nodes.append(new_node)
    node_lookup[new_pt] = len(nodes)-1

    # --- rewire neighbours if we give them a cheaper path ---
    for i in neighbour_ids:
        n = nodes[i]
        new_cost = new_node.cost + dist((new_node.x,new_node.y), (n.x,n.y))
        if new_cost < n.cost:
            nodes[i] = RRTNode(n.x, n.y, parent=node_lookup[new_pt], cost=new_cost)

    # --- record frame ---
    record_frame(new_node)

    # --- check goal proximity & build path ---
    if dist(new_pt, GOAL) <= STEP_PIXELS and collision_free(new_pt, GOAL, maze):
        reached_goal = len(nodes)-1
        break

# ─────────────────────────── reconstruct path ─────────────────────────
path = []
if reached_goal is not None:
    cur = RRTNode(*GOAL, parent=reached_goal, cost=0)  # virtual goal node
    while cur.parent is not None:
        path.append((cur.x,cur.y))
        cur = nodes[cur.parent]
    path.append(START)
    path = path[::-1]
    record_frame(final_path=path)  # final frame

# ────────────────────────────── animation ─────────────────────────────
cmap = ListedColormap([
    "white",      # 0 free
    "black",      # 1 wall
    "#4C73E8",    # 2 tree nodes
    "#6CE3F5",    # 3 latest node
    "#6BDF6F",    # 4 final path
    "red",        # 5 start
    "red"         # 6 goal
])
norm = BoundaryNorm(np.arange(-0.5,7), cmap.N)

fig, ax = plt.subplots(figsize=(8,5))
im = ax.imshow(frames[0], cmap=cmap, norm=norm, interpolation="none")
ax.set_xticks([]); ax.set_yticks([])

def update(i):
    im.set_data(frames[i])
    ax.set_title(f"step {i+1}/{len(frames)}")
    return (im,)

ani = animation.FuncAnimation(fig, update,
                              frames=len(frames),
                              interval=VIS_DELAY_MS,
                              blit=True, repeat=False)
plt.show()
