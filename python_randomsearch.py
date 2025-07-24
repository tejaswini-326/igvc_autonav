#!/usr/bin/env python3
import heapq, random
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap, BoundaryNorm
from itertools import count
import json

# ───────────────────────────────────────────────────────── parameters ──
GRID_W, GRID_H   = 80, 80
WALL_PROB        = 0.3
VIS_DELAY_MS     = 0
DIAGONALS        = False

# ──────────────────────────────────────────────────────── utilities ─
# We keep f,g fields for compatibility but they aren't used in random search
Node = namedtuple("Node", "f g x y parent")

def neighbours(x, y):
    steps = [(-1,0),(1,0),(0,-1),(0,1)]
    if DIAGONALS:
        steps += [(-1,-1),(-1,1),(1,-1),(1,1)]
    for dx,dy in steps:
        nx, ny = x+dx, y+dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
            yield nx, ny

# ───────────────────────────────── build maze from JSON ──
with open("maze.json", "r") as f:
    maze = np.array(json.load(f))


# add black border
maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1

# carve start/end
start_coord = (1, 1)
goal_coord  = (GRID_W-2, GRID_H-2)
maze[start_coord[1], start_coord[0]] = 0
maze[goal_coord[1],  goal_coord[0]]  = 0

# ───────────────────────────────────────── capture animation frames ──
frames = []
def record_frame(open_set, closed_set, path=None):
    grid = maze.copy()
    for x,y in closed_set:      grid[y,x] = 2
    for n in open_set:          grid[n.y,n.x] = 3
    if path:
        for x,y in path:        grid[y,x] = 4
    grid[start_coord[1], start_coord[0]] = 5
    grid[goal_coord[1],  goal_coord[0]]  = 6
    frames.append(grid)

def reconstruct(node):
    pts = []
    while node:
        pts.append((node.x, node.y))
        node = node.parent
    return pts[::-1]

# ───────────────────────────────────────────── random search ──
def random_search():
    open_list = [Node(0, 0, start_coord[0], start_coord[1], None)]
    closed_set = set()

    # initial frame
    record_frame(open_list, closed_set)

    while open_list:
        # pick and remove a random node
        idx = random.randrange(len(open_list))
        cur = open_list.pop(idx)
        key = (cur.x, cur.y)
        if key in closed_set:
            continue
        closed_set.add(key)

        if key == goal_coord:
            record_frame(open_list, closed_set, reconstruct(cur))
            return

        for nx, ny in neighbours(cur.x, cur.y):
            if maze[ny, nx] == 1 or (nx, ny) in closed_set:
                continue
            # push neighbor with dummy f,g
            open_list.append(Node(0, 0, nx, ny, cur))

        # record after each expansion
        record_frame(open_list, closed_set)

# ─────────────────────────────────────────────────── run & animate ──
random_search()

# colormap (soft colours)
cmap = ListedColormap([
    "white",     # 0 free
    "black",     # 1 wall
    "#4C73E8",      # 2 explored
    "#6CE3F5", # 3 frontier
    "#6BDF6F",      # 4 final path
    "red",      # 5 start
    "red"        # 6 goal
])
norm = BoundaryNorm(np.arange(-0.5,7), cmap.N)

fig, ax = plt.subplots(figsize=(6,6))
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
