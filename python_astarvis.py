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
HEURISTIC_WEIGHT = 1.0

# ──────────────────────────────────────────────────────── A* utilities ─
Node = namedtuple("Node", "f h g x y parent")

def heuristic(ax, ay, bx, by):
    base = max(abs(ax-bx), abs(ay-by)) if DIAGONALS else abs(ax-bx) + abs(ay-by)
    return HEURISTIC_WEIGHT * base

def neighbours(x, y):
    steps = [(-1,0),(1,0),(0,-1),(0,1)]
    if DIAGONALS:
        steps += [(-1,-1),(-1,1),(1,-1),(1,1)]
    for dx,dy in steps:
        nx, ny = x+dx, y+dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
            yield nx, ny

# ────────────────────────────────────────────────── build random maze ──
# maze = (np.random.rand(GRID_H, GRID_W) < WALL_PROB).astype(int)
# with open("maze.json", "w") as f:
#     json.dump(maze.tolist(), f)
with open("maze.json", "r") as f:
    maze = np.array(json.load(f))
    
maze[0,    :] = 1   # top row
maze[-1,   :] = 1   # bottom row
maze[:,    0] = 1   # left  col
maze[:,   -1] = 1   # right col

# ─── keep start & goal free ────────────────────────────────────────
maze[1,1]     = 0
maze[-2,-2]   = 0

# ──────────────────────────────────────────── run A* & capture frames ──
frames = []

def record_frame(open_set, closed_set, path=None):
    grid = maze.copy()
    for x,y in closed_set:          grid[y,x] = 2
    for n in open_set:              grid[n.y,n.x] = 3
    if path:
        for x,y in path:            grid[y,x] = 4
    grid[1,1], grid[-2,-2] = 5, 6
    frames.append(grid)

def reconstruct(node):
    pts=[]
    while node: pts.append((node.x,node.y)); node=node.parent
    return pts[::-1]


def astar():
    """Unbiased A* with FIFO tie‑breaking."""
    tic   = count()                                # monotonically‑increasing counter
    start = Node(0, 0, 0, 1, 1, None)
    open_heap  = []
    heapq.heappush(open_heap, (0, next(tic), start))
    best_g     = {(1, 1): 0}
    closed_set = set()

    # first frame
    record_frame([n for _,__,n in open_heap], closed_set)

    while open_heap:
        _, __, cur = heapq.heappop(open_heap)
        key = (cur.x, cur.y)
        if key in closed_set:
            continue
        closed_set.add(key)

        if key == (GRID_W - 2, GRID_H - 2):        # reached goal
            record_frame([n for _,__,n in open_heap], closed_set, reconstruct(cur))
            return                                  # done

        for nx, ny in neighbours(*key):
            if maze[ny, nx] == 1 or (nx, ny) in closed_set:
                continue

            step_cost = 1.4142 if DIAGONALS and (nx != cur.x and ny != cur.y) else 1
            g_new = cur.g + step_cost

            if (nx, ny) not in best_g or g_new < best_g[(nx, ny)]:
                h_new = heuristic(nx, ny, GRID_W - 1, GRID_H - 1)
                node  = Node(g_new + h_new, h_new, g_new, nx, ny, cur)
                heapq.heappush(open_heap, (node.f, next(tic), node))
                best_g[(nx, ny)] = g_new

        # capture this expansion step
        record_frame([n for _,__,n in open_heap], closed_set)


astar()

# ────────────────────────────────────────────── matplotlib animation ──
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
