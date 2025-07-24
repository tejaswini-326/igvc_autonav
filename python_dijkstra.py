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
GRID_W, GRID_H   = 50, 50
WALL_PROB        = 0.3
VIS_DELAY_MS     = 0
DIAGONALS        = False

# ──────────────────────────────────────────────────────── utilities ─
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
start_coord = (1,1)
goal_coord  = (GRID_W-2, GRID_H-2)
maze[start_coord[1], start_coord[0]] = 0
maze[goal_coord[1],  goal_coord[0]]  = 0

# ────────────────────────────────────────── capture animation frames ──
frames = []
def record_frame(open_set, closed_set, path=None):
    grid = maze.copy()
    for x,y in closed_set:    grid[y,x] = 2
    for n in open_set:        grid[n.y,n.x] = 3
    if path:
        for x,y in path:      grid[y,x] = 4
    grid[start_coord[1], start_coord[0]] = 5
    grid[goal_coord[1],  goal_coord[0]]  = 6
    frames.append(grid)

def reconstruct(node):
    pts=[]
    while node:
        pts.append((node.x,node.y))
        node = node.parent
    return pts[::-1]

# ────────────────────────────────────────────── Dijkstra’s ──
def dijkstra():
    tic = count()  # for FIFO tie‑break
    # Node.f == g here
    start = Node(f=0, g=0, x=start_coord[0], y=start_coord[1], parent=None)
    open_heap = []
    heapq.heappush(open_heap, (0, next(tic), start))
    best_g = {start_coord: 0}
    closed_set = set()

    # initial frame
    record_frame([n for _,__,n in open_heap], closed_set)

    while open_heap:
        _,__, cur = heapq.heappop(open_heap)
        key = (cur.x, cur.y)
        if key in closed_set:
            continue
        closed_set.add(key)

        if key == goal_coord:
            record_frame([n for _,__,n in open_heap], closed_set, reconstruct(cur))
            return

        for nx, ny in neighbours(cur.x, cur.y):
            if maze[ny,nx] == 1 or (nx,ny) in closed_set:
                continue

            step_cost = 1.4142 if DIAGONALS and (nx!=cur.x and ny!=cur.y) else 1
            g_new = cur.g + step_cost
            if (nx,ny) not in best_g or g_new < best_g[(nx,ny)]:
                node = Node(f=g_new, g=g_new, x=nx, y=ny, parent=cur)
                heapq.heappush(open_heap, (node.f, next(tic), node))
                best_g[(nx,ny)] = g_new

        # record after each expansion
        record_frame([n for _,__,n in open_heap], closed_set)

# ────────────────────────────────────────────────── run & animate ──
# swap between astar() and dijkstra() here:
dijkstra()
# astar()  # if you'd like to compare

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
