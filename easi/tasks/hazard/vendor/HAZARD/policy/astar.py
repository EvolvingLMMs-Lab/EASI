import pyastar2d
from typing import List
import numpy as np

def check_mergeable(p1, p2, p3):
    d1 = p2[0] - p1[0], p2[1] - p1[1]
    d2 = p3[0] - p2[0], p3[1] - p2[1]
    # if d1 is too long, don't merge
    if d1[0] * d1[0] + d1[1] * d1[1] > 256:
        return False
    if d1[0] * d2[1] == d1[1] * d2[0]:
        return True
    return False

def path_cleanup(path: List[List[int]]) -> List[List[int]]:
    """
    if two consecutive movements are in the same direction, combine them into one
    """
    q = []
    for i in range(len(path)):
        while len(q) >= 2 and check_mergeable(q[-2], q[-1], path[i]):
            q.pop()
        q.append(path[i])
    return q

def get_astar_path(weight, origin, destination):
    if origin[0] < 0 or origin[0] >= weight.shape[0] or origin[1] < 0 or origin[1] >= weight.shape[1]:
        return None
    if destination[0] < 0 or destination[0] >= weight.shape[0] or destination[1] < 0 or destination[1] >= weight.shape[1]:
        return None
    path = pyastar2d.astar_path(weight, origin, destination)
    path = path_cleanup(path)
    return path

# sem_map: [channel, height, width]
def get_astar_weight(sem_map, origin, destination):
    # unexplored: 5, explored: 1
    # height > 0.1: += 10
    # height > 1: += 1000
    explored = sem_map["explored"]
    height = sem_map["height"]
    w, h = explored.shape
    weight = np.ones((w, h))
    weight[explored == 0] = 30
    
    # exp(x * 0.) * y= 10
    # exp(x * 1.6) * y = 1000
    # x = 3, y = 7
    weight[height > 0.5] += np.exp(height[height > 0.5] * 3) * 7
    weight[height > 0.1] += 10
    # weight[height > 0.5] += 50
    # weight[height > 1.6] += 1000
    # for each position, if it is near an obstacle, increase its weight by 50
    conv_weight = np.zeros((w, h))
    for i in range(-2, 3):
        for j in range(-2, 3):
            conv_weight[max(0, i): min(w, w+i), max(0, j): min(h, h+j)] += weight[max(0, -i): min(w, w-i), max(0, -j): min(h, h-j)] * 1.0 / (abs(i) + abs(j) + 1)
    return conv_weight.astype(np.float32)
