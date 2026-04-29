import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random

base_dir = os.path.dirname(os.path.abspath(__file__))

def get_valid_moves(pos, board_size, visited):
    x, y = pos
    moves = [(x+2,y+1), (x+2,y-1), (x-2,y+1), (x-2,y-1), 
             (x+1,y+2), (x+1,y-2), (x-1,y+2), (x-1,y-2)]
    return [(mx, my) for mx, my in moves if 0 <= mx < board_size and 0 <= my < board_size and (mx, my) not in visited]

# 1. Blind Search (DFS)
def dfs(pos, board_size, visited, nodes_expanded, limit=3000):
    nodes_expanded[0] += 1
    if nodes_expanded[0] > limit: return None
    if len(visited) == board_size * board_size: return visited
    for move in get_valid_moves(pos, board_size, visited):
        visited.append(move)
        if dfs(move, board_size, visited, nodes_expanded, limit): return visited
        visited.pop()
    return None

# 2. Standard Heuristic (Warnsdorff's)
def warnsdorff_standard(board_size):
    pos = (0, 0)
    visited = [pos]
    nodes = 0
    while len(visited) < board_size * board_size:
        nodes += 1
        current = visited[-1]
        moves = get_valid_moves(current, board_size, visited)
        if not moves: return None, nodes
        moves.sort(key=lambda m: len(get_valid_moves(m, board_size, visited + [m])))
        visited.append(moves[0])
    return visited, nodes

# 3. Robust Adaptive Heuristic (Warnsdorff with Random Tie-Breaking)
def warnsdorff_robust(board_size, start_pos=(3, 3)):
    pos = start_pos
    visited = [pos]
    nodes = 0
    while len(visited) < board_size * board_size:
        nodes += 1
        current = visited[-1]
        moves = get_valid_moves(current, board_size, visited)
        if not moves: return None, nodes
        # Calculate degrees
        degrees = [(len(get_valid_moves(m, board_size, visited + [m])), m) for m in moves]
        min_deg = min(d for d, m in degrees)
        best_moves = [m for d, m in degrees if d == min_deg]
        visited.append(random.choice(best_moves))
    return visited, nodes

# Run and Plot
def plot_board(path, size, algo_name):
    if path is None: return
    board = np.zeros((size, size))
    for step_num, (x, y) in enumerate(path):
        board[x][y] = step_num + 1
    
    plt.figure(figsize=(6,6))
    plt.imshow(board, cmap='Blues', interpolation='nearest')
    for i in range(size):
        for j in range(size):
            plt.text(j, i, int(board[i, j]), ha='center', va='center', color='black')
    plt.title(f"Knight's Tour: {algo_name} ({size}x{size})")
    plt.savefig(os.path.join(base_dir, f'viz_{algo_name}_{size}.png'))
    plt.close()

results = []
for size in [8]:
    # Run all three
    path1, n1 = dfs((0,0), size, [(0,0)], [0]), 3000
    path2, n2 = warnsdorff_standard(size)
    path3, n3 = warnsdorff_robust(size, start_pos=(3,3))
    
    results.append({'Algo': 'DFS (Blind)', 'Nodes': n1 if path1 else 'Failed'})
    results.append({'Algo': 'Standard Warnsdorff', 'Nodes': n2})
    results.append({'Algo': 'Robust Adaptive', 'Nodes': n3})
    
    # Plot successful tours
    if path1: plot_board(path1, size, 'DFS')
    if path2: plot_board(path2, size, 'Standard_Warnsdorff')
    if path3: plot_board(path3, size, 'Robust_Adaptive')

pd.DataFrame(results).to_csv(os.path.join(base_dir, 'comparison.csv'), index=False)
print("Comparison complete. Check 'comparison.csv' and PNG files in your folder.")