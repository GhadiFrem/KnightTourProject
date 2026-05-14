# ============================================================
# KNIGHT'S TOUR PROBLEM — AI Search Algorithms Comparison
# ============================================================
# The Knight's Tour is a classic AI/chess problem:
#   Given a knight placed on an N×N chessboard, find a sequence
#   of moves that visits every square exactly once.
#
# This project compares THREE algorithms:
#   1. Blind DFS (Depth-First Search) — no guidance, brute-force
#   2. Warnsdorff's Heuristic (Standard) — greedy, deterministic
#   3. Warnsdorff's Heuristic (Robust/Adaptive) — greedy + random tie-breaking
# ============================================================

import time
import pandas as pd  # Used to save the results comparison table as a CSV
import matplotlib.pyplot as plt  # Used to draw and save the board visualizations
import numpy as np  # Used to create the 2D board grid for visualization
import os  # Used to build file paths that work on any OS
import random  # Used in the robust heuristic for random tie-breaking

# Resolve the directory where this script lives, so output files are saved next to it
base_dir = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# HELPER: Get all valid knight moves from a position
# ============================================================
# A knight in chess moves in an "L" shape:
#   2 squares in one direction + 1 square perpendicular (or vice versa).
# That gives exactly 8 possible target squares from any position.
#
# This function returns only the moves that are:
#   - Still inside the board (0 <= x,y < board_size)
#   - Not already visited (to avoid revisiting squares)
def get_valid_moves(pos, board_size, visited):
    x, y = pos
    # All 8 possible L-shaped knight moves from (x, y)
    moves = [
        (x + 2, y + 1),
        (x + 2, y - 1),
        (x - 2, y + 1),
        (x - 2, y - 1),
        (x + 1, y + 2),
        (x + 1, y - 2),
        (x - 1, y + 2),
        (x - 1, y - 2),
    ]
    # Filter: keep only moves that are on the board and not yet visited
    return [
        (mx, my)
        for mx, my in moves
        if 0 <= mx < board_size and 0 <= my < board_size and (mx, my) not in visited
    ]


# ============================================================
# ALGORITHM 1: Blind Search — Depth-First Search (DFS)
# ============================================================
# DFS explores paths by going as deep as possible before backtracking.
# It has NO heuristic — it blindly tries each move in order.
#
# How it works:
#   - Start at a position, mark it visited.
#   - Recursively try each valid move.
#   - If we reach a dead end (no moves left and board not complete), backtrack.
#   - If all squares are visited, we found a complete tour.
#
# Weakness: Without guidance, DFS may explore many dead-end paths.
#   The `limit` cap (3000 nodes) prevents it from running forever.
#
# Parameters:
#   pos            — current position on the board
#   board_size     — N (board is N×N)
#   visited        — list of squares visited so far (in order)
#   nodes_expanded — mutable counter [int] tracking how many nodes were explored
#   limit          — maximum nodes to expand before giving up
def dfs(pos, board_size, visited, nodes_expanded, limit=3000):
    nodes_expanded[0] += 1  # Count this node as expanded
    if nodes_expanded[0] > limit:
        return None  # Abort if we hit the node limit
    if len(visited) == board_size * board_size:  # All squares visited → success!
        return visited
    for move in get_valid_moves(pos, board_size, visited):
        visited.append(move)  # Choose this move
        if dfs(move, board_size, visited, nodes_expanded, limit):
            return visited  # Propagate success upward
        visited.pop()  # Backtrack: undo the move
    return None  # No valid move found → backtrack


# ============================================================
# ALGORITHM 2: Standard Warnsdorff's Heuristic
# ============================================================
# Warnsdorff's Rule (1823) is a greedy heuristic:
#   At each step, move to the square that has the FEWEST onward moves.
#
# Intuition: By visiting "trapped" or hard-to-reach squares early,
#   we avoid getting stuck in corners later in the tour.
#
# This version always starts from (0, 0) and breaks ties
#   by simply picking the first option (deterministic).
#
# It is much faster than DFS because there is no backtracking —
#   it makes a single greedy decision at every step.
#
# Parameters:
#   board_size — N (board is N×N)
def warnsdorff_standard(board_size):
    pos = (0, 0)  # Always start from the top-left corner
    visited = [pos]  # Track the order of visited squares
    nodes = 0  # Count how many steps (decisions) were made

    while len(visited) < board_size * board_size:
        nodes += 1
        current = visited[-1]  # Current position
        moves = get_valid_moves(current, board_size, visited)  # All valid next moves
        if not moves:
            return None, nodes  # Dead end — tour failed

        # Sort moves by how many onward moves each leads to (ascending = fewest first)
        # This is Warnsdorff's Rule: prefer the move with the least future options
        moves.sort(key=lambda m: len(get_valid_moves(m, board_size, visited + [m])))
        visited.append(moves[0])  # Always pick the move with the fewest onward moves

    return visited, nodes  # Return the complete tour and node count


# ============================================================
# ALGORITHM 3: Robust Adaptive Warnsdorff's (with Random Tie-Breaking)
# ============================================================
# This is an enhanced version of Warnsdorff's heuristic.
# The key difference: when multiple moves tie for fewest onward moves,
#   this algorithm picks ONE OF THEM AT RANDOM instead of always the first.
#
# Why this helps:
#   - Standard Warnsdorff can fail on some starting positions due to bad tie resolution.
#   - Randomizing tie-breaking explores different paths and avoids systematic failures.
#   - It can also be run multiple times; different runs may succeed where others fail.
#
# Additional feature: supports a configurable starting position (not just (0,0)).
#
# Parameters:
#   board_size — N (board is N×N)
#   start_pos  — (row, col) starting square for the knight
def warnsdorff_robust(board_size, start_pos=(3, 3)):
    pos = start_pos
    visited = [pos]  # Start from the given position
    nodes = 0

    while len(visited) < board_size * board_size:
        nodes += 1
        current = visited[-1]
        moves = get_valid_moves(current, board_size, visited)
        if not moves:
            return None, nodes  # Dead end — tour failed

        # Compute the degree (number of onward moves) for each candidate move
        degrees = [
            (len(get_valid_moves(m, board_size, visited + [m])), m) for m in moves
        ]

        # Find the minimum degree among all candidates
        min_deg = min(d for d, m in degrees)

        # Collect ALL moves that share the minimum degree (the "best" moves)
        best_moves = [m for d, m in degrees if d == min_deg]

        # Randomly pick one of the best moves — this is the key difference from standard
        visited.append(random.choice(best_moves))

    return visited, nodes  # Return the complete tour and node count


# ============================================================
# VISUALIZATION: Draw and save the knight's tour as a PNG image
# ============================================================
# Creates a colored board where each cell shows the step number
# at which the knight visited that square (1 = first, N² = last).
# The board is saved as a PNG file next to the script.
#
# Parameters:
#   path      — ordered list of (row, col) positions the knight visited
#   size      — board dimension N
#   algo_name — string label used in the title and filename
def plot_board(path, size, algo_name):
    if path is None:
        return  # Skip if the algorithm failed to find a tour

    # Create an N×N grid initialized to 0
    board = np.zeros((size, size))

    # Fill each square with its visit order (step_num + 1 for 1-based numbering)
    for step_num, (x, y) in enumerate(path):
        board[x][y] = step_num + 1

    plt.figure(figsize=(6, 6))
    plt.imshow(
        board, cmap="Blues", interpolation="nearest"
    )  # Color cells by step number

    # Overlay the step number as text inside each cell
    for i in range(size):
        for j in range(size):
            plt.text(j, i, int(board[i, j]), ha="center", va="center", color="black")

    plt.title(f"Knight's Tour: {algo_name} ({size}x{size})")
    plt.savefig(os.path.join(base_dir, f"viz_{algo_name}_{size}.png"))  # Save to file
    plt.close()  # Free memory


# ============================================================
# MAIN: Run all three algorithms and compare their performance
# ============================================================
# For each board size (currently 8×8), we:
#   1. Run DFS (blind search) with a 3000-node expansion limit
#   2. Run Standard Warnsdorff's from (0,0)
#   3. Run Robust Warnsdorff's from (3,3)
#   4. Record how many nodes each algorithm expanded
#   5. Save board visualizations for successful tours
#   6. Export a CSV summary comparing all three algorithms

results = []

for size in [8]:  # Board size: currently only 8×8; can be extended to [5, 6, 8] etc.

    # --- Algorithm 1: DFS ---
    # nodes_expanded is passed as a mutable list [0] so the recursive function can update it
    # path1 is either the full tour (list of positions) or None if DFS failed/hit the limit
    path1, n1 = dfs((0, 0), size, [(0, 0)], [0]), 3000

    # --- Algorithm 2: Standard Warnsdorff's ---
    path2, n2 = warnsdorff_standard(size)

    # --- Algorithm 3: Robust Adaptive Warnsdorff's ---
    path3, n3 = warnsdorff_robust(size, start_pos=(3, 3))

    # Record results for the comparison table
    # DFS shows 'Failed' in the Nodes column if it didn't find a complete tour
    results.append({"Algo": "DFS (Blind)", "Nodes": n1 if path1 else "Failed"})
    results.append({"Algo": "Standard Warnsdorff", "Nodes": n2})
    results.append({"Algo": "Robust Adaptive", "Nodes": n3})

    # Save board images only for successful tours
    if path1:
        plot_board(path1, size, "DFS")
    if path2:
        plot_board(path2, size, "Standard_Warnsdorff")
    for path3, start_pos in robust_paths:
        if path3:
            plot_board(
                path3, size, f"Robust_Adaptive_start_{start_pos[0]}_{start_pos[1]}"
            )

# Export the comparison table to a CSV file for analysis
pd.DataFrame(results).to_csv(os.path.join(base_dir, "comparison.csv"), index=False)
print("Comparison complete. Check 'comparison.csv' and PNG files in your folder.")
