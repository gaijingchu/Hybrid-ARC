"""Grid <-> token serialization shared by every task (rule-toy now; Maze/Sudoku/ARC later).

A "task instance" is few-shot: some demo (input_grid -> output_grid) pairs that reveal a hidden
rule, then a test input whose output the model must produce. We serialize it as a flat token
sequence so a plain causal/bidirectional transformer can consume it. Fixed grid size keeps every
example the same length (clean batching, no padding) for the Phase-1a MVP.

Token id layout (single shared vocab):
  0..N_SPECIAL-1 : special tokens
  N_SPECIAL..    : colors  (color c -> N_SPECIAL + c)
"""
import numpy as np

PAD, BOS, EOS, SEP_IO, SEP_PAIR, NL, MASKG = range(7)  # MASKG = absorbing mask for flat-DLM
N_SPECIAL = 7


def vocab_size(n_colors):
    return N_SPECIAL + n_colors


def color_id(c):
    return N_SPECIAL + int(c)


def serialize_grid(grid):
    """grid: (H,W) int array -> list[int] of cell tokens with a NL after each row."""
    out = []
    for row in grid:
        out.extend(color_id(c) for c in row)
        out.append(NL)
    return out


def grid_len(H, W):
    return H * (W + 1)  # W cells + 1 NL per row


def build_context(demos, test_in):
    """[BOS] (demo_in SEP_IO demo_out SEP_PAIR)* test_in SEP_IO   -- the answer (test_out) follows."""
    ids = [BOS]
    for din, dout in demos:
        ids += serialize_grid(din) + [SEP_IO] + serialize_grid(dout) + [SEP_PAIR]
    ids += serialize_grid(test_in) + [SEP_IO]
    return ids


def build_answer(test_out):
    return serialize_grid(test_out) + [EOS]


def parse_grid(tokens, H, W):
    """Inverse of serialize_grid over a flat token list -> (H,W) array, or None if malformed.
    Reads exactly H rows of W color tokens each, NL-separated; stops at EOS."""
    grid = np.full((H, W), -1, dtype=np.int64)
    r = c = 0
    for t in tokens:
        if t == EOS:
            break
        if t == NL:
            if c != W:
                return None
            r += 1; c = 0
            if r == H:
                return grid
            continue
        if t < N_SPECIAL:          # unexpected special inside a row
            return None
        if r >= H or c >= W:
            return None
        grid[r, c] = t - N_SPECIAL
        c += 1
    return grid if r == H else None
