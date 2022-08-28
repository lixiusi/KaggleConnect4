import numpy as np
import matplotlib.pyplot as plt
from kaggle_environments import evaluate


def action_space(observation, env_config):
    return [col for col in range(env_config.columns) if observation.board[col] == 0]


def drop_piece(grid, col, piece, config):
    next_grid = grid.copy()
    for row in range(config.rows - 1, -1, -1):
        if next_grid[row][col] == 0:
            break
    next_grid[row][col] = piece
    return next_grid


def check_winning_move(obs, config, col, piece):
    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    next_grid = drop_piece(grid, col, piece, config)
    # horizontal
    for row in range(config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(next_grid[row, col:col + config.inarow])
            if window.count(piece) == config.inarow:
                return True
    # vertical
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns):
            window = list(next_grid[row:row + config.inarow, col])
            if window.count(piece) == config.inarow:
                return True
    # positive diagonal
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(next_grid[range(row, row + config.inarow), range(col, col + config.inarow)])
            if window.count(piece) == config.inarow:
                return True
    # negative diagonal
    for row in range(config.inarow - 1, config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(next_grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
            if window.count(piece) == config.inarow:
                return True
    return False


def win_percentage(player, opponent, num_episodes=10):
    episodes = num_episodes // 2
    outcomes = evaluate("connectx", [player, opponent], num_episodes=episodes)
    outcomes += [[b, a] for [a, b] in evaluate("connectx", [opponent, player], num_episodes=num_episodes - episodes)]
    wins = outcomes.count([1, -1])
    return float((np.sum(wins) / len(outcomes)) * 100)


def visualize_board_and_q_values(board, q_values, n_rows=6, n_cols=7):
    fig, (q_bars_ax, board_ax) = plt.subplots(
        nrows=2, sharex=True, gridspec_kw={'height_ratios': [1, 3]}, figsize=(8, 10)
    )

    q_bars_ax.bar(x=np.arange(n_cols), height=q_values.cpu().detach().numpy())

    board_ax.set_ylim([0, 7])
    board_ax.set_facecolor("lightgrey")
    board_ax.set_xticks(np.arange(n_cols + 1) - 0.5)
    board_ax.set_yticks(np.arange(n_rows + 1) + 0.5)
    board_ax.tick_params(axis='x', colors=(0, 0, 0, 0))
    board_ax.tick_params(axis='y', colors=(0, 0, 0, 0))

    board = board.view(n_rows, n_cols).cpu().detach().numpy() - 1
    marker_colours = ["dodgerblue", "orangered"]
    for i in range(n_rows):
        for j in range(n_cols):
            if board[i][j] >= 0:
                board_ax.scatter(x=j, y=n_rows - i, marker='o', c=marker_colours[board[i][j]], s=1000)
    board_ax.grid()

    return fig
