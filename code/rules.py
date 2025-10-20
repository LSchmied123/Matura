import math
import random

connections = {
        (0, 0): [(0, 3), (3, 0)],
        (0, 3): [(0, 0), (0, 6), (1, 3)],
        (0, 6): [(0, 3), (3, 6)],
        (1, 1): [(1, 3), (3, 1)],
        (1, 3): [(1, 1), (1, 5), (0, 3), (2, 3)],
        (1, 5): [(1, 3), (3, 5)],
        (2, 2): [(2, 3), (3, 2)],
        (2, 3): [(2, 2), (2, 4), (1, 3)],
        (2, 4): [(2, 3), (3, 4)],
        (3, 0): [(0, 0), (3, 1), (6, 0)],
        (3, 1): [(3, 0), (3, 2), (1, 1), (5, 1)],
        (3, 2): [(3, 1), (4, 2), (2, 2)],
        (3, 4): [(2, 4), (3, 5), (4, 4)],
        (3, 5): [(3, 4), (3, 6), (1, 5), (5, 5)],
        (3, 6): [(0, 6), (3, 5), (6, 6)],
        (4, 2): [(3, 2), (4, 3)],
        (4, 3): [(4, 2), (4, 4), (5, 3)],
        (4, 4): [(4, 3), (3, 4)],
        (5, 1): [(3, 1), (5, 3)],
        (5, 3): [(5, 1), (5, 5), (4, 3), (6, 3)],
        (5, 5): [(5, 3), (3, 5)],
        (6, 0): [(3, 0), (6, 3)],
        (6, 3): [(6, 0), (6, 6), (5, 3)],
        (6, 6): [(3, 6), (6, 3)],
    }

mill_lines = [
    [(0, 0), (0, 3), (0, 6)],
    [(1, 1), (1, 3), (1, 5)],
    [(2, 2), (2, 3), (2, 4)],
    [(3, 0), (3, 1), (3, 2)],
    [(3, 4), (3, 5), (3, 6)],
    [(4, 2), (4, 3), (4, 4)],
    [(5, 1), (5, 3), (5, 5)],
    [(6, 0), (6, 3), (6, 6)],
    [(0, 0), (3, 0), (6, 0)],
    [(1, 1), (3, 1), (5, 1)],
    [(2, 2), (3, 2), (4, 2)],
    [(0, 3), (1, 3), (2, 3)],
    [(4, 3), (5, 3), (6, 3)],
    [(2, 4), (3, 4), (4, 4)],
    [(1, 5), (3, 5), (5, 5)],
    [(0, 6), (3, 6), (6, 6)],
]

VALID_POSITIONS = [
    (0, 0), (0, 3), (0, 6),
    (1, 1), (1, 3), (1, 5),
    (2, 2), (2, 3), (2, 4),
    (3, 0), (3, 1), (3, 2), (3, 4), (3, 5), (3, 6),
    (4, 2), (4, 3), (4, 4),
    (5, 1), (5, 3), (5, 5),
    (6, 0), (6, 3), (6, 6)
]

def evaluate_board(board, player):
    opponent = 1 if player == 2 else 2
    score = 0
    for row in board:
        for cell in row:
            if cell == player:
                score += 1
            if cell == opponent:
                score -= 1
    '''
    player_mobility = count_mobility(board, player)
    opponent_mobility = count_mobility(board, opponent)
    score += 0.1 * (player_mobility - opponent_mobility)
    '''
    return score

def count_potential_mills(board, player, mill_lines):
    count = 0
    for mill_line in mill_lines:
        values = [board[row][col] for row, col in mill_line]
        if values.count(player) == 2 and values.count(0) == 1:
            count += 1
    return count

def count_mill(board, player, mill_lines):
    count = 0
    for line in mill_lines:
        if all(board[i][j] == player for i, j in line):
            count += 1
    return count

def is_mill(board, player, mill_lines, move):
    lines = []
    for mills in mill_lines:
        if move in mills:
            lines.append(mills)
    if count_mill(board, player, lines) != 0:
        return True
    else:
        return False

def count_mobility(board, player):
    mobility = 0
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == player:
                mobility += sum(1 for ni, nj in connections.get((i, j), []) if board[ni][nj] == 0)
    return mobility

def get_possible_moves(board, player, move_count):

    if move_count < 18:
        moves = []
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == 0:
                    moves.append((i, j))
        return moves

    else:
        possible_moves = []

        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == player:
                    for ni, nj in connections.get((i, j), []):
                        if board[ni][nj] == 0:
                            possible_moves.append([(i, j), (ni, nj)])
        return possible_moves

def apply_move(board, move, player, deterministic=False):
    # Normalize move into a canonical form
    norm = None
    if isinstance(move, tuple):
        if len(move) == 2 and all(isinstance(x, int) for x in move):
            # placement
            norm = ('place', move)
        elif len(move) == 4 and all(isinstance(x, int) for x in move):
            # movement as 4-tuple
            norm = ('move', ((move[0], move[1]), (move[2], move[3])))
        elif len(move) == 2 and all(isinstance(x, tuple) and len(x) == 2 for x in move):
            # movement as tuple of tuples
            norm = ('move', (move[0], move[1]))
    elif isinstance(move, list):
        if len(move) == 2 and all(isinstance(x, tuple) and len(x) == 2 for x in move):
            # movement as list of tuples
            norm = ('move', (move[0], move[1]))

    if norm is None:
        # everything breaks more gracefully now
        return (False, f"Invalid move format: {type(move)} {move}")

    kind, payload = norm
    if kind == 'place':
        row, col = payload
        if board[row][col] != 0:
            return (False, "Spot not empty")
        board[row][col] = player
        if is_mill(board, player, mill_lines, (row, col)):
            remove_random_opponent_piece(board, player, deterministic=deterministic)
        return (True, board)
    else:
        (row1, col1), (row2, col2) = payload
        if (row1, col1) == (row2, col2):
            return (False, "Cant move there")
        if board[row1][col1] != player:
            return (False, "No piece there")
        if board[row2][col2] != 0:
            return (False, "Target not empty")
        # Execute move
        board[row1][col1] = 0
        board[row2][col2] = player
        if is_mill(board, player, mill_lines, (row2, col2)):
            remove_random_opponent_piece(board, player, deterministic=deterministic)
        return (True, board)

def remove_random_opponent_piece(board, player, deterministic: bool = False):
    opponent = 1 if player == 2 else 2
    opponent_positions = [(i, j) for i in range(len(board)) for j in range(len(board[i])) if board[i][j] == opponent]
    # Check if all are mills
    all_in_mills = all(is_mill(board, opponent, mill_lines, pos) for pos in opponent_positions)
    removable = [pos for pos in opponent_positions if all_in_mills or not is_mill(board, opponent, mill_lines, pos)]
    if removable:
        if deterministic:
            # Deterministic choice for search stability: choose the first in scan order
            pos = removable[0]
        else:
            pos = random.choice(removable)
        board[pos[0]][pos[1]] = 0

def check_game_over(board, move_count):

    count1 = sum(cell == 1 for row in board for cell in row)
    count2 = sum(cell == 2 for row in board for cell in row)

    if ((count1 < 3) or (count2 < 3)) and move_count > 18:
        return True
    else:
        return False

def get_actions():
    actions = []
    for element in VALID_POSITIONS:
        actions.append(element)
    for start, ends in connections.items():
            for end in ends:
                actions.append([start, end])
    return actions

def board_to_key(board):
    return str(tuple(tuple(row) for row in board))

def count_pieces(board):
    count1 = sum(cell == 1 for row in board for cell in row)
    count2 = sum(cell == 2 for row in board for cell in row)
    return count1, count2

def state_to_board(state):
    board = [[0 for _ in range(7)] for _ in range(7)]
    index = 0
    for i in range(7):
        for j in range(7):
            if (i, j) in VALID_POSITIONS:
                board[i][j] = state[index]
                index += 1
    player1_pieces, player2_pieces = state[24], state[25]
    if player1_pieces + player2_pieces > 0:
        move_count = 18 - player1_pieces - player2_pieces
    else:
        move_count = 20
    return board, move_count

