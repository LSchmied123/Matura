import math
import copy
import rules

# try moves that form a mill first to improve pruning
def _forms_mill_fast(board, move, player):
    # Determine the destination
    if isinstance(move, tuple):
        pos = move
    elif isinstance(move, list) and len(move) == 2:
        pos = move[1]
    else:
        return False

    # another fail-safe check
    if board[pos[0]][pos[1]] != 0:
        return False

    # cant use rules.is_mill here since it needs the move to be fully applied
    for line in rules.mill_lines:
        if pos in line:
            other = [p for p in line if p != pos]
            if board[other[0][0]][other[0][1]] == player and board[other[1][0]][other[1][1]] == player:
                return True
    return False

def minimax(board, depth, player, is_maximizing, move_count, alpha, beta, _tt=None):
    if _tt is None:
        _tt = {}

    # Terminal evaluation
    if depth == 0:
        return rules.evaluate_board(board, player), [], None

    # Early terminal detection
    if rules.check_game_over(board, move_count):
        c1, c2 = rules.count_pieces(board)
        if player == 1:
            if c2 < 3:
                return 10**9, [], None  # player wins
            if c1 < 3:
                return -10**9, [], None  # player loses
        else:
            if c1 < 3:
                return 10**9, [], None
            if c2 < 3:
                return -10**9, [], None
        # more unnecessary failsafe
        return rules.evaluate_board(board, player), [], None

    # Transposition table lookup makes code much faster
    tt_key = (rules.board_to_key(board), depth, is_maximizing, move_count, player)
    if tt_key in _tt:
        return _tt[tt_key]
    opponent = 1 if player == 2 else 2
    best_score = -math.inf if is_maximizing else math.inf
    best_move = None
    best_move_chain = []

    # Generate and order moves
    current_actor = player if is_maximizing else opponent
    moves = rules.get_possible_moves(board, current_actor, move_count)
    moves.sort(key=lambda m: _forms_mill_fast(board, m, current_actor), reverse=True)

    if not moves:
        # more unnecessary failsafe
        eval_score = rules.evaluate_board(board, player)
        result = (eval_score, [], None)
        _tt[tt_key] = result
        return result

    for move in moves:
        new_board = copy.deepcopy(board)
        success, new_board = rules.apply_move(new_board, move, current_actor, deterministic=True)
        if not success:
            continue
        sub_score, sub_chain, _ = minimax(new_board, depth - 1, player, not is_maximizing, move_count + 1, alpha=alpha, beta=beta, _tt=_tt)
        if is_maximizing:
            if sub_score > best_score:
                best_score = sub_score
                best_move = move
                best_move_chain = [move] + sub_chain
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        else:
            if sub_score < best_score:
                best_score = sub_score
                best_move = move
                best_move_chain = [move] + sub_chain
            beta = min(beta, best_score)
            if beta <= alpha:
                break
    result = (best_score, best_move_chain, best_move)
    _tt[tt_key] = result
    return result
