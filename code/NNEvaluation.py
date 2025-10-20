import torch
import rules
import NeuralNetData
import minimax
import display
from animate_game import animate_game
import random
import math

def initialize_model():
    policy_net = NeuralNetData.BiggerPolicyNetwork(state_size=28, action_size=88)
    policy_net.load_state_dict(torch.load('selfplay_checkpoints_large/selfplay_model_iter_2600.pth'))
    policy_net.eval()
    return policy_net

def initialize_variables():

    board = [
        [0, -1, -1, 0, -1, -1, 0],
        [-1, 0, -1, 0, -1, 0, -1],
        [-1, -1, 0, 0, 0, -1, -1],
        [0, 0, 0, -1, 0, 0, 0],
        [-1, -1, 0, 0, 0, -1, -1],
        [-1, 0, -1, 0, -1, 0, -1],
        [0, -1, -1, 0, -1, -1, 0],
    ]
    move_count = 0
    game_moves = []
    board_history = []
    depth = 3 

    return board, move_count, depth, board_history, game_moves

def match(player1, player2, matches, policy_net, opening_random_moves: int = 0):
    board, move_count, depth, board_history, game_moves = initialize_variables()
    player1wins = 0
    player2wins = 0
    draws = 0
    board_history.append(rules.board_to_key(board))
    if player1 is None and player2 is None:
        player1 = input("Choose Player 1 (1 for Minimax, 2 for NN, 3 for Human, 4 for random move selection): ")
        player2 = input("Choose Player 2 (1 for Minimax, 2 for NN, 3 for Human, 4 for random move selection): ")
        if player1 not in ['1', '2', '3', '4'] or player2 not in ['1', '2', '3', '4']:
            print("Invalid choice. Please choose 1, 2, 3 or 4 for both players.")
            return
        if player1 and player2 in ['1', '2', '4']:
            matches = int(input("How many matches should be played? "))
        else:
            matches = 1

    for match_num in range(matches):
        board, move_count, depth, board_history, game_moves = initialize_variables()
        winner = None
        while True:
            if move_count % 2 == 0:
                # If within opening randomization window, force a random legal move
                if opening_random_moves and move_count < opening_random_moves:
                    board, move = random_player_move(board, move_count, 1)
                else:
                    if player1 == '1':
                        board, move = minimax_player_move(board, depth, 1, move_count)
                    elif player1 == '2':
                        board, move = NN_player_move(board, move_count, 1, policy_net)
                    elif player1 == '3':
                        move_raw = display.display_board(board)
                        board, move = player_move(board, move_count, 1, move_raw)
                    elif player1 == '4':
                        board, move = random_player_move(board, move_count, 1)
                if move is None:
                    winner = "Player 2"
                    break
                game_moves.append(move)
                board_history.append(rules.board_to_key(board))
                move_count += 1
                game_over = rules.check_game_over(board, move_count)
                if game_over:
                    winner = "Player 1"
                    break
                if board_history.count(rules.board_to_key(board)) >= 3:
                    winner = None
                    break
            else:
                # If within opening randomization window, force a random legal move
                if opening_random_moves and move_count < opening_random_moves:
                    board, move = random_player_move(board, move_count, 2)
                else:
                    if player2 == '1':
                        board, move = minimax_player_move(board, depth, 2, move_count)
                    elif player2 == '2':
                        board, move = NN_player_move(board, move_count, 2, policy_net)
                    elif player2 == '3':
                        move_raw = display.display_board(board)
                        board, move = player_move(board, move_count, 2, move_raw)
                    elif player2 == '4':
                        board, move = random_player_move(board, move_count, 2)
                if move is None:
                    winner = "Player 1"
                    break
                game_moves.append(move)
                board_history.append(rules.board_to_key(board))
                move_count += 1
                game_over = rules.check_game_over(board, move_count)
                if game_over:
                    winner = "Player 2"
                    break
                if board_history.count(rules.board_to_key(board)) >= 3:
                    winner = None
                    break

        if winner:
            if winner == "Player 1":
                player1wins += 1
            else:
                player2wins += 1
        else:
            draws += 1

    if matches > 1:
        return player1, player2, player1wins, player2wins, draws
    else:
        if winner:
            print(f"{winner} wins!")
        else:
            print("It's a draw due to threefold repetition!")
        return game_moves, board, board_history

def player_move(board, move_count, player, move_raw):
    move = move_raw
    move_parts = move.split()
    if move_parts[0] == 'place' and len(move_parts) == 3:
        try:
            row, col = int(move_parts[1]), int(move_parts[2])
            action = (row, col)
        except ValueError:
            print("Invalid input. Please enter integers for row and column.")
            display.display_board(board)
    elif move_parts[0] == 'move' and len(move_parts) == 5:
        try:
            from_row, from_col, to_row, to_col = map(int, move_parts[1:])
            action = (from_row, from_col, to_row, to_col)
        except ValueError:
            print("Invalid input. Please enter integers for rows and columns.")
            display.display_board(board)
    else:
        print("Invalid command. Use 'place' or 'move'.")
        display.display_board(board)
    possible_actions = rules.get_possible_moves(board, player, move_count=move_count)
    if action not in possible_actions:
        print("Invalid move. Try again.")
        display.display_board(board)
    _, board = rules.apply_move(board, action, player)
    return board, action

def minimax_player_move(board, depth, player, move_count):
    alpha, beta = -math.inf, math.inf
    _, _, best_move = minimax.minimax(board, depth, player, is_maximizing=True, move_count=move_count, alpha=alpha, beta=beta)
    _, board = rules.apply_move(board, best_move, player)
    return board, best_move

def NN_player_move(board, move_count, player, policy_net):
    actions = rules.get_actions()
    player1_pieces, player2_pieces = rules.count_pieces(board)
    selected_action = None
    board_positions = [board[row][col] for (row, col) in rules.VALID_POSITIONS]
    count1 = sum(cell == 1 for row in board for cell in row)
    count2 = sum(cell == 2 for row in board for cell in row)
    state = board_positions + [player1_pieces, player2_pieces, count1, count2]
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = policy_net(state_tensor)
        probs = torch.softmax(logits, dim=1)
        sorted_indices = torch.argsort(probs, dim=1, descending=True)[0]  # Get 1D tensor of indices
        possible_actions = rules.get_possible_moves(board, player, move_count=move_count)
        for idx in sorted_indices:
            top_action = actions[idx.item()]
            if top_action in possible_actions:
                selected_action = top_action
                break

    if selected_action:
        _, board = rules.apply_move(board, selected_action, player)
        return board, selected_action
    else:
        return board, None

def random_player_move(board, move_count, player):
    possible_actions = rules.get_possible_moves(board, player, move_count=move_count)
    if possible_actions:
        selected_action = random.choice(possible_actions)
        _, board = rules.apply_move(board, selected_action, player)
        return board, selected_action
    else:
        return board, None  # No move possible

if __name__ == "__main__":
    policy_net = initialize_model()
    players = ['Minimax', 'NN', 'Human', 'Random']
    result = match(None, None, None, policy_net)
    if isinstance(result[2], int):  # Multiple matches
        player1, player2, player1wins, player2wins, draws = result
        player1 = players[int(player1)-1]
        player2 = players[int(player2)-1]
        print(f"{player1} wins: {player1wins}, {player2} wins: {player2wins}, Draws: {draws}")
        display.graph_wins(player1, player2, player1wins, player2wins, draws)
    else:
        game_moves, board, board_history = result
        #print("Move chain:", game_moves)
        #display.display_board(board)
        animate_game(board_history)
