import minimax
import rules
import numpy as np
import ast

PATH = 'DATASET.txt'
DATASET_LENGTH = 100154

def fetch(n):

    with open(PATH, 'r') as file:
            for i, line in enumerate(file, 1):
                if i == n:
                    return line.strip()
            return None

def process(input_str):

    parts = input_str.split('-')
    if len(parts) != 2:
        return "wrong format"
    first_part, second_part = parts

    replacement = {'O': '0', 'M': '1', 'E': '2'}
    state = [int(replacement.get(char, char)) for char in first_part]

    coords = []
    letters = second_part[0::2]
    digits = second_part[1::2]

    if int(first_part[24]) != 0:
        x = ord(letters[0]) - ord('a')
        y = 7 - int(digits[0])
        coords = (x,y)

    else:
        x = ord(letters[0]) - ord('a')
        y = 7 - int(digits[0])
        coords.append((x,y))
        x = ord(letters[1]) - ord('a')
        y = 7 - int(digits[1])
        coords.append((x,y))

    return state, coords

def make_set(n):
    training_set = []
    i = 1
    while len(training_set) < n and i < DATASET_LENGTH:
        state, coords = process(fetch(i))
        if state[26] > 3 and state[27] > 3:
            training_set.append((state, coords))
        i += 1
    return training_set

def minimax_make_set(depth, n, output_path):
    training_set = []
    i = 1
    while len(training_set) < n and i < DATASET_LENGTH:
        state, data_coords = process(fetch(i))
        if state[26] > 3 and state[27] > 3:
            board, move_count = rules.state_to_board(state)
            coords = minimax.minimax(board, depth, 1 if move_count % 2 == 0 else 2, True, move_count, -float('inf'), float('inf'))[2]
            if coords not in rules.VALID_POSITIONS:
                coords = data_coords
            training_set.append((state, coords))
        i += 1
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in training_set:
            f.write(str(item) + '\n')

def generate_symmetries(input_path, output_path):

    input = []
    with open(input_path, 'r') as infile:
        for line in infile:
            input.append(ast.literal_eval(line.strip()))

    with open(output_path, 'w', encoding='utf-8') as outfile:

        def board_to_matrix(board):
            matrix = [[0 for _ in range(7)] for _ in range(7)]
            index = 0
            for i in range(7):
                for j in range(7):
                    if (i, j) in rules.VALID_POSITIONS:
                        matrix[i][j] = board[index]
                        index += 1
            return matrix

        def rotate_coords(coords):
            x, y = coords
            new_best_moves = []
            new_best_moves.append((x, y))
            new_best_moves.append((y, 6 - x))
            new_best_moves.append((6 - x, 6 - y))
            new_best_moves.append((6 - y, x))

            mirrored_moves = []
            for mx, my in new_best_moves:
                mirrored_moves.append((mx, 6 - my))

            return new_best_moves, mirrored_moves

        def rotate_point(point):
            if type(point) is tuple:
                return rotate_coords(point)
            else:
                p1, m1 = rotate_coords(point[0])
                p2, m2 = rotate_coords(point[1])
            new_best_moves = []
            mirrored_moves = []
            for i in range(4):
                new_best_moves.append([p1[i], p2[i]])
                mirrored_moves.append([m1[i], m2[i]])
            return new_best_moves, mirrored_moves

        def matrix_to_board(matrix, board):
            new_board = []
            for i in range(7):
                for j in range(7):
                    if (i, j) in rules.VALID_POSITIONS:
                        new_board.append(int(matrix[i][j]))
            new_board.extend(board[24:28])
            return new_board

        for board, best_move in input:
            board_matrix = board_to_matrix(board)
            new_best_moves, mirrored_moves = rotate_point(best_move)
            outfile.write(str((board, new_best_moves[0])) + '\n')
            for i in range(3):
                board_matrix = np.rot90(board_matrix)
                outfile.write(str((matrix_to_board(board_matrix, board), new_best_moves[i+1])) + '\n')
            mirrored_matrix = np.fliplr(board_matrix)
            outfile.write(str((matrix_to_board(mirrored_matrix, board), mirrored_moves[0])) + '\n')
            for i in range(3):
                mirrored_matrix = np.rot90(mirrored_matrix)
                outfile.write(str((matrix_to_board(mirrored_matrix, board), mirrored_moves[i+1])) + '\n')

