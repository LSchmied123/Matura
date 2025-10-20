import pygame
import ast

def draw_board_pygame(board, screen):
    screen.fill((255, 255, 255))
    cell_size = 40
    offset = 25
    colors = {0: (200, 200, 200), 1: (0, 0, 0), 2: (200, 0, 0), -1: (255, 255, 255)}
    for row in range(len(board)):
        for col in range(len(board[row])):
            x = offset + col * cell_size
            y = offset + row * cell_size
            color = colors.get(board[row][col], (100, 100, 100))
            pygame.draw.circle(screen, color, (x, y), 15)
    pygame.display.flip()

def animate_game(board_history, speed=1.0):
    pygame.init()
    screen_size = (350, 350)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Nine Men's Morris Animation")

    clock = pygame.time.Clock()
    running = True
    idx = 0
    last_update = pygame.time.get_ticks()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    idx = 0  # Restart animation

        now = pygame.time.get_ticks()
        if now - last_update > speed * 1000:
            last_update = now
            if idx < len(board_history):
                board_key = board_history[idx]
                board = [list(map(int, row)) for row in ast.literal_eval(board_key)]
                draw_board_pygame(board, screen)
                idx += 1

        clock.tick(60)  # Limit to 60 FPS