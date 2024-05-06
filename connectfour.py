import pygame
import sys
import numpy as np
from aiconnectfour import Bot, Connect
from pygame.locals import *
 
HEIGHT = 450
WIDTH = 400
FPS = 60
FRAMEPERSEC = pygame.time.Clock()

ROWS = 6
COLUMNS = 7

board = np.zeros((6, 7))
board[0, 6] = 1

circle_radius = HEIGHT * .05

circle_padding = circle_radius * 0.3
circle_total_width = (circle_radius * 2) + circle_padding
side_padding = (WIDTH - (7 * circle_total_width)) / 2
COL_PX_COORDS = [int(side_padding + circle_total_width * (0.5 + i)) for i in range(7)]

circle_padding_height = circle_radius * 0.3
circle_total_height = (circle_radius * 2) + circle_padding_height
top_padding = (HEIGHT * 0.85 - 6 * circle_total_height) / 2
ROW_PX_COORDS = [int(HEIGHT * .15 + top_padding + circle_total_height * (0.5 + i)) for i in range(6)]

def draw_board(screen, board):
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(0, 0, WIDTH, HEIGHT * .15))
    pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(0, HEIGHT * .15, WIDTH, HEIGHT * 0.85))
    for i in range(7):
        for j in range(6):
            if board[j, i] == 0:
                # empty circles should be slightly larger (easier on the eyes)
                pygame.draw.circle(screen, (255, 255, 255), (int(COL_PX_COORDS[i]), int(ROW_PX_COORDS[j])), circle_radius * 1.1)
            else:
                pygame.draw.circle(screen, 
                                (255, 0, 0) if board[j, i] == 1 else (255, 255, 0),
                                (int(COL_PX_COORDS[i]), int(ROW_PX_COORDS[j])),
                                circle_radius)

def draw_active_piece(screen, column, player):
    pygame.draw.circle(screen, 
                       (255, 0, 0) if player == 1 else (255, 255, 0),
                       (int(COL_PX_COORDS[column]), int(HEIGHT * .075)),
                       circle_radius)

if __name__ == "__main__":
    board_env = Connect(ROWS, COLUMNS)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect Four")

    p1 = None
    p2 = None

    # print("Welcome to Connect Four! Player 1 is red and Player 2 is yellow.")

    # valid_input = False
    # while not valid_input:
    #     p1 = input("Player 1 human or bot? (h/b):")
    #     if p1 == 'h':
    #         valid_input = True
    #     elif p1 == 'b':
    #         valid_input = True
    #     else:
    #         print("Invalid input. Please enter 'h' or 'b'")
    #         valid_input = False
    #         continue

    # valid_input = False
    # while not valid_input:
    #     p2 = input("Player 2 human or bot? (h/b):")
    #     if p1 == 'h':
    #         valid_input = True
    #     elif p1 == 'b':
    #         valid_input = True
    #     else:
    #         print("Invalid input. Please enter 'h' or 'b'")
    #         valid_input = False
    #         continue

    active_column = COLUMNS // 2
    active_player = 1

    while True:
        draw_board(screen, board)
        draw_active_piece(screen, active_column, active_player)

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key == K_LEFT:
                    active_column = max(0, active_column - 1)  # Decrement the column, but don't go below 0
                elif event.key == K_RIGHT:
                    active_column = min(6, active_column + 1)  # Increment the column, but don't go above 6
                elif event.key == K_SPACE:
                    print("space pressed")
                    active_player *= -1
        FRAMEPERSEC.tick(FPS)