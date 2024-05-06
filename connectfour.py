import pygame
import sys
import numpy as np
from aiconnectfour import Bot, Connect
from pygame.locals import *
 
HEIGHT = 450
WIDTH = 400
FPS = 60
FRAMEPERSEC = pygame.time.Clock()

BOARD_WIDTH = 6
BOARD_HEIGHT = 7

circle_radius = HEIGHT * .05

circle_padding_height = circle_radius * 0.3
circle_total_height = (circle_radius * 2) + circle_padding_height
board_padding = (HEIGHT * 0.85 - BOARD_HEIGHT * circle_total_height) / 2
ROW_PX_COORDS = [int(HEIGHT - (board_padding + circle_total_height * (0.5 + i))) for i in range(BOARD_HEIGHT)]

circle_padding_width = circle_radius * 0.3
circle_total_width = (circle_radius * 2) + circle_padding_width
side_padding = (WIDTH - BOARD_WIDTH * circle_total_width) / 2
COL_PX_COORDS = [int(side_padding + circle_total_width * (0.5 + i)) for i in range(BOARD_WIDTH)]

def draw_board(screen, board):
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(0, 0, WIDTH, HEIGHT * .15))
    pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(0, HEIGHT * .15, WIDTH, HEIGHT * 0.85))
    for w in range(BOARD_WIDTH):
        for h in range(BOARD_HEIGHT):
            if board[w, h] == 0:
                pygame.draw.circle(screen, (255, 255, 255), (COL_PX_COORDS[w], ROW_PX_COORDS[h]), circle_radius * 1.1)
            else:
                pygame.draw.circle(screen, 
                                   (255, 0, 0) if board[w, h] == 1 else (255, 255, 0),
                                   (COL_PX_COORDS[w], ROW_PX_COORDS[h]),
                                   circle_radius)


def draw_active_piece(screen, column, player):
    pygame.draw.circle(screen, 
                       (255, 0, 0) if player == 1 else (255, 255, 0),
                       (COL_PX_COORDS[column], WIDTH * .075),
                       circle_radius)

if __name__ == "__main__":
    board_env = Connect(width=BOARD_WIDTH, height=BOARD_HEIGHT)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect Four")

    # main loop
    while True:
        print("cool")
        board_env.reset()
        in_game = False
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

        in_game = True

        active_drop = BOARD_WIDTH // 2
        active_player = 1

        # game loop
        while in_game:
            draw_board(screen, board_env._get_obs())
            draw_active_piece(screen, active_drop, active_player)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN:
                    if event.key == K_LEFT:
                        active_drop = max(0, active_drop - 1)  # Decrement the column, but don't go below 0
                    elif event.key == K_RIGHT:
                        active_drop = min(BOARD_WIDTH - 1, active_drop + 1)  # Increment the column, but don't go above 6
                    elif event.key == K_SPACE:
                        col = board_env._get_obs()[active_drop]
                        if col[-1] == 0:
                            ind = np.argmin(np.abs(col))
                            board_env.manual_adjust_grid((active_drop, ind), active_player)
                            if board_env.determine_win(board_env._get_obs(), (active_drop, ind)) != 0:
                                # one more draw to show the winning move
                                draw_board(screen, board_env._get_obs())
                                pygame.display.flip()

                                if active_player == -1:
                                    active_player = 2
                                print("Player", active_player, "wins!")

                                in_game = False
                            else:
                                active_player = -active_player
                        # if a bot decides to make an illegal move for whatever reason just resample

                FRAMEPERSEC.tick(FPS)

        valid_input = False
        while not valid_input:
            response = input("Do you wish to play again? (y/n):")
            if response == 'y':
                valid_input = True
                print("restarting...")
            elif response == 'n':
                pygame.quit()
                sys.exit()
            else:
                print("Invalid input. Please enter 'y' or 'n'")
                valid_input = False
                continue
