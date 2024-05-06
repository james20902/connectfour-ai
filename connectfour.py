import pygame
import sys
import numpy as np
from aiconnectfour import Bot, MiniMaxBot, MCTSBot, Connect
from pygame.locals import *
 
HEIGHT = 450
WIDTH = 400
FPS = 60
FRAMEPERSEC = pygame.time.Clock()
BLANK_RATIO = 0.15
BOARD_RATIO = 0.85

BOARD_WIDTH = 7
BOARD_HEIGHT = 6

circle_radius = HEIGHT * .05

circle_padding_height = circle_radius * 0.3
circle_total_height = (circle_radius * 2) + circle_padding_height
board_padding = (HEIGHT * BOARD_RATIO - BOARD_HEIGHT * circle_total_height) / 2
ROW_PX_COORDS = [int(HEIGHT - (board_padding + circle_total_height * (0.5 + i))) for i in range(BOARD_HEIGHT)]

circle_padding_width = circle_radius * 0.3
circle_total_width = (circle_radius * 2) + circle_padding_width
side_padding = (WIDTH - BOARD_WIDTH * circle_total_width) / 2
COL_PX_COORDS = [int(side_padding + circle_total_width * (0.5 + i)) for i in range(BOARD_WIDTH)]

def draw_board(screen, board):
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(0, 0, WIDTH, HEIGHT * BLANK_RATIO))
    pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(0, HEIGHT * BLANK_RATIO, WIDTH, HEIGHT * BOARD_RATIO))
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
                       (COL_PX_COORDS[column], HEIGHT * (BLANK_RATIO / 2)),
                       circle_radius)

if __name__ == "__main__":
    board_env = Connect(width=BOARD_WIDTH, height=BOARD_HEIGHT)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect Four")

    b1 = MiniMaxBot("Player 1", 3)
    b2 = MiniMaxBot("Player 2", 3)

    # main loop
    while True:
        board_env.reset()
        in_game = False

        b1.instantiate(1, board_env.action_space, board_env)
        b2.instantiate(-1, board_env.action_space, board_env)

        p1 = None
        p2 = None

        print("Welcome to Connect Four! Player 1 is red and Player 2 is yellow.")

        valid_input = False
        while not valid_input:
            p1 = input("Player 1 human or bot? (h/b):")
            if p1 == 'h':
                valid_input = True
            elif p1 == 'b':
                valid_input = True
            else:
                print("Invalid input. Please enter 'h' or 'b'")
                valid_input = False
                continue

        valid_input = False
        while not valid_input:
            p2 = input("Player 2 human or bot? (h/b):")
            if p1 == 'h':
                valid_input = True
            elif p1 == 'b':
                valid_input = True
            else:
                print("Invalid input. Please enter 'h' or 'b'")
                valid_input = False
                continue

        in_game = True

        active_drop = BOARD_WIDTH // 2
        active_player = 1
        move_made = False

        # game loop
        while in_game:            

            draw_board(screen, board_env._get_obs())
            draw_active_piece(screen, active_drop, active_player)
            pygame.display.flip()

            if (p1 == 'b' and active_player == 1) or (p2 == 'b' and active_player == -1):
                active_bot = b1 if active_player == 1 else b2

                active_drop = active_bot.get_move(board_env._get_obs())
                col = board_env._get_obs()[active_drop]
                ind = np.argmin(np.abs(col))

                board_env.step(active_drop, active_player)
                move_made = True
            else:            
                # human input check
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
                                move_made = True

            if move_made:
                if board_env.determine_win(board_env._get_obs(), (active_drop, ind)) != 0:
                    # one more draw to show the winning move
                    draw_board(screen, board_env._get_obs())
                    pygame.display.flip()

                    if active_player == -1:
                        active_player = 2
                    print("Player", active_player, "wins!")

                    in_game = False
                else:
                    active_drop = BOARD_WIDTH // 2
                    active_player = -active_player
                    move_made = False

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
