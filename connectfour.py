import pygame
import sys
from aiconnectfour import Bot, Connect
from pygame.locals import *
 
pygame.init()
vec = pygame.math.Vector2  # 2 for two dimensional
 
HEIGHT = 450
WIDTH = 400
FPS = 60
 
FramePerSec = pygame.time.Clock()
 
displaysurface = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Game")

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__() 
        self.surf = pygame.Surface((30, 30))
        self.surf.fill((128,255,40))
        self.rect = self.surf.get_rect(center = (10, 420))

        # todo, move this out somewhere
        self.bot = Bot()
        self.env = Connect()
        self.obs, self.info = self.env.reset()
        color = 1
        
        # do this every time 
        move = self.bot.get_move(self.obs)
        obs, reward, terminated, _, _ = self.env.step(move, color)
 
P1 = Player()

all_sprites = pygame.sprite.Group()
all_sprites.add(P1)
 

# Main loop at 60hz 
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
     
    displaysurface.fill((0,0,0))
 
    for entity in all_sprites:
        displaysurface.blit(entity.surf, entity.rect)
 
    pygame.display.update()
    FramePerSec.tick(FPS)
