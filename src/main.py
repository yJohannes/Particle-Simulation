import sys
import pygame as pg

from window import Window

# TODO:
# solve the best settings for smoothing radius & others
# best was sr = 25 with current (soon to be old ones)

if __name__ == '__main__':
    pg.init()
    WIDTH, HEIGHT = 800, 600
    FPS = 60

    window = Window(WIDTH, HEIGHT, FPS)
    window.start()

    while window.running:
        window.update()

    pg.quit()
    sys.exit()