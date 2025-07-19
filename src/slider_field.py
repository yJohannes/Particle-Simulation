import pygame_gui
from pygame_gui.elements import UIHorizontalSlider

import pygame as pg

from pygame import font

pg.init()


class SerializeField:
    WHITE = (255,255,255)
    font = font.Font(None, 26)

    winSize = (800, 600)
    manager = pygame_gui.UIManager(winSize)

    fields = []


    def __init__(
            self,
            x,
            y,
            text, 
            range: tuple,
            startValue,
            width=winSize[0]//4,
            height=15,
            winSize=False
            ):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.baseText = text
        self.text = text
        self.range = range
        self.startValue = startValue
        self.value = startValue

        if winSize:
            SerializeField.winSize = winSize

        self.textSurface = self.font.render(self.text, True, self.WHITE)        
        self.textRect = self.textSurface.get_rect()
        self.textRect.center = (self.x + self.width / 2, self.y + self.height * 3/2)

        self.slider = UIHorizontalSlider(
            relative_rect=pg.Rect((self.x, self.y), (self.width, self.height)),
            start_value=self.startValue,  # Initial value of the slider
            value_range=self.range,  # Minimum and maximum values of the slider
            manager=SerializeField.manager
        )
        
        SerializeField.fields.append(self)

    def update():
        for field in SerializeField.fields:
            field.value = round(field.slider.current_value, 2)
            field.text = field.baseText + str(field.value)
            field.textSurface = field.font.render(field.text, True, field.WHITE)
            field.textRect = field.textSurface.get_rect()
            field.textRect.center = (field.x + field.width / 2, field.y + field.height * 3/2)


    def draw(screen):
        for field in SerializeField.fields:
            SerializeField.manager.draw_ui(screen)
            screen.blit(field.textSurface, field.textRect)