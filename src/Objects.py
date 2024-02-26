from pygame.draw import circle, rect
from pygame import Rect
import numpy as np


# use numpy operatoins to find indices within rectangle
# apply force in ??? dir
class Rectangle:
    color = (100,100,100)
    rectangles = []

    def __init__(
            self,
            posWH: np.ndarray,
            repulsionBorderWidth: float
            ) -> None:
        # nää koordinaatit ja leveydet on väärin
        self.repulsionBorderWidth = repulsionBorderWidth
        self.pos = posWH[:2]

        self.left = posWH[0] - posWH[2] // 2
        self.right = posWH[0] + posWH[2] // 2

        self.top = posWH[1] + posWH[3] // 2
        self.bottom = posWH[1] - posWH[3] // 2

        self.widthheight = posWH[2:4]
        self.width, self.height = self.widthheight



        self.lefttop = self.pos - self.widthheight // 2
        self.rect = Rect(self.lefttop, self.widthheight)
        self.repulsionRect = Rect(
            self.lefttop - repulsionBorderWidth,
            2 * repulsionBorderWidth + self.widthheight
        )

        Rectangle.rectangles.append(self)

    @classmethod
    def handleCollisions(cls, positions, velocities):
        for rectangle in cls.rectangles:
            inside = np.logical_and.reduce((
                positions[:, 0] > rectangle.left,
                positions[:, 0] < rectangle.right,
                positions[:, 1] > rectangle.bottom,
                positions[:, 1] < rectangle.top
            ))

            if inside.size > 0:
                
                dx = np.maximum(0, np.maximum(rectangle.left - positions[:, 0], positions[:, 0] - rectangle.right))
                dy = np.maximum(0, np.maximum(rectangle.bottom - positions[:, 1], positions[:, 1] - rectangle.top))

                dists = np.sqrt(dx*dx + dy*dy)
            
                print(dists)



    @classmethod
    def draw(cls, screen):
        for rectangle in cls.rectangles:
            rect(screen, cls.color, rectangle.rect, 1)


class Circle:
    color = (50,50,50)
    circles = []

    def __init__(
            self,
            pos,
            ) -> None:
        pass


class Pipe:
    ...

class Teleporter:
    color=(77, 76, 107)

    teleporters = []
    def __init__(
            self,
            pos,
            teleportPos,
            radius: float
            ) -> None:
        
        self.active = True
        self.pos = pos
        self.xy = (pos[0], pos[1])
        self.radius = radius
        self.teleportPos = teleportPos

        Teleporter.teleporters.append(self)


    def teleport(
            self,
            indices,
            positions,
            velocities=None,
            resetVelocity=True
            ):
        
        if self.active:
            positions[indices] = self.teleportPos

            if resetVelocity and velocities:
                # add variety, the particles get stuck on top of each other
                velocities[indices] = 0

    @classmethod
    def draw(cls, screen):
        for teleporter in cls.teleporters:
            circle(screen, cls.color, teleporter.xy, teleporter.radius)