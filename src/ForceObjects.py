from pygame.draw import circle
from pygame.mouse import get_pos
from numpy import array

forceObjects = []

def drawForceObjects():
    for obj in forceObjects:
        obj.draw()

class MouseForce:
    color = (100,100,128)

    def __init__(
            self,
            forceRadius: float = 75,
            forceFunction=None
            ) -> None:

        self.forceRadius = forceRadius
        self.forceDir = 1
        self.active = False

        if forceFunction:
            self.forceFunction = forceFunction
        else:
            self.forceFunction = lambda x: (2 * x + 90) * self.forceDir

        forceObjects.append(self)

    def update(self):
        mx, my = get_pos()
        self.xy = (mx, my)
        self.pos = array([mx, my])

    def draw(self, screen):
        if self.active:
            circle(screen, self.color, self.xy, self.forceRadius, 2)


class PointForce:
    color=(255,255,255)
    points=[]
    def __init__(
            self,
            pos,
            forceRadius: float,
            forceFunction=None,
            radius: float=8,
            strength: float=1
            ) -> None:
        
        self.pos = pos
        self.xy = (pos[0], pos[1])
        self.forceRadius = forceRadius
        self.radius = radius
        self.strenght = strength
        self.active = True


        if forceFunction:
            self.forceFunction = forceFunction
        else:
            self.forceFunction = lambda x: 1/(self.strenght * x ** 2) if x > 0 else 0

        PointForce.points.append(self)
        forceObjects.append(self)


    @classmethod
    def draw(cls, screen):
        for point in cls.points:
            circle(screen, cls.color, point.xy, point.radius)

class ForceField:
    ...
