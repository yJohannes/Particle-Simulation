from pygame.draw import circle

class PointForce:
    color=(255,255,255)
    points=[]
    def __init__(
            self,
            x: float,
            y: float,
            forceRadius: float,
            radius: float=8,
            strength: float=1,
            forceFunction=None
            ) -> None:
        
        self.x = x
        self.y = y
        self.forceRadius = forceRadius
        self.radius = radius
        self.strenght = strength

        if forceFunction:
            self.forceFunction = forceFunction
        else:
            self.forceFunction = lambda x: 1/(self.strenght * x ** 2) if x > 0 else 0

        PointForce.points.append(self)

    @classmethod
    def draw(cls, screen):
        for field in cls.points:
            circle(screen, cls.color, (field.x, field.y), field.radius)

class ForceField:
    ...