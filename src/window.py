import time

import pygame as pg
from pygame.draw import circle

import numpy as np 
from numpy.linalg import norm
from numba import njit, prange
from numba.typed import List as Lst
from scipy.spatial import cKDTree

from physics import Physics
from colors import *

from slider_field import *
from gradient import *
from objects import *
from force_objects import *


class Window:
    def __init__(self, width, height, fps):
        self.width = width
        self.height = height
        self.fps = fps

        self.screen = pg.display.set_mode((self.width, self.height)) #, pg.OPENGL)
        # pg.display.gl_set_attribute(pg.GL_ACCELERATED_VISUAL, 1)

        pg.display.set_caption("Particle Simulation")
        self.clock = pg.time.Clock()
        self.running = True
        self.playing = True
        self.nextFrame = False

        self.x1 = 0
        self.x2 = self.width - 1
        self.y1 = 0
        self.y2 = self.height - 1
        self.move_left = False
        self.move_right = False
        self.move_up = False
        self.move_down = False

    @staticmethod
    def initProps():
        global sliderNumParticles, sliderRadius, sliderParticleSpacing, sliderSpatialRadius, sliderViscosity, sliderSmoothingRadius, sliderGravityX, sliderGravityY, teleporter1, mouseForce
        
        sliderNumParticles = SerializeField(
            0,0, "Particles: ", (1, 5_000), 4
        )

        sliderRadius = SerializeField(
            0, 30, "Radius: ", (1, 25), 5
        )

        sliderParticleSpacing = SerializeField(
            0, 60, "Particle spacing: ", (0.1, 15), 2
        )

        sliderSpatialRadius = SerializeField(
            0, 90, "Spatial radius: ", (1, 40), 20
        )

        sliderViscosity = SerializeField(
            SerializeField.winSize[0] * 3 // 4, 0, 'Viscosity: ', (0.0, 1.0), 1
        )

        sliderSmoothingRadius = SerializeField(
            SerializeField.winSize[0] * 3 // 4, 30, 'Smoothing radius: ', (0, 25), 15
        )

        sliderGravityX = SerializeField(
            SerializeField.winSize[0] * 3 // 4, 60, 'Gravity x: ', (-100, 100), 0
        )

        sliderGravityY = SerializeField(
            SerializeField.winSize[0] * 3 // 4, 90, 'Gravity y: ', (-100, 100), 0
        )

        # pointForce1 = PointForce(
        #     pos=np.array([400, 300]),
        #     forceRadius=300,
        #     forceFunction=lambda x: 90 * np.sin(0.14*x)
        # )

        teleporter1 = Teleporter(
            pos=np.array([400, 600]),
            teleportPos=np.array([400,0]),
            radius=40
        )
        teleporter1.active = False

        # rectangle1 = Rectangle(
        #     posWH=np.array([600, 450, 150, 300]),
        #     repulsionBorderWidth=5
        # )

        mouseForce = MouseForce(
            forceFunction=None
        )

    def processEvents(self):
        for event in pg.event.get():
            match event.type:
                case pg.QUIT:
                    self.running = False

                case pg.KEYDOWN:
                    match event.key:
                        case pg.K_LEFT:
                            self.move_left = True
                        case pg.K_RIGHT:
                            self.move_right = True
                        case pg.K_UP:
                            self.move_up = True
                        case pg.K_DOWN:
                            self.move_down = True

                        case pg.K_r:
                            Physics.reset()

                        case pg.K_SPACE:
                            self.playing = not self.playing

                        case pg.K_f:
                            self.nextFrame = not self.nextFrame
                            self.playing = True
                
                case pg.KEYUP:
                    match event.key:
                        case pg.K_LEFT:
                            self.move_left = False
                        case pg.K_RIGHT:
                            self.move_right = False
                        case pg.K_UP:
                            self.move_up = False
                        case pg.K_DOWN:
                            self.move_down = False

                case pg.MOUSEBUTTONDOWN:
                    match event.button:
                        case 1: # lmb
                            mouseForce.forceDir = 1
                            mouseForce.active = True
                        case 3:
                            mouseForce.forceDir = -1
                            mouseForce.active = True

                        case 4: # mw up
                            mouseForce.forceRadius += 1
                        case 5:
                            mouseForce.forceRadius -= 1


                case pg.MOUSEBUTTONUP:
                    match event.button:
                        case 1:
                            mouseForce.active = False
                        case 3:
                            mouseForce.active = False

                case pg.USEREVENT:
                    if event.user_type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                        
                        # update field value
                        SerializeField.update()
            
                        element = event.ui_element

                        match element:
                            case sliderNumParticles.slider:
                                Physics.numParticles = sliderNumParticles.value
                                Physics.reset()
                            case sliderParticleSpacing.slider:
                                Physics.particleSpacing = sliderParticleSpacing.value
                                Physics.arrangeParticles()

                        Physics.spatialRadius = sliderSpatialRadius.value

                        Physics.viscosity = sliderViscosity.value
                        Physics.radius = sliderRadius.value
                        Physics.gravity[0] = sliderGravityX.value

                        Physics.gravity[1] = sliderGravityY.value

                            
                        Physics.smoothingRadius = sliderSmoothingRadius.value
                
            SerializeField.manager.process_events(event)

        if self.move_left:
            self.x1 += 1
            self.x2 -= 1
        if self.move_right:
            self.x1 -= 1
            self.x2 += 1
        if self.move_up:
            self.y1 += 1
            self.y2 -= 1
        if self.move_down:
            self.y1 -= 1
            self.y2 += 1

    def drawBorders(self):
        # käytä vaan rect mitä ohjaillaan laittamalla
        # height +- 1 ja width +- 1 ni 
        rect = pg.Rect(self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)
        pg.draw.rect(self.screen, (230,230,230), rect, 1)

    def drawGrid(self):
        maxWidth = self.width+Physics.gridSquareSize
        maxHeight = self.height+Physics.gridSquareSize

        for i in range(0, maxWidth, Physics.gridSquareSize):
            pg.draw.line(self.screen, Physics.gridColor, (i, 0), (i, self.height))
            
            if i <= maxHeight:
                pg.draw.line(self.screen, Physics.gridColor, (0, i), (self.width, i))


    def start(self):
        Window.initProps()
        Physics.arrangeParticles()


    def update(self):
        start = time.perf_counter()
        if self.nextFrame:
            self.playing = False
            self.nextFrame = False

        self.processEvents()
        mouseForce.update()
        self.dt = self.clock.tick(self.fps) / 1000.0

        self.screen.fill(BLACK)
        self.drawGrid()

        SerializeField.manager.update(self.dt)
        SerializeField.draw(self.screen)

        self.simulation()

        renderStart = time.perf_counter()
        mouseForce.draw(self.screen)
        self.drawParticles()
        self.drawBorders()

        # Rectangle.draw(self.screen)
        PointForce.draw(self.screen)
        
        pg.display.flip()
        renderEnd = time.perf_counter()
        renderTime = renderEnd-renderStart

        timeElapsed = time.perf_counter() - start

        # force smooth fix ??
        Physics.timestep = 1 / (self.fps * timeElapsed)




    def simulation(self):
        if self.playing:
            # performing calculations with "predicted positoins"
            # helps particles to settle down

            Physics.predictedPositions = Physics.positions + Physics.velocities * self.dt * Physics.timestep

            # startTree = time.perf_counter()

            # TODO: OPTIMIZE
            Physics.tree = cKDTree(Physics.predictedPositions)
            # print(f"TREE: {time.perf_counter() - startTree}")

            forceObjectNeighbours = Physics.tree.query_ball_point(
                x = [obj.pos for obj in forceObjects],
                r = [obj.forceRadius for obj in forceObjects]
            )

            Physics.calculateObjectForces(
                Physics.velocities,
                Physics.predictedPositions,
                forceObjectNeighbours,
                forceObjects,
                self.dt,
                Physics.timestep
            )

            teleporterNeighbours = Physics.tree.query_ball_point(
                x = [tel.pos for tel in Teleporter.teleporters],
                r = [tel.radius for tel in Teleporter.teleporters]
            )

            for i, tel in enumerate(Teleporter.teleporters):
                tel.teleport(teleporterNeighbours[i], Physics.positions)

            """
       workers | time / 5 000 queries
            1  | 61.38 ms
            4  | 38.31 ms
            6  | 39.13 ms
            7  | 30.92 ms
            7  | 32.03 ms
            8  | 35.45 ms
            8  | 37.89 ms
            16 | 50.82 ms
            -1 | 34.04 ms
            """

            neighborsArray = Physics.tree.query_ball_point(
                                Physics.predictedPositions,
                                Physics.spatialRadius,
                                workers=7
                            )
            neighborsArray = Lst(np.array(arr) for arr in neighborsArray)

            start = time.perf_counter()
            Physics.calculateForces(
                Physics.addedVelocities,
                Physics.predictedPositions,
                Physics.velocities,
                Physics.positions,
                neighborsArray, 
                Physics.numParticles,
                Physics.smoothingRadius,
                Physics.viscosity,
                Physics.gravity,
                self.dt,
                Physics.timestep
            )
            end = time.perf_counter()

            text_surface = SerializeField.font.render(f"Frame time: {round(end-start, 4)} s", True, WHITE)
            
            self.screen.blit(text_surface, (15, 120))

            # Physics.velocities +=  Physics.timestep * self.dt * (Physics.addedVelocities * Physics.viscosity + Physics.gravity)
            # Physics.positions +=   Physics.timestep * self.dt * Physics.velocities

            # Rectangle.handleCollisions(Physics.positions, Physics.velocities)

            Physics.borderCollisions(
                Physics.positions,
                Physics.velocities,
                Physics.radius,
                Physics.bounciness,
                self.x1, self.x2, self.y1, self.y2
            )

    def drawParticles(self):
        norms: np.ndarray = norm(Physics.velocities, axis=1)
        colorIDs = np.minimum((norms * scaleFactor).astype(int), gradientLen - 1)

        # pygame.draw.circle would have to truncate the 
        # position coordinates which causes overhead
        intPositions = Physics.positions.astype(int)
        
        # precompute colors?  
        for i in range(Physics.numParticles):
            circle(
                self.screen,
                gradientColors[colorIDs[i]],
                intPositions[i],
                Physics.radius
            )