import sys
import pygame as pg
from pygame.draw import circle

import numpy as np 
from numpy.linalg import norm

from numba import jit
from numba.typed import List as Lst

import time
from scipy.spatial import cKDTree


from colors import *
from physics import Physics
from SerializeField import *
from gradient import *

class Window:
    def __init__(self):
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption("Particle Simulation")
        self.clock = pg.time.Clock()
        self.running = True
        self.playing = True
        self.holding = False
        self.nextFrame = False
        self.mouseForceDir = 1

        self.x1 = 0
        self.x2 = WIDTH - 1
        self.y1 = 0
        self.y2 = HEIGHT - 1
        self.move_left = False
        self.move_right = False
        self.move_up = False
        self.move_down = False

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

                        # case pg.K_s:
                        #     self.playing = not self.playing

                        # case pg.K_p:
                        #     self.playing = not self.playing
                        
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
                    if event.button == 3:
                        self.mouseForceDir *= -1
                    self.holding = True

                case pg.MOUSEBUTTONUP:
                    self.holding = False

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
            

                        Physics.viscosity = sliderViscosity.value
                        Physics.radius = sliderRadius.value
                        Physics.attractionForce = sliderAttractionForce.value
                        Physics.gravity[1] = sliderGravity.value
                            
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
        pg.draw.rect(self.screen, (230,230,230), rect, 1) # (100, 255, 100)

    def drawGrid(self):
        maxWidth = WIDTH+Physics.gridSize
        maxHeight = HEIGHT+Physics.gridSize

        # for i_x in range(0, maxWidth, Physics.gridSize):
        #     pg.draw.line(self.screen, Physics.gridColor, (i_x, 0), (i_x, HEIGHT))

        # for i_y in range(0, maxHeight, Physics.gridSize):
        #     pg.draw.line(self.screen, Physics.gridColor, (0, i_y), (WIDTH, i_y))

        for i in range(0, maxWidth, Physics.gridSize):
            pg.draw.line(self.screen, Physics.gridColor, (i, 0), (i, HEIGHT))
            
            if i <= maxWidth:
                pg.draw.line(self.screen, Physics.gridColor, (0, i), (WIDTH, i))


    def start(self):
        Physics.arrangeParticles()


    def update(self):
        if self.nextFrame:
            self.playing = False
            self.nextFrame = False

        self.processEvents()
        self.screen.fill(BLACK)
        self.drawGrid()

        self.dt = self.clock.tick(FPS) / 1000.0
        SerializeField.manager.update(self.dt)
        SerializeField.draw(self.screen)
        

        s =time.time()
        self.drawParticles()
        self.simulation()
        self.drawBorders()
        # print(time.time()-s)

        pg.display.flip()


    def simulation(self):
        if self.playing:
            Physics.predictedPositions = Physics.positions + Physics.velocities * self.dt * Physics.timestep
            Physics.tree = cKDTree(Physics.predictedPositions)

            if self.holding:
                mx, my = pg.mouse.get_pos()
                circle(self.screen, (100,100,128), (mx, my), Physics.mouseForceRadius, 2)

                # hitaampi luoda mut nopeempi laskee
                cursorPos = np.array([mx, my])
                # cursorPos = [mx, my]

                nearIndices = Physics.tree.query_ball_point(
                    x=cursorPos,
                    r=Physics.mouseForceRadius
                    )

                if nearIndices:
                    forceVectors = np.zeros_like(nearIndices)
                    distVectors = Physics.predictedPositions[nearIndices] - cursorPos
                    dists = norm(distVectors, axis=1)
                    distsNonzero = np.nonzero(dists)
                    forces = 2 * dists + 6 * Physics.smoothingRadius
                    distUnitVectors = distVectors[distsNonzero] / dists[distsNonzero][:, np.newaxis]
                    forceVectors = distUnitVectors * forces[:, np.newaxis]

                    Physics.velocities[nearIndices] += self.mouseForceDir * forceVectors * self.dt

            # SAIRASTA
                    
            # neighborsArray = [
            #     np.array(Physics.tree.query_ball_point(x=Physics.predictedPositions[i], r=15))
            #     for i in range(Physics.numParticles)
            # ]

            neighborsArray = Lst(
                np.array(Physics.tree.query_ball_point(x=Physics.predictedPositions[i], r=15))
                for i in range(Physics.numParticles)
            )
            
            Physics.calculateForces(
                Physics.addedVelocities,
                Physics.predictedPositions,
                neighborsArray, 
                Physics.numParticles,
                Physics.maxForce,
                Physics.smoothingRadius
            )

            # neighborsArrayOG = [
            #     Physics.tree.query_ball_point(x=Physics.predictedPositions[i], r=25)
            #     for i in range(Physics.numParticles)
            # ]

            # Physics.calculateForcesOG(
            #     Physics.addedVelocities,
            #     Physics.predictedPositions,
            #     neighborsArray, 
            #     Physics.numParticles,
            #     Physics.maxForce,
            #     Physics.smoothingRadius
            #     )

            # Physics.velocities += Physics.addedVelocities * self.dt * Physics.viscosity
            # Physics.velocities += Physics.gravity * self.dt
            Physics.velocities +=  Physics.timestep * self.dt * (Physics.addedVelocities * Physics.viscosity + Physics.gravity)
            Physics.positions +=   Physics.timestep * self.dt * Physics.velocities

            # moved up here
            Physics.borderCollisions(
                Physics.positions,
                Physics.velocities,
                Physics.radius,
                Physics.bounciness,
                self.x1, self.x2, self.y1, self.y2
            )
            
    def drawParticles(self):
        norms = norm(Physics.velocities, axis=1)
        colorIDs = np.minimum((norms * scaleFactor).astype(int), gradientLen - 1)

        for i in range(Physics.numParticles):
            circle(
                self.screen,
                gradientColors[colorIDs[i]],
                Physics.positions[i],
                Physics.radius
                )

if __name__ == '__main__':
    pg.init()
    WIDTH, HEIGHT = 800, 600
    FPS = 60

    sliderNumParticles = SerializeField(
        0,0, "Particles: ", (1, 2500), 4
        )
    
    sliderRadius = SerializeField(
        0, 30, "Radius: ", (1, 50), 5
        )
    
    sliderParticleSpacing = SerializeField(
        0, 60, "Particle spacing: ", (1, 15), 2
    )
    

    sliderAttractionForce = SerializeField(
        WIDTH-SerializeField.winSize[0]//4, 0, "Attraction force: ", (0, 1), 0.4) 
    
    sliderViscosity = SerializeField(
        WIDTH-SerializeField.winSize[0]//4, 30, 'Viscosity: ', (0.00001, 1), 1
    )

    sliderSmoothingRadius = SerializeField(
        WIDTH-SerializeField.winSize[0]//4, 60, 'Smoothing radius: ', (5, 100), 20
    )

    sliderGravity = SerializeField(
        WIDTH-SerializeField.winSize[0]//4, 90, 'Gravity: ', (-100, 100), 0
        )

    window = Window()

    window.start()
    while window.running:
        s = time.time()
        window.update()
        print(time.time()-s)

    pg.quit()
    sys.exit()