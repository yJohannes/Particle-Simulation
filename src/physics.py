import math
import numpy as np
from numba import jit, njit
from random import uniform

import time

class Physics:
    # note: v = FΔt/m, x = vΔt
    x = 0
    y = 1

    numParticles = 4
    positions = np.zeros((numParticles, 2))
    predictedPositions = np.zeros((numParticles, 2))
    velocities = np.zeros((numParticles, 2))
    addedVelocities = np.zeros_like(positions)

    tree = ...
    spatialRadius = 15

    gravity = np.array([0, 0])
    attractionRadius = 2
    attractionForce = 0.4
    smoothingRadius = 15
    # SR 15 917 PARTICLES GRAVITY 41
    mouseForceRadius = 75

    radius = 5
    viscosity = 0.5
    bounciness = 0.5
    timestep = 1 # 0.14
    epsilon = 3 #10
    maxForce = 800 # 400

    # for arranging particles
    particleSpacing = 2

    # background grid size
    gridSize = 50
    gridColor = (50,50,50)
    
    # parallel=True JOSKUS EHKÄ
    # @jit(nopython=True)
    @njit()
    def calculateForces(
        addedVelocities: np.ndarray,
        predictedPositions: np.ndarray,
        neighborsArray: np.ndarray,
        numParticles: int,
        maxForce: int,
        smoothingRadius: int
        ):
        
        addedVelocities[:] = 0 
        # laita rangeNumParticles
        for i in range(numParticles):
            # neighbors = neighborsArray[i][neighborsArray[i] !=i]
            # if np.any(neighbors):
            # mielummin jos pituus yli 1 ehkä mut silti surkee
                neighbors = neighborsArray[i]
                distVectors = predictedPositions[i] - predictedPositions[neighbors]
                dists = np.sqrt(np.sum(distVectors**2, axis=1))
                distsNonzero = np.nonzero(dists)

                # laita dists nexaxis ehkä

                # vectorize  ei voi
                forces = np.minimum(
                    maxForce,
                    np.maximum(
                        -5,
                        (smoothingRadius - dists[distsNonzero]) ** 3
                        ) # - Physics.attractionForce
                        )

                # distUnitVectors = distVectors[distsNonzero] / dists[distsNonzero][:, np.newaxis]
                # forceVectors = forces[:, np.newaxis] * distUnitVectors
                forceVectors = forces[:, np.newaxis] * distVectors[distsNonzero] / dists[distsNonzero][:, np.newaxis]

                addedVelocities[i] = np.sum(forceVectors, axis=0)

    def calculateForcesOG(
        addedVelocities,
        predictedPositions,
        neighborsArray,
        numParticles,
        maxForce,
        smoothingRadius
        ):
        
        addedVelocities[:] = 0 

        for i in range(numParticles):
            neighbors = neighborsArray[i]
            # remove instance of self
            neighbors.remove(i)
            

            if neighbors:
                dist_vectors = predictedPositions[i] - predictedPositions[neighbors]
                dists = np.linalg.norm(dist_vectors, axis=1, keepdims=True) # enne ei keepdims
                dists_nonzero = np.nonzero(dists)
                force = np.minimum(
                    maxForce,
                    np.maximum(
                        -5,
                        (smoothingRadius - dists - 2) ** 3 #[dists_nonzero]
                        ) # - Physics.attractionForce
                        )
                
                # ENNEN
                
                    # np.maximum(
                    #     -5,
                    #     smoothingRadius - dists - 2
                    #     ) ** 3
                    #     
                
                # force_vectors = force[:, np.newaxis] * dist_vectors[dists_nonzero] / dists[dists_nonzero][:, np.newaxis]
                force_vectors = force * dist_vectors / dists

                addedVelocities[i] = np.sum(force_vectors[dists_nonzero[0]], axis=0)


    def start():
        # add stuff later
        Physics.arrangeParticles()


    def reset():
        # tee yks zeros() ja lopu [:] = 0
        Physics.positions = np.zeros((Physics.numParticles, 2))
        Physics.velocities = np.zeros((Physics.numParticles, 2))
        Physics.predictedPositions = np.zeros((Physics.numParticles, 2))
        Physics.addedVelocities = np.zeros_like(Physics.velocities)
        
        Physics.arrangeParticles()


    def arrangeParticles():
        particlesPerRow = int(math.sqrt(Physics.numParticles))
        particlesPerCol = (Physics.numParticles - 1) / particlesPerRow + 1
        spacing = Physics.radius * 2 + Physics.particleSpacing

        offsetX = 400 - particlesPerRow / 2 
        offsetY = 300 - particlesPerCol / 2

        for i in range(Physics.numParticles):
            xPos = offsetX + (i % particlesPerRow - particlesPerRow / 2 + 0.5) * spacing
            yPos = offsetY + (i / particlesPerRow - particlesPerCol / 2 + 0.5) * spacing
            Physics.positions[i] = np.array([xPos, yPos])

    def particleForce(dist):
        return max(0, Physics.smoothingRadius - dist) ** 3 - Physics.attractionForce

    def particleForces(dists):
        return np.minimum(Physics.maxForce, np.maximum(0, Physics.smoothingRadius - dists) ** 3 - Physics.attractionForce)

    def particleForces2(dists):
        return np.minimum(Physics.maxForce, np.maximum((Physics.smoothingRadius-dists)**3,-np.e ** (Physics.smoothingRadius-dists)-Physics.attractionForce))

    def LJpotential(dists):
        potential = 4*Physics.epsilon*((Physics.smoothingRadius/dists)**12-(Physics.smoothingRadius/dists)**6) - Physics.attractionForce
        return np.minimum(Physics.maxForce, potential)

    def LJforce(dists):
        force = 24*Physics.epsilon/dists*(2*(Physics.smoothingRadius/dists)**12-(Physics.smoothingRadius/dists)**6)
        return np.minimum(600, force)

    # TÄHÄN ON PAKKO OLLA PAREMPI TAPA; ESIM JOKU RECT
    @njit()
    def borderCollisions(positions, velocities, radius, bounciness, x1, x2, y1, y2):
        left_mask = positions[:, 0] < x1 + radius
        right_mask = positions[:, 0] > x2 - radius
        bottom_mask = positions[:, 1] < y1 + radius
        top_mask = positions[:, 1] > y2 - radius

        # Apply left and right border collisions
        velocities[left_mask | right_mask, 0] *= -bounciness
        positions[left_mask, 0] = x1 + radius
        positions[right_mask, 0] = x2 - radius

        # Apply bottom and top border collisions
        velocities[bottom_mask | top_mask, 1] *= -bounciness
        positions[bottom_mask, 1] = y1 + radius
        positions[top_mask, 1] = y2 - radius