import numpy as np
from numba import njit, prange, int32
from numba import typed
from numba import types

import time

# @njit(parallel=True) ei nopeuta :(
# def _norm(arr: np.ndarray) -> np.ndarray: 
#     m = arr.shape[0]
#     norms = np.empty((m,), dtype=float)
#     for i in prange(m):
#         norms[i] = np.sqrt(arr[i, 0] * arr[i, 0] + arr[i, 1] * arr[i, 1])

#     return norms



keyType = types.UniTuple(types.int32, 2)
intArray = types.int32[:]

class Physics:
    # note: v = FΔt/m, x = vΔt
    x = 0
    y = 1

    WIDTH, HEIGHT = 800, 600

    numParticles = 4
    positions = np.zeros((numParticles, 2))
    predictedPositions = np.zeros((numParticles, 2))
    velocities = np.zeros((numParticles, 2))
    addedVelocities = np.zeros_like(positions)

    tree = ...
    spatialRadius = 15

    gravity = np.array([0, 0])
    smoothingRadius = 15

    # SR 15 917 PARTICLES GRAVITY 41

    radius = 5
    viscosity = 1
    bounciness = 0.5
    timestep = 1 # 0.14
    maxForce = 800 # 400

    # for arranging particles
    particleSpacing = 2

    # background grid size
    gridSquareSize = 50
    gridColor = (50,50,50)

    mouseForceColor = (100,100,128)
    mouseForceRadius = 75



    # | 10 000 PARTICLES |
    # w/o parallel : 130 ms
    #   w parallel : 40 ms
    # + boundcheck : 40 ms
    # + no gil     : 40 ms
    @njit(cache=True, parallel=True)#, nogil=True)#, boundscheck=False)
    def calculateForces(
        addedVelocities: np.ndarray,
        predictedPositions: np.ndarray,
        neighborsArray: np.ndarray,
        numParticles: int,
        maxForce: int,
        smoothingRadius: int,
        gridSquareSize: np.ndarray
        ):

        addedVelocities.fill(0)

        # NOTE: np.concatenate 
        # assign grid index by position
        # grid2D = predictedPositions.astype(np.int64) // gridSquareSize

        # convert to 1d grid
        # grid1D = grid2D[:,1] * 16 + grid2D[:,0]
        
        # particleGroups = [typed.List.empty_list(np.int64) for _ in range(192)]
        # particleGroups = typed.List(particleGroups)

        # for i in prange(numParticles):
            # particle i's grid index
            # particleGroup: np.int64 = grid1D[i]
            # particleGroups[particleGroup].append(i)


        for i in prange(numParticles):
            # gridPos2D = predictedPositions[i].astype(np.int64) // gridSquareSize
            # groupID = gridPos2D[1] * 16 + gridPos2D[0]
            # print(groupID)
            neighbors = neighborsArray[i]
            if neighbors.size > 1:
            # if neighbors:
                distVectors = predictedPositions[i] - predictedPositions[neighbors]
                dists = np.sqrt(np.sum(distVectors**2, axis=1))
                distsNonzero = dists > 0
                
                forces = np.minimum(
                    maxForce, (smoothingRadius - dists[distsNonzero]) ** 3
                )
                # paranna
                forces, distVectors, dists = np.broadcast_arrays(forces[:, np.newaxis], distVectors[distsNonzero], dists[distsNonzero][:, np.newaxis])
                forceVectors = forces * distVectors / dists

                # forceVectors = forces[:, np.newaxis] * distVectors[distsNonzero] / dists[distsNonzero][:, np.newaxis]
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
                        (smoothingRadius -2 - dists) ** 3 #[dists_nonzero]
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
        particlesPerRow = int(np.sqrt(Physics.numParticles))
        particlesPerCol = (Physics.numParticles - 1) // particlesPerRow + 1
        spacing = Physics.radius * 2 + Physics.particleSpacing

        offsetX = 400 - particlesPerRow // 2 
        offsetY = 300 - particlesPerCol // 2

        for i in range(Physics.numParticles):
            xPos = offsetX + (i % particlesPerRow - particlesPerRow // 2 + 0.5) * spacing
            yPos = offsetY + (i // particlesPerRow - particlesPerCol // 2 + 0.5) * spacing
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
    @njit(cache=True)
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