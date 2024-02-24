import numpy as np
from numba import njit, prange, int32, int64
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
    # maxForce = 800 # 400

    # for arranging particles
    particleSpacing = 2

    # background grid size
    gridSquareSize = 50
    gridColor = (50,50,50)

    # | 10 000 PARTICLES |
    # w/o parallel : 130 ms
    #   w parallel : 40 ms
    # + boundcheck : 40 ms
    # + no gil     : 40 ms
    # kokeile laittaa return
    
    @njit(cache=True, parallel=True)#, nogil=True)#, boundscheck=False)
    def calculateForces(
        addedVelocities: np.ndarray,
        predictedPositions: np.ndarray,
        velocities: np.ndarray,
        positions: np.ndarray,
        neighborsArray: types.List,
        numParticles: int,
        smoothingRadius: float,
        viscosity: float,
        gravity: np.ndarray,
        dt: float,
        timestep: float

    ):

        addedVelocities.fill(0)

        for i in prange(numParticles):
            neighbors = neighborsArray[i]
            if neighbors.size > 1:
                distVectors = predictedPositions[i] - predictedPositions[neighbors]
                dists = np.sqrt(np.sum(distVectors**2, axis=1))
                distsNonzero = dists > 0
                
                forces = np.maximum(
                    -5,
                    (smoothingRadius -2 - dists[distsNonzero]) ** 3
                    )
                # paras sr = 25
                


                # for parallel only
                forces, distVectors, dists = np.broadcast_arrays(forces[:, np.newaxis], distVectors[distsNonzero], dists[distsNonzero][:, np.newaxis])
                forceVectors = forces * distVectors / dists

                # forceVectors = forces[:, np.newaxis] * distVectors[distsNonzero] / dists[distsNonzero][:, np.newaxis]


                addedVelocities[i] = np.sum(forceVectors, axis=0)

        velocities += timestep * dt * (addedVelocities * viscosity + gravity)
        positions += timestep * dt * velocities

        # 4 ms / 2 000 uusi
        # vanha: 3,4,5 näkyy vaan / 2000



    # @njit(cache=True)
    def calculateObjectForces(
        velocities: np.ndarray,
        predictedPositions: np.ndarray,
        neighborsArray: types.List,
        objects: list[object],
        dt: float,
        timestep: float
        ):
    
        for i, obj in enumerate(objects):
            neighbors = neighborsArray[i]

            if neighbors and obj.active:
                # distv pitäs olla like distsnonzero?
                distVectors = predictedPositions[neighbors] - obj.pos
                dists = np.sqrt(np.sum(distVectors**2, axis=1))
                distsNonzero = dists > 0
                forces = obj.forceFunction(dists[distsNonzero])
                distUnitVectors = distVectors[distsNonzero] / dists[distsNonzero][:, np.newaxis]
                forceVectors = distUnitVectors * forces[:, np.newaxis]
                # NOTE: nearIndices can be larger than distsNonzero => ValueError
                velocities[neighbors] += forceVectors * dt * timestep

        

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
                dists = np.linalg.norm(dist_vectors, axis=1) # , keepdims=True enne ei keepdims
                dists_nonzero = dists > 0 # np.nonzero(dists)
                
                forces = np.minimum(
                    maxForce,
                    np.maximum(
                        -5,
                        (smoothingRadius -2 - dists[dists_nonzero]) ** 3 #[dists_nonzero]
                        )
                        )
                
                # ENNEN
                    # np.maximum(
                    #     -5,
                    #     smoothingRadius - dists - 2
                    #     ) ** 3
                    #     
                force_vectors = forces[:, np.newaxis] * dist_vectors[dists_nonzero] / dists[dists_nonzero][:, np.newaxis]


                # force_vectors = force * dist_vectors / dists

                addedVelocities[i] = np.sum(force_vectors, axis=0)

    
    def start():
        # add stuff later
        Physics.arrangeParticles()

    @staticmethod
    def reset():
        # tee yks zeros() ja lopu [:] = 0
        Physics.positions = np.zeros((Physics.numParticles, 2))
        Physics.velocities = np.zeros((Physics.numParticles, 2))
        Physics.predictedPositions = np.zeros((Physics.numParticles, 2))
        Physics.addedVelocities = np.zeros_like(Physics.velocities)
        
        Physics.arrangeParticles()

    @staticmethod
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