import time
from timeit import timeit
import numpy as np
import numba
from numba import types, typed, prange


positions = np.array([[0.,   2.],
                      [37., 400.],
                      [25., 67.],
                      [251., 80.],
                      [10.,  2.]])
gridSquareSize = 50

numParticles = 5

@numba.njit(parallel=True)
def func():
    # assign grid index by position
    grid2D = positions.astype(np.int64) // gridSquareSize

    
    # jokaiselle hiukkaselle lasketaan sitten oma gridIndex
    # gridIndexillä pääsee listaan, joka sisältää saman ruudun muut hiukkasten indeksit

    smallGridSquares = np.zeros((12,16), dtype=np.int64)
    for index in grid2D:
        smallGridSquares[index[0], index[1]] += 1

    maxParticlesPerSquare = np.max(smallGridSquares)

    squaresWithParticles = np.nonzero(smallGridSquares)
    
    # numba.typeof()
    # d = typed.Dict.empty(
    #     key_type=(types.int64, types.int64),
    #     value_type=types.List
    # )
    
    # I try to avoid using empty lists with numba, mainly because an empty list cannot be typed.
    # Check out nb.typeof([])

    # You can initialize an empty but typed List with List.empty_list(<type>)

    #empty_f32 = [np.float32(x) for x in range(0)]. That last one only works when inside a jitted function

    # convert to 1d grid
    grid1D = grid2D[:,1] * 16 + grid2D[:,0]
    print(grid1D)
    
    # particleGroups = [np.empty((1,), dtype=np.int64) for _ in range(192)]
    # particleGroups = [np.array([-1], dtype=np.int64) for _ in range(192)]
    # particleGroups = [[-1] for _ in range(192)]

    particleGroups = [typed.List.empty_list(np.int64) for _ in range(192)]
    particleGroups[0].append(2)
    # np.concatenate
    for i in prange(numParticles):
        # particle i's grid index
        particleGroup: np.int64 = grid1D[i]
        print(particleGroup) # hiukkasen i indeksi 
        # eli voidaan listaan kohtaan squareNums[i] lisätä i !!!!!
        particleGroups[particleGroup].append(i)

    print(particleGroups)





    # print(squaresWithParticles)
    # print(squaresWithParticles[0])
    # print(squaresWithParticles[1])


    # gridSquares = np.zeros((12,16, maxParticlesPerSquare), dtype=np.int64)
    # for i in range(numParticles):
    #     gridSquares[index[0], index[1]].append(i)
    # np.empty

    # gridSquares = []




    maxLen = 16 * 12
    gridSquares = np.zeros((maxLen,), dtype=np.int64)

    # # muuta prangeks ehkä
    # for index in gridIndices:
    #     squareNum = index[1] * 16 + index[0]
    #     gridSquares[squareNum] += 1


    # xMax = np.max(gridIndices[:,0])
    # yMax = np.max(gridIndices[:,1])
    # print(xMax,yMax)

    # neighborsArray = [] # typed.List()     

    # print(uniqueIndices)

    # list numpy arrays

func()