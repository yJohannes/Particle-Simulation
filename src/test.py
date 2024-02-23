import time
from timeit import timeit
import numpy as np
import numba
from numba import types, typed, prange


            # grid_indices = (Physics.predictedPositions // Physics.gridSquareSize).astype(int)
            # flattened_indices = grid_indices[:, 0] + grid_indices[:, 1] * 4
            # unique_elements, occur_indices = np.unique(flattened_indices, return_inverse=True,)

            # neighborsArray = numbaList(
            #     np.full((1,), -1, dtype=np.int64)
            #     for i in range(12)
            # )

            # for i, element in enumerate ( unique_elements ):
            #     neighborsArray[element] = np.where(occur_indices == i)[0]



positions = np.array([[0.,   2.],
                      [37., 400.],
                      [25., 67.],
                      [251., 80.],
                      [10.,  2.]])
gridSquareSize = 50

numParticles = 5

grid2D = positions.astype(np.int64) // gridSquareSize


grid_indices = (positions / gridSquareSize).astype(int)

# Calculate flattened indices
flattened_indices = grid_indices[:, 0] + grid_indices[:, 1] * (800 // 50)

# Find unique flattened indices and their corresponding indices
unique_flattened_indices, inverse_indices = np.unique(flattened_indices, return_inverse=True)

# Group positions by cell
grouped_positions = [np.where(inverse_indices == i)[0] for i in range(len(unique_flattened_indices))]

print("Hash map:")
for cell_id, indices in enumerate(grouped_positions):
    print(f"Cell {cell_id}: {indices}")


# Convert positions to grid indices
grid_indices = (positions / gridSquareSize).astype(int)

# Calculate flattened indices
flattened_indices = grid_indices[:, 0] + grid_indices[:, 1] * (800 // gridSquareSize)

# Sort the flattened indices to ensure positions within the same cell are contiguous
sorted_indices = np.argsort(flattened_indices)

# Group positions by cell
uniques, indices, counts = np.unique(flattened_indices, return_index=True, return_counts=True)

print(flattened_indices)
print(uniques)
# indices that give the unique values
print(indices)
print(counts)
# cumulative_counts = np.cumsum(counts)
# grouped_positions = np.split(sorted_indices, cumulative_counts[:-1])

# print(grouped_positions)
# print("Hash map:")
# for cell_id, indices in enumerate(grouped_positions):
#     print(f"Cell {cell_id}: {indices}")



# # hashMap = [[] for _ in range(192)]
# # typed.List.empty_list(np.int64)
# hashMap = [[] for _ in range(192)]
# # hashMap = typed.List(hashMap)

# indices = np.unique(grid2D, axis=1)


# for i_xy in indices:
#     key = np.where(grid2D == i_xy)
#     hashMap[i_xy[1] * 16 + i_xy[0]] = key

# print(hashMap)




@numba.njit(parallel=True)
def func():
    # # assign grid index by position
    # grid2D = positions.astype(np.int64) // gridSquareSize

    
    # jokaiselle hiukkaselle lasketaan sitten oma gridIndex
    # # gridIndexillä pääsee listaan, joka sisältää saman ruudun muut hiukkasten indeksit

    # smallGridSquares = np.zeros((12,16), dtype=np.int64)
    # for index in grid2D:
    #     smallGridSquares[index[0], index[1]] += 1

    # maxParticlesPerSquare = np.max(smallGridSquares)

    # squaresWithParticles = np.nonzero(smallGridSquares)
    
    # numba.typeof()
    # d = typed.Dict.empty(
    #     key_type=(types.int64, types.int64),
    #     value_type=types.List
    # )
    
    # I try to avoid using empty lists with numba, mainly because an empty list cannot be typed.
    # Check out nb.typeof([])

    # You can initialize an empty but typed List with List.empty_list(<type>)

    #empty_f32 = [np.float32(x) for x in range(0)]. That last one only works when inside a jitted function

    # NOTE: np.concatenate 
    # assign grid index by position
    grid2D = positions.astype(np.int64) // gridSquareSize

    # indices = np.ravel_multi_index(grid2D.T, grid2D.shape)
    # print("INDICES:",indices)


    # convert to 1d grid
    grid1D = grid2D[:,1] * 16 + grid2D[:,0]
    
    particleGroups = [typed.List.empty_list(np.int64) for _ in range(192)]
    particleGroups = typed.List(particleGroups)

    for i in prange(numParticles):
        # particle i's grid index
        particleGroup: np.int64 = grid1D[i]
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

# func()