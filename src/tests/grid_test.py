import numpy as np

# Generate sample positions
# positions = np.random.randint(0, 800, size=(100, 2))

positions = np.array([[0.,   2.], # 0
                      [37., 400.],# 3
                      [25., 67.], # 1
                      [251., 80.],# 2
                      [10.,  2.]])# 0

numParticles = 5

# Define grid cell size
cell_size = np.array([50, 50])

# Convert positions to grid indices
grid_indices = (positions // cell_size).astype(int)

# Calculate flattened indices
flattened_indices = grid_indices[:, 0] + grid_indices[:, 1] * (800 // 50)
print("    Flattened indices:", flattened_indices)


unique_elements, occur_indices = np.unique(flattened_indices, return_inverse=True)

print("      Unique elements:", unique_elements)
print("Indices of occurrence:", occur_indices)

gridList = [-1 for _ in range(16*12)]


unique_elements, occur_indices = np.unique(flattened_indices, return_inverse=True)

gridList = [[] for _ in range(16*12)]
for i, element in enumerate ( unique_elements ):
    gridList[element] = np.where(occur_indices == i)

gridList = np.full((occur_indices.size))

unique_elements, occur_indices = np.unique(flattened_indices, return_inverse=True)
gridList = np.zeros((16*12, flattened_indices.size), dtype=int) - 1
gridList[occur_indices, np.arange(flattened_indices.size)] = np.arange(flattened_indices.size)

print(gridList)

# 1: yhdistä uniikkien elementtien indeksit indekseihin, joista ne löytyvät 
# 2: yhdistä 

# indices of occureccnse vastaavat flattened indices
# esim. occ : id 4 numero kertoo mitä 
"⭠ ⭡ ⭢ ⭣ ↖ ↗ ↘ ↙"

# unique elements ovat ruudut. occur_indices on hiukkasten id
"""                         

gridList = [ [] for _ in range( numParticles ) ]
for i, element in enumerate(unique_elements):
    gridList[element] = np.where(occur_indices == i)

    # uniikkia_elementtiä_i_vastaavat_löydät_lytätystä_listasta
    x = np.where(occur_indices == i)
    [  0   3   1   2   0] == 0
    >>> 0, 4
    Eli hiukkaset 0,4 ovat gridissä 0. 
    
    flattened_indices = hiukkasen i sijainti gridissä

                          0   1   2   3
      Unique elements: [  0  16  21 128]

                          0   1   2   3   4  
Indices of occurrence: [  0   3   1   2   0]

                          0   1   2   3   4
    Flattened indices: [  0 128  16  21   0]



"""

# Find unique flattened indices and their corresponding indices in the positions array
unique_indices, positions_indices = np.unique(flattened_indices, return_inverse=True)
# print("All unique indices:",unique_indices)
# print("unique indices occur at indices:",positions_indices)
IDS = np.where(flattened_indices[:, None]==unique_indices, flattened_indices[:, None], unique_indices)
# print(IDS)
# Use numpy's bincount to count occurrences of each unique flattened index
cell_counts = np.bincount(positions_indices)

# Use numpy's cumulative sum to find the starting index of each group of positions
cumulative_indices = np.cumsum(cell_counts)

# Create a mask to find the indices where the groups of positions start
mask = np.zeros_like(flattened_indices, dtype=bool)
mask[cumulative_indices[:-1]] = True

# Use numpy's where to find the indices of the positions where the groups start
start_indices = np.where(mask)[0]

# Shift the start indices to find the end indices of each group
end_indices = np.roll(start_indices, -1)
end_indices[-1] = len(flattened_indices)

# Use numpy's split to split the positions array into groups based on the start and end indices
grouped_positions = np.split(positions, end_indices)
