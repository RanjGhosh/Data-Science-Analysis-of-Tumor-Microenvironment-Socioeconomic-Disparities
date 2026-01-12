'''
0 = Glass/No tissue
1 = Adipose
2 = Immune
3 = Necrosis
4 = Normal
5 = Stroma
6 = Tumor
'''
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import pickle

with open(r"C:\Users\rghosh6\Desktop\GSU Courses\GSU Courses\UAP RESEARCH\infers\infers\FS12-05195- F3 - 2022-01-11 01.39.46.pkl", 'rb') as f:
    file_data = pickle.load(f)

impdata = []

# extract only tissue tiles (1–6)
for i in file_data['tilesData']:
    coord = [i['row'], i['col']]
    if i['pred'] in [1, 2, 3, 4, 5, 6]:
        impdata.append([coord, i['pred']])

# build grid for subsampling
coordinates = np.array([item[0] for item in impdata])
grid_size = 20

min_x, max_x = np.min(coordinates[:, 0]), np.max(coordinates[:, 0])
min_y, max_y = np.min(coordinates[:, 1]), np.max(coordinates[:, 1])

x_grid = np.linspace(min_x, max_x, grid_size)
y_grid = np.linspace(min_y, max_y, grid_size)

# subsample
subsampled_points = []
for x in x_grid:
    for y in y_grid:
        idx = np.argmin(np.linalg.norm(coordinates - np.array([x, y]), axis=1))
        subsampled_points.append(impdata[idx])

# Delaunay plot
coordinates = np.array([item[0] for item in subsampled_points])
codes = [item[1] for item in subsampled_points]

triangulation = Delaunay(coordinates)

plt.triplot(coordinates[:, 0], coordinates[:, 1], triangulation.simplices)
plt.plot(coordinates[:, 0], coordinates[:, 1], 'o')

for i, code in enumerate(codes):
    plt.annotate(str(code), (coordinates[i, 0], coordinates[i, 1]))

plt.show()
