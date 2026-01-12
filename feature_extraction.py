'''
0 = Glass/No tissue
1 = Adipose
2 = Immune
3 = Necrosis
4 = Normal
5 = Stroma
6 = Tumor
'''
import pickle
import pandas as pd
import numpy as np
from scipy.stats import describe
from skimage.measure import regionprops

with open(r"C:\Users\rghosh6\Desktop\GSU Courses\GSU Courses\UAP RESEARCH\infers\infers\FS12-05195- F3 - 2022-01-11 01.39.46.pkl", 'rb') as f:
    file_data = pickle.load(f)

impdata = []

for i in file_data['tilesData']:
    coord = [i['row'], i['col']]
    if i['pred'] in [1, 2, 3, 4, 5, 6]:
        impdata.append([coord, i['pred']])

df = pd.DataFrame(impdata, columns=['coord', 'pred'])
df['coord'] = df['coord'].apply(tuple)

# 1. Class Distribution
class_distribution = df['pred'].value_counts()

# 2. Spatial Distribution
spatial_distribution = df.groupby('pred')['coord'].agg(list).reset_index()

# 3. Size & Shape Features
region_props = df.groupby('pred')['coord'].apply(lambda x: regionprops(np.array(x))[0])

# 4. Intensity Features
intensity_features = df.groupby('pred')['coord'].apply(lambda x: describe(np.array(x).mean()))

# Display
print("1. Class Distribution:")
print(class_distribution)

print("\n2. Spatial Distribution:")
print(spatial_distribution)

print("\n3. Size and Shape Features:")
for idx, props in region_props.items():
    print(f"Tissue Type {idx} - Area: {props.area}, Perimeter: {props.perimeter}")

print("\n4. Intensity Features:")
for idx, features in intensity_features.items():
    print(f"Tissue Type {idx} - Mean Intensity: {features.mean}")
