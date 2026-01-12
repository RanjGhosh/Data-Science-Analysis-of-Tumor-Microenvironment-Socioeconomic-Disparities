import os
import pickle
import pandas as pd
import numpy as np


def findNeighbors(grid, x, y, window=2):
    xi = np.arange(-window, window+1) if 0 < x < len(grid[0]) - 1 else (np.arange(0, -(window+1), -1) if x > 0 else np.arange(0, window+1))
    yi = np.arange(-window, window+1) if 0 < y < len(grid[0]) - 1 else (np.arange(0, -(window+1), -1) if y > 0 else np.arange(0, window+1))
    for a in xi:
        for b in yi:
            if a == b == 0:
                continue
            if x+a >= grid.shape[0]:
                continue
            if y+b >= grid.shape[1]:
                continue
            yield grid[x + a, y + b]
    
def get_ratio(class_nums, list1, list2):
    numr = sum([class_nums[i] for i in list1])
    denom = sum([class_nums[i] for i in list2])
    return numr/(numr+denom)


if __name__ == "__main__":
    class_ids = {
        'Adipose tissue':1,
        'Immune cells':2,
        'Necrosis':3,
        'Normal tissue':4,
        'Stroma':5,
        'Tumor':6}

    df = pd.read_excel(
        'ADI_TME clinical file.xlsx',
        sheet_name='Sheet1')
    slide_ids = df['Path ID'].values.tolist()
    # print(len(slide_ids))
    
    folder = "infers"
    infers = os.listdir(folder)
    # print(len(infers))

    for i,row in df.iterrows():
        slide_id = row[0]
        race = row[1]
        adi = row[2]
        print(f'{i+1} : {slide_id}, {race}, {adi}')
        
        for j in infers:
            if slide_id in j:
                infer_file = j
        class_mask = np.load(os.path.join(folder, infer_file))
        
        class_nums = []
        for j in range(1,7):
            class_nums.append(np.count_nonzero(class_mask == j))
        non_fat = sum(class_nums[1:])

        print("Calculating Area measures...")
        area_features = []
        class_nums.append(non_fat)
        # 1. area_tumor_vs_non_fat
        x = get_ratio(class_nums, [5], [6])
        area_features.append(x)
        # 2. area_tumor+necrosis_vs_non_fat
        x = get_ratio(class_nums, [5, 2], [6])        
        area_features.append(x)
        # 3. area_tumor+necrosis+immune_vs_non_fat
        x = get_ratio(class_nums, [5, 2, 1], [6])
        area_features.append(x)
        # 4. area_necrosis vs tumor
        x = get_ratio(class_nums, [2], [5])
        area_features.append(x)
        # 5. area_immune_vs_tumor
        x = get_ratio(class_nums, [1], [5])
        area_features.append(x)
        # 6. area_stroma_vs_tumor
        x = get_ratio(class_nums, [4], [5])
        area_features.append(x)
        # 7. area_immune vs tumor+necrosis
        x = get_ratio(class_nums, [1], [5, 2])
        area_features.append(x)
        # 8. area_stroma vs tumor+necrosis
        x = get_ratio(class_nums, [4], [5, 2])
        area_features.append(x)
        # 9. area_stroma vs tumor+stroma
        x = get_ratio(class_nums, [4], [5, 4])
        area_features.append(x)
        # 10. area_stroma vs tumor+stroma+necrosis
        x = get_ratio(class_nums, [4], [5, 4, 2])
        area_features.append(x)
        # 11. area_necrosis vs tumor+necrosis
        x = get_ratio(class_nums, [2], [5, 2])
        area_features.append(x)
        print("Completed!")
        
        print("Calculating Co-Occurence counts...")
        print("Generating label grid")
        # grid = get_labelgrid(data)
        grid = class_mask
        grid = grid.astype(int)
        print("Iterating over windows")
        counts = np.zeros((6,6), dtype=int)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                center = grid[i,j]
                counts_perwindow = np.zeros((1,6), dtype=int)  
                if center !=0:
                    window = np.array(list(findNeighbors(grid, i, j, window=8)))
                    for k in np.arange(1,7):
                        count = np.count_nonzero(window==k)
                        counts_perwindow[0, k-1] = count
                    if counts_perwindow[0,center-1] > 0:
                        counts_perwindow[0,center-1] -=1
                    counts[center-1,:] += counts_perwindow[0,:]
                else:
                    pass

        adj_measures = []
        # 1. adj_necrosis vs tumor
        x = counts[2,5]/(counts[2,5]+class_nums[5])
        adj_measures.append(x)
        # 2. adj_immune vs tumor
        x = counts[1,5]/(counts[1,5]+class_nums[5])
        adj_measures.append(x)
        # 3. adj_stroma vs tumor
        x = counts[4,5]/(counts[4,5]+class_nums[5])
        adj_measures.append(x)
        # 4. adj_necrosis vs tumor+necrosis
        x = (counts[2,5]+counts[2,2])/(counts[2,5]+counts[2,2]+class_nums[5]+class_nums[2])
        adj_measures.append(x)
        # 5. adj_immune vs tumor+necrosis
        x = (counts[1,5]+counts[1,2])/(counts[1,5]+counts[1,2]+class_nums[5]+class_nums[2])
        adj_measures.append(x)
        # 6. adj_stroma vs tumor+necrosis
        x = (counts[4,5]+counts[4,2])/(counts[4,5]+counts[4,2]+class_nums[5]+class_nums[2])
        adj_measures.append(x)
        # 7. adj_necrosis vs any
        x = sum(counts[2,:])/(sum(counts[2,:])+sum(class_nums))
        adj_measures.append(x)
        # 8. adj_immune vs any
        x = sum(counts[1,:])/(sum(counts[1,:])+sum(class_nums))
        adj_measures.append(x)
        # 9. adj_stroma vs any
        x = sum(counts[4,:])/(sum(counts[4,:])+sum(class_nums))
        adj_measures.append(x)
        # 10. adj_necrosis vs non fat
        x = sum(counts[2,1:])/(sum(counts[2,1:])+non_fat)
        adj_measures.append(x)
        # 11. adj_immune vs non fat
        x = sum(counts[1,1:])/(sum(counts[1,1:])+non_fat)
        adj_measures.append(x)
        # 12. adj_stroma vs non fat
        x = sum(counts[4,1:])/(sum(counts[4,1:])+non_fat)
        adj_measures.append(x)
        print("Completed!")
        # print(adj_measures)

        data_tme = {
            'file': infer_file,
            'race': race,
            'adi': adi,
            'class_nums': class_nums,
            'area':area_features,
            'adj': adj_measures}
        pickle.dump(data_tme, open( "tme_measures/"+infer_file[:-5]+".pkl", "wb" ) )
        print("Data saved\n")
        # print(len(adj_measures))

    """
    Following code is for combining all features into a single excel file
    TODO: Incorporate this into the above snippet.
    """
    # folder = "tme_measures"
    # files = os.listdir(folder)
    # col_names = ["slide", "race", "adi",
    #     "area_tumor_vs_non_fat", "area_tumor+necrosis_vs_non_fat", "area_tumor+necrosis+immune_vs_non_fat", "area_necrosis_vs_tumor",
    #     "area_immune_vs_tumor", "area_stroma_vs_tumor", "area_immune_vs_tumor+necrosis", "area_stroma_vs_tumor+necrosis",
    #     "area_stroma_vs_tumor+stroma", "area_stroma_vs_tumor+stroma+necrosis", "area_necrosis_vs_tumor+necrosis",
    #     "adj_necrosis_vs_tumor", "adj_immune_vs_tumor", "adj_stroma_vs_tumor", "adj_necrosis_vs_tumor+necrosis", "adj_immune_vs_tumor+necrosis",
    #     "adj_stroma vs tumor+necrosis", "adj_necrosis vs any", "adj_immune vs any", "adj_stroma vs any",
    #     "adj_necrosis vs non fat", "adj_immune vs non fat", "adj_stroma vs non fat" 
    # ]
    # print(len(col_names))
    # l1 = []
    # for i,file in enumerate(files):
    #     print(i,file)
    #     data = pickle.load(open( os.path.join(folder,file), "rb" ))
    #     l2 = []
    #     keys = ['file', 'race', 'adi', 'area', 'adj']
    #     for key in keys:
    #         v = data[key]
    #         if isinstance(v, list):
    #             l2 += v
    #         else:
    #             if key=='file':
    #                 l2.append(v[:-5])
    #             else:
    #                 l2.append(v)
    #     l1.append(l2)
    #     # break
    # df = pd.DataFrame(l1, columns=col_names)
    # writer = pd.ExcelWriter("tme_features.xlsx", engine="xlsxwriter")
    # ea = df[df.race == 'White']
    # aa = df[df.race == 'Black']
    # dff = [aa, ea]
    # dff = pd.concat(dff)
    # dff.to_excel(writer, sheet_name='Sheet1', index=False)
    # aa.to_excel(writer, sheet_name='AA', index=False)
    # ea.to_excel(writer, sheet_name='EA', index=False)
    # writer.close()