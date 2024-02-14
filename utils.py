import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas
from shapely.geometry import Point
import cv2
from scipy.signal import convolve2d
import math
import re
def convert_xyz_to_img(xyz_path, resolution_in_meters=10, plot_image=False):
    lines = open(xyz_path,'r').readlines()
    delimiter = search_for_delimiter(lines[0])
    lines = [l.strip().split(delimiter) for l in lines]
    df = pd.DataFrame(lines, columns=['x','y','z'], dtype=float)
    min_x, min_y, min_z = min(df['x']), min(df['y']), min(df['z'])
    if min_z < 0:
        df['z'] += -1* min_z
    df['x'] = (df['x'] - min_x)/resolution_in_meters
    df['y'] = (df['y'] - min_y)/resolution_in_meters
    df['x'] = df['x'].astype(int)
    df['y'] = df['y'].astype(int)
    df.set_index(['x','y'], inplace=True)
    df.sort_index(inplace=True)
    df.reset_index(inplace=True)


    x_max, y_max = df[['x', 'y']].max()
    # x_min, y_min = df[['x', 'y']].min()
    img = np.zeros((x_max + 1, y_max + 1), dtype=df['z'].dtype)
    img[df['x'].to_numpy(dtype=int), df['y'].to_numpy(dtype=int)] = df['z'].to_numpy()

    img = np.lib.flipud(img)
    # img = np.lib.fliplr(img)

    df.set_index(['x','y'], inplace=True)
    if plot_image:
        sns.heatmap(img)
        plt.show()
    return img, delimiter, resolution_in_meters, min_x, min_y, min_z, df


def convert_img_to_coordinates(img, df, output_path, delimiter, resolution_in_meters, min_x, min_y, min_z, out_format ="xyz"):
    img = np.lib.flipud(img)
    zero_value_pairs = np.where(img == 0)
    idx_zero_value = [(zero_value_pairs[0][i],zero_value_pairs[1][i]) for i in range(len(zero_value_pairs[0]))
                      if (zero_value_pairs[0][i],zero_value_pairs[1][i]) in df.index]
    # df.set_index(['x','y'], inplace=True)

    if min_z < 0:
        df['z'] -= -1 * min_z
    df['z'] = np.round(df['z'], 3)
    df.loc[idx_zero_value] = 0
    df = pd.DataFrame(df[df['z'] != 0])
    # df['z'] = 1


    df.reset_index(inplace=True)
    df['x'] = (df['x'] * resolution_in_meters + min_x).astype(int).astype(str)
    df['y'] = (df['y'] * resolution_in_meters + min_y).astype(int).astype(str)
    df['z'] = df['z'].astype(str)

    if out_format == "xyz":
        with open(output_path, "+w") as f:
            for _, line in df.iterrows():
                f.write(delimiter.join(line) + "\n")
            f.close()
        # img = convert_xyz_to_img(output_path)[0]
        # plt.imshow(img)
        # plt.show()
    elif out_format in ["shapefile", 'shp', 'shape']:
        df['geometry'] = df.apply(lambda x: Point((float(x.x), float(x.y), float(x.z))), axis=1)
        df_geo = geopandas.GeoDataFrame(df, geometry='geometry')
        df_geo.to_file(output_path[:-4]+'.shp', driver='ESRI Shapefile')

def search_for_delimiter(line):
    line = line.strip()
    comma_in_string = "," in line
    space_in_string = " " in line
    tab_in_string = "\t" in line
    if comma_in_string == True:
        return ","
    elif space_in_string == True:
        return " "
    elif tab_in_string == True:
        return "\t"
    else:
        return None



### For Zero crossing
# Functions to plot our results
def plot_input(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()


def handle_img_padding(img1, img2):
    M1, N1 = img1.shape[:2]
    M2, N2 = img2.shape[:2]
    padding_x = int(np.abs(M2 - M1) / 2)
    padding_y = int(np.abs(N2 - N1) / 2)
    img2 = img2[padding_x:M1 + padding_x, padding_y: N1 + padding_y]
    return img2

def zero_cross_detection(image):
    z_c_image = np.zeros(image.shape)

    for i in range(0,image.shape[0]-1):
        for j in range(0,image.shape[1]-1):
            if image[i][j]>0:
                if image[i+1][j] < 0 or image[i+1][j+1] < 0 or image[i][j+1] < 0:
                    z_c_image[i,j] = 1
            elif image[i][j] < 0:
                if image[i+1][j] > 0 or image[i+1][j+1] > 0 or image[i][j+1] > 0:
                    z_c_image[i,j] = 1
    return z_c_image

DoG_kernel = [
            [0,   0, -1, -1, -1, 0, 0],
            [0,  -2, -3, -3, -3,-2, 0],
            [-1, -3,  5,  5,  5,-3,-1],
            [-1, -3,  5, 16,  5,-3,-1],
            [-1, -3,  5,  5,  5,-3,-1],
            [0,  -2, -3, -3, -3,-2, 0],
            [0,   0, -1, -1, -1, 0, 0]
        ]

LoG_kernel = np.array([
                        [0, 0,  1, 0, 0],
                        [0, 1,  2, 1, 0],
                        [1, 2,-16, 2, 1],
                        [0, 1,  2, 1, 0],
                        [0, 0,  1, 0, 0]
                    ])