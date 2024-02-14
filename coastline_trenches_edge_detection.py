#Implementation of Canny Edge Detector
import os
import time

from utils import *
from canny_utils import *

input_path = "./test_data/"
output_path = "./canny_output/"
img_list = [path for path in os.listdir(input_path)]
# img_list = [path for path in os.listdir(input_path) if ".xyz" in path]
show_images = False
# Load the input images
for img in img_list:
    extension = img.split(".")[-1]
    is_input_color = extension in ["png","jpeg","jpg"]
    if not is_input_color and not extension == 'xyz':
        print("Illegal format. Skipping file")
        continue

    img_output_path = output_path + img.split('.')[0]
    if not is_input_color:
        xyzfilename = input_path + img
        input_img, delimiter, resolution_in_meters, min_x, min_y, min_z, df= convert_xyz_to_img(input_path + img)
    else:
        input_img = imageio.v2.imread(input_path + img)
    if show_images:
        plt.imshow(input_img)
        plt.show()

    # Convert the image to grayscale if color and apply gaussian blurring
    sigma = 1.5
    if is_input_color:
        grey_img = rgb2gray(input_img)
        blur_img = ndimage.gaussian_filter(grey_img, sigma=sigma)
    else:
        blur_img = ndimage.gaussian_filter(input_img, sigma=sigma)
    if show_images:
        plt.imshow(blur_img)
        plt.show()


    # Find gradient Fx
    x_grad = gradient_x(blur_img)

    # Find gradient Fy
    y_grad = gradient_y(blur_img)

    # Compute edge strength
    grad_mag = gradient_mag(x_grad, y_grad)
    if show_images:
        plt.imshow(grad_mag, cmap=plt.get_cmap('gray'))
        plt.show()

    os.makedirs(img_output_path, exist_ok=True)
    plt.imsave(img_output_path + "/img1_grad_mag.png", grad_mag, cmap=plt.get_cmap('gray'))

    # Compute direction of gradient
    grad_dir = np.degrees(np.arctan2(y_grad, x_grad))
    if show_images:
        plt.imshow(grad_dir, cmap=plt.get_cmap('gray'))
        plt.show()
    plt.imsave(img_output_path + "/img2_grad_dir.png", grad_dir,cmap=plt.get_cmap('gray'))

    # Phase 2 : Non maximal suppression
    closest_dir = closest_dir_function(grad_dir)
    # thinned_output = grad_mag
    thinned_output = non_maximal_suppressor(grad_mag, closest_dir)

    if show_images:
        plt.imshow(thinned_output, cmap=plt.get_cmap('gray'))
        plt.show()
    plt.imsave(img_output_path + "/img3_thinned.png", thinned_output, cmap=plt.get_cmap('gray'))


    # Phase 3 : Hysteresis Thresholding
    if is_input_color:
        low_ratio, high_ratio = 0.1, 0.13
    else:
        low_ratio, high_ratio = 0.015, 0.04
    t0 = time.time()
    output_img = hysteresis_thresholding(thinned_output, low_ratio=low_ratio, high_ratio=high_ratio)
    if show_images:
        plt.imshow(output_img, cmap=plt.get_cmap('gray'))
        plt.show()
    plt.imsave(img_output_path + "/img4_output.png", output_img, cmap=plt.get_cmap('gray'))
    print(f"Finished Phase 3 in {round(time.time() - t0)} seconds")

    # Phase 4 : If origin is XYZ, convert to XYZ
    t0 = time.time()
    # if not is_input_color:
        # convert_img_to_coordinates(output_img, df, img_output_path + "/edges.xyz",
        #                            delimiter, resolution_in_meters, min_x, min_y, min_z,
        #                            out_format='shp')
        # convert_img_to_coordinates(output_img, df, img_output_path + "/edges.xyz",
        #                            delimiter, resolution_in_meters, min_x, min_y, min_z,
        #                            out_format='xyz')
    print(f"Finished Phase 4 in {round(time.time() - t0)} seconds")