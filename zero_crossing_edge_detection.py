#Implementation of Canny Edge Detector
import os
from utils import *
from canny_utils import *

def zero_crossing(input_path,kernel_type, output_format):
    basename,extension = input_path.split(".")
    output_path = os.path.join(".","output",basename)
    is_input_color = extension in ["png", "jpeg", "jpg"]
    if not is_input_color and not extension == 'xyz':
        print("Illegal format. Skipping file")
        return

    # For debugging if needed
    show_images = False
    save_images = True


    img_output_path = output_path + basename
    if not is_input_color:
        input_img, delimiter, resolution_in_meters, min_x, min_y, min_z, df = convert_xyz_to_img(input_path)
    else:
        input_img = imageio.v2.imread(input_path)
    if show_images:
        plt.imshow(input_img)
        plt.show()
    kernel = DoG_kernel if kernel_type.lower() == 'dog' else LoG_kernel  # Can use DoG_kernel or LoG_kernel
    diff_of_gauss_img = convolve2d(input_img, kernel)

    zero_crossing_dog = zero_cross_detection(diff_of_gauss_img)
    # zero_crossing_dog = handle_img_padding(input_img, zero_crossing_dog)
    if show_images:
        plot_input(zero_crossing_dog, 'DoG - Zero Crossing')
    ksize = 3
    delta = 2
    sobelx = cv2.Sobel(diff_of_gauss_img, cv2.CV_64F, delta, 0, ksize=ksize)
    sobely = cv2.Sobel(diff_of_gauss_img, cv2.CV_64F, 0, delta, ksize=ksize)
    sobel_first_derivative = cv2.magnitude(sobelx, sobely)
    # plot_input(sobel_first_derivative, 'Sobel First Order Derivative')
    if show_images:
        plt.hist(sobel_first_derivative)
        plt.show()

    sobel_test = np.empty_like(sobel_first_derivative)
    sobel_test[:] = sobel_first_derivative / 255
    sobel_test = sobel_test * 255 / np.max(sobel_test)

    # if show_images:
    plt.hist(sobel_test)
    plt.show()

    threshold = 0.9
    sobel_test[sobel_test > threshold] = 255
    sobel_test[sobel_test < threshold] = 0
    output_img = cv2.bitwise_and(zero_crossing_dog, sobel_test)

    output_filename = f"output_ksize-{ksize}_thrsh-{threshold}_krnl-{kernel_type}_delta-{delta}"
    print(output_filename)
    if show_images:
        plot_input(sobel_test, 'Boosted 1st order Derivative - Sobel edges')
        plot_input(output_img, 'Strong Edges detected by DoG Zero Crossing')

    if save_images:
        plt.imsave(img_output_path + f"/{output_filename}.png", output_img, cmap=plt.get_cmap('gray'))

    # Phase 4 : If origin is XYZ, convert to XYZ
    if not is_input_color:
        convert_img_to_coordinates(output_img, df, img_output_path + "/edges.xyz",
                                   delimiter, resolution_in_meters, min_x, min_y, min_z,
                                   out_format=output_format)