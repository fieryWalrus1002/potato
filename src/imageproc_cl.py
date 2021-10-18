import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import glob
import yaml
from datetime import date
from scipy import ndimage
import cv2 as cv
from itertools import combinations
from itertools import product

from imageprocessing import ImageProcessing
from roi import PlotCoordinates
import sys
import numpy
#https://realpython.com/python-command-line-arguments/

class ImageProcCL:
    """class for processing images without notebook"""

    def define_camera_parameters(self):
        camera_parameters = {
            "blue-444": 444,
            "blue": 475,
            "green-531": 531,
            "green": 560,
            "red-650": 650,
            "red": 668,
            "red-edge-705": 705,
            "red-edge": 717,
            "red-edge-740": 740,
            "nir": 842,
        }
        return camera_parameters

    # define camera wavelengths and file image labels in a dict
    # RedEdge-MX Dual Camera Imaging System bands
    # channel names: blue-444, blue, green-531, green, red-650, red, red-edge-705, red-edge, red-edge-740, nir

    # define notebook parameters
    params = {
        "project_stub": "Potato_Fertilizer_Othello",
        "image_format": "*.tif",
        "data_acquisition_date": "Jun22_2020",
        "NDVI_threshold": 0.3,
        "data_import_path": Path.cwd() / "data" / "raw" / "Jun22_2020",
        "data_export_path": Path.cwd() / "data" / "processed" / "Jun22_2020",
        "plot_export_path": Path.cwd() / "image_export",
        "ground_truth_path": Path.cwd() / "data" / "raw" / "ground_truth.csv",
    }

    improc = ImageProcessing(params=params)

    # import images with given parameters
    field_image = np.stack(
        [
            ndimage.rotate(
                improc.load_img(channel_name), angle=182.4, reshape=True
            )
            for channel_name in rededge_mx_band_wl
        ]
    )

    # origin is upper left
    y_limits = [2400, 9800]
    x_limits = [1460, 3050]

    # crop to desired size (channels, y axis, x axis)
    field_image = field_image[
        :, y_limits[0] : y_limits[1], x_limits[0] : x_limits[1]
    ]
    print(f"final field_image.shape: {field_image.shape}")

    improc.show_image(field_image[9], size=(4, 5))
    # # Test threshold values and create a mask
    # 1. Calculate NDVI
    # 2. Choose NDVI threshold
    # 3. Create a boolean mask using the threshold
    # 4. Apply the mask to the NDVI image.
    # 5. Display masked NDVI image for verification
    # 1. calculate NDVI
    ndvi = improc.calc_spec_idx((9, 5), field_image)

    # 2. choose ndvi threshold
    ndvi_th = 0.3

    # 3. create a boolean mask of pixels > ndvi_th
    mask = np.where(ndvi > ndvi_th, True, False)

    # 4. apply mask to cropped image
    ndvi_masked = np.multiply(ndvi, mask)

    # 5. Display the images as one figure.
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(ndvi, cmap="gray")

    axs[0].set_title("NDVI")

    axs[1].imshow(ndvi_masked, cmap="gray")
    axs[1].set_title(f"NDVI_masked, th:{ndvi_th}")

    fig.tight_layout()
    fig.set_figheight(8)
    # Congratulations! You have a field_image with all ten bands, and a mask to segment the background out from the vegetation.
    # save the field_image array to disk for use in other notebooks
    # this will be an np.array object, which can then be loaded using np.load()
    array_filename = os.path.join(
        params["data_export_path"],
        f"{params['project_stub']}_{params['data_acquisition_date']}_fieldimage",
    )
    mask_filename = os.path.join(
        params["data_export_path"],
        f"{params['project_stub']}_{params['data_acquisition_date']}_mask",
    )
    np.save(file=array_filename, arr=field_image)
    np.save(file=mask_filename, arr=mask)

    # # Define the NDSIs
    # NDSIs are defined by the equation $\frac{a-b}{a+b}$, where a and b are different color bands taken from the available channels in the field image. There are 10 channels available on data taken from the camera utilized for this dataset, which would yield $\frac {10!}{(10-2)!}=90$ combinations. However, half of those would be the inverse of the other half: $\frac{a-b}{a+b}$ vs. $\frac{b-a}{b+a}$.

    # The inversed NDSI features would be very highly autocorrelated with their partner, and would ultimately need to be removed from the dataset during dimensional reduction. Instead we choose to take only the unique permutations, resulting in $\frac {10!}{2!(10-2)!}=45$ NDSI features to calculate.

    img_chan = {
        0: "blue",
        1: "blue_444",
        2: "green",
        3: "green_531",
        4: "red_650",
        5: "red",
        6: "red_edge_705",
        7: "red_edge",
        8: "red_edge_740",
        9: "nir",
    }

    ndsi_list = [combo for combo in combinations(iter(img_chan), 2)]

    ndsi_name_list = [
        f"{img_chan.get(combo[0])}-{img_chan.get(combo[1])}"
        for combo in ndsi_list
    ]

    print(
        f"There are {len(ndsi_list)} unique combinations in contained in ndsi_list."
    )

    # create an image stack with a channel for each NDSI in our list
    # We need to perform the calculations to generate a new image stack, with one channel for each NDSI. We can use the function calc_spec_idx_from_combo() to create the stack of np.arrays. It takes a tuple of two ints, with each tuple representing a combination of two image channels.
    # # create ndsi stack
    ndsi_stack = np.stack(
        [improc.calc_spec_idx(combo, field_image) for combo in ndsi_list]
    )

    print(f"ndsi_stack.shape={ndsi_stack.shape}")
    ndsistack_filename = os.path.join(
        params["data_export_path"],
        f"{params['project_stub']}_{params['data_acquisition_date']}_mask",
    )
    np.save(file=ndsistack_filename, arr=ndsi_stack)

    # Calculate boundary of plots
    # In the case of the potato, the lower left of the field is plot 0, with plot_id incrementing with range, then starting again from the bottom.
    pc = PlotCoordinates()

    # variables
    plot_shape = (200, 492)  # w,h
    edge_buf = 40  # buffer around edge of plot
    roi_shape = pc.get_roi_shape(
        plot_shape, edge_buf
    )  # smaller coordinates within plot_shape
    num_ranges = 13
    bottom_offset = 50  # offset from bottom of image

    # set the x origins for the plots, and the y origins will be calculated
    x_origins = [50, 355, 555, 850, 1050, 1350]
    y_origins = [
        ndsi_stack.shape[1] - bottom_offset - plot_shape[1] * y
        for y in range(1, num_ranges + 1)
    ]

    # use these values to calculate the plot coordinates
    plot_coords = list(product(x_origins, y_origins))
    roi_coords = [
        pc.get_roi_coord(plot_coord=plot_coord, edge_buf=edge_buf)
        for plot_coord in plot_coords
    ]

    # now plot them for verification on the NDVI image
    plot_id_list = pc.plot_boundaries(
        img=ndvi,
        plot_coords=plot_coords,
        roi_coords=roi_coords,
        plot_shape=plot_shape,
        roi_shape=roi_shape,
    )
    # A note on calculating mean values.
    ### You have to exclude the background!
    # We need to be careful when we calculate our mean values for the roi. We can't include values from the background. To exclude these, we utilize the NDVI thresholded mask we created above. Just to see how it works, here is a simple test of the mask on a very small array. A mask is provided that excludes values less than 1. They are not included in the number of pixels when the average value is calculated, as seen below.
    roi = np.array([0, 1, 2, 3])
    roi_mask = np.where(roi >= 1, True, False)
    roi_avg4 = (1 + 2 + 3) / 4
    roi_avg3 = (1 + 2 + 3) / 3

    print(f" sum(roi)/4 = {np.sum(roi)/4}, sum(roi)/3 = {sum(roi)/3}")
    print(roi_mask)
    print(f"np.mean(roi) = {np.mean(roi)}")
    print(f"np.mean(roi, where=mask) = {np.mean(roi, where=roi_mask)}")

    # calculate the NDSI means and export the dataframe as a *.csv
    # We want to use this data in other notebooks for modeling, so lets combine it with our ground truth data. After it is joined on the plot id, we export it to the processed data path.
    ndsi_means = np.stack(
        [
            [
                improc.ndsi_mean(
                    arr=ndsi, origin=origin, shape=roi_shape, mask=mask
                )
                for ndsi in ndsi_stack
            ]
            for origin in roi_coords
        ]
    )

    df = pd.read_csv(params["ground_truth_path"])[["plot_id", "yield"]]

    ndsi_df = pd.concat(
        [
            pd.DataFrame(plot_id_list, columns=["plot_id"]),
            pd.DataFrame(ndsi_means, columns=ndsi_name_list),
        ],
        axis=1,
    )

    export_df = df.join(ndsi_df.set_index("plot_id"), on="plot_id")

    export_df.to_csv(os.path.join(params["data_export_path"], "df.csv"))

    # Deep Learning image export
    # For deep learning, we need images. The exact format of those images is determined by the model and type of deep learning you're doing. I don't know that yet. So this notebook ends here.


def main():
    script = sys.argv[0]
    filename = sys.argv[1]
    data = numpy.loadtxt(filename, delimiter=",")

    for row_mean in numpy.mean(data, axis=1):
        print(row_mean)


if __name__ == "__main__":
    main()
