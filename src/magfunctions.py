import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from itertools import combinations
from itertools import product
import matplotlib.patches as patches
from PIL import Image
import os
import cv2 as cv
import glob


class PlotCoordinates:
    """contains functions related to calculation of roi/plot coordinates"""

    def __init__(self):
        pass

    def get_roi_shape(
        self, plot_shape: tuple, edge_buf: int
    ) -> tuple[int, int]:
        """takes the plot_shape tuple and edge buffer then returns the inset roi_shape tuple"""
        return (
            int(plot_shape[0] - edge_buf),
            int(plot_shape[1] - 4 * edge_buf),
        )

    def get_roi_coord(
        self, plot_coord: tuple, edge_buf: int
    ) -> tuple[int, int]:
        """takes the plot origin tuple and edge buffer then returns the inset roi origin tuple"""
        return (
            int(plot_coord[0] + 0.5 * edge_buf),
            int(plot_coord[1] + 2 * edge_buf),
        )

    def plot_boundaries(
        image_stack: np.array,
        plot_coords,
        roi_coords,
        plot_shape,
        roi_shape,
    ):
        """creates pyplot figure with plot boundaries for visual verification"""
        img = image_stack[38]
        figure, ax = plt.subplots(1, figsize=(6, 20))
        ax.imshow(img, cmap="gray")
        ax.set_title("plot boundaries (red), plot roi (green)")
        plt.gca().legend(("plot border", "plot region of interest"))

        for plot_id, (plot_coord, roi_coord) in enumerate(
            zip(plot_coords, roi_coords)
        ):

            roi_x, roi_y = roi_coord

            plot_boundary = patches.Rectangle(
                xy=plot_coord,
                width=plot_shape[0],
                height=plot_shape[1],
                edgecolor="r",
                lw=2,
                facecolor="r",
                alpha=0.1,
            )
            ax.add_patch(plot_boundary)

            plot_subsection = patches.Rectangle(
                xy=roi_coord,
                width=roi_shape[0],
                height=roi_shape[1],
                edgecolor="None",
                facecolor="green",
                alpha=0.4,
            )
            ax.add_patch(plot_subsection)

            plt.scatter(
                x=roi_x,
                y=roi_y,
                c="red",
                marker="o",
            )

            ax.text(
                x=roi_x + 0.27 * roi_shape[0],
                y=roi_y + 0.5 * roi_shape[1],
                s=plot_id,
                c="magenta",
            )
            print(f"plot: {plot_id}, roi_origin: {roi_coord}")


class ImageProcessing:
    def __init__(self, params):
        self.params = params
        pass

    def save_idx_img(self, array, plot, index):
        """takes np.array and converts to a PIL.Image, then saves it into the data_export_path"""
        im = Image.fromarray(array)
        im.save(
            self.params["data_export_path"]
            + "plot_"
            + str(plot)
            + "_index_"
            + index
            + ".png"
        )

    def load_img(self, channel_name: str):
        """open image file, replace all '-10000' transparent values with zero, and return"""
        image_path = glob.glob(
            os.path.join(
                self.params["data_import_path"], f"*{channel_name}.tif"
            )
        )[0]

        image = cv.imread(image_path, cv.IMREAD_UNCHANGED)

        out_image = np.where(
            image < 0, 0.0, image
        )  # gets rid of -10000 transparency
        return out_image

        # scaled_image = np.multiply(out_image, 255.0).astype(np.float32)

        # print(f"importing {channel_name}... done. dtype: {out_image.dtype}")
        # return scaled_image

    def crop_image(self, image: np.array, crop_percent: float):
        """takes image: np.array, and crop_percent: float, return a center cropped np.array"""
        h, w = image.shape
        h0 = int(h * (1 - crop_percent))
        h1 = int(h * crop_percent)
        w0 = int(w * (1 - crop_percent))
        w1 = int(w * crop_percent)
        print(h0, h1, w0, w1)
        return image[h0:h1, w0:w1]

    def show_image(self, image, size=(8, 30)):
        """plot array as img"""
        plt.figure(figsize=size)
        plt.imshow(image, cmap="viridis")

    def get_channel_names(self, path_list: list) -> list:
        """gets the channel name from the file path"""
        return [
            os.path.split(path)[1].split("_")[-1].split(".")[0]
            for path in path_list
        ]

    def calc_spec_idx(self, combo: tuple[int, int], bands: np.array):
        """calculates spectral index from channel nums of np.array
        NDSI = (band[0] - band[1]) / (band[0] + band[1])
        This function avoids divide by zero error."""
        band_a = bands[combo[0]]
        band_b = bands[combo[1]]

        numer = np.subtract(band_a, band_b)
        denom = np.add(band_a, band_b)
        return np.divide(
            numer, denom, out=np.zeros_like(numer), where=(denom != 0)
        )


# def print_range(arr_list: list):
#     s_out = ""
#     for arr in arr_list:
#         s_out += f"({np.min(arr)}, {np.max(arr)}), "
#     print(s_out)

# print_range([img, img_rotate, img_crop1, img_crop2])

# 4. Display the images as one figure
