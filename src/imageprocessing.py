import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import cv2 as cv
import glob
import numpy.typing as npt


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

    def ndsi_mean(
        self,
        arr: npt.NDArray,
        origin: tuple[int, int],
        shape: tuple[int, int],
        mask: npt.NDArray,
    ) -> float:
        """Return mean value for arr in the given region of interest.

        Origin and shape are (x, y), but the np.array is (y, x).

        Calculates mean using a boolean mask to exclude bg values.
        """

        roi_width, roi_height = shape
        roi_x, roi_y = origin

        return np.mean(
            a=arr[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width],
            where=mask[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width],
        )


# def print_range(arr_list: list):
#     s_out = ""
#     for arr in arr_list:
#         s_out += f"({np.min(arr)}, {np.max(arr)}), "
#     print(s_out)

# print_range([img, img_rotate, img_crop1, img_crop2])

# 4. Display the images as one figure
