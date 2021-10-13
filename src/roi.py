import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


class PlotCoordinates:
    """contains functions related to calculation of roi/plot coordinates"""

    def __init__(self):
        pass

    def get_roi_shape(
        self, plot_shape: tuple, edge_buf: int
    ) -> tuple[int, int]:
        """takes the plot_shape tuple and edge buffer then returns the inset roi_shape tuple
        (x, y)"""
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
        self,
        img: np.array,
        plot_coords: list,
        roi_coords: list,
        plot_shape: tuple[tuple, tuple],
        roi_shape: tuple[int, int],
    ) -> list:
        """creates pyplot figure with plot boundaries for visual verification"""

        plot_id_list = []
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
            # print(f"plot: {plot_id}, roi_origin: {roi_coord}")

            plot_id_list.append(plot_id)
        return plot_id_list
