import yaml
from dataclasses import dataclass, asdict, field
from datetime import date
from pathlib import Path
import os


@dataclass
class Config:
    """creates a config object that has all the configuration parameters for the field of interest"""

    project_stub: str = field(default=None)
    data_acquisition_date: str = field(default=None)
    NDVI_TH: float = field(default=None)
    field_dim: tuple[int, int] = field(default=None)
    field_origin: tuple[int, int] = field(default=None)
    field_angle: float = field(default=None)
    plot_shape: tuple[int, int] = field(default=None)
    edge_buf: int = field(default=None)
    num_ranges: int = field(default=None)
    plot_offset: tuple[int, int] = field(default=None)
    image_format: str = field(default="*.tif")
    camera: str = field(default="rededge_mx_band_wl")
    camera_wl: dict = field(default=None)
    plot_export_path: str = field(default=None)
    ground_truth_path: str = field(default=None)
    data_import_path: str = field(default=None)
    data_export_path: str = field(default=None)
    load_from_config: bool = field(default=False, compare=False)
    config_file: str = field(default=None, compare=False)

    def __post_init__(self):
        """ use the provided attributes to create the rest of the config object """
        if self.load_from_config:
            self.import_config(
                config_path=Path.cwd() / "config" / self.config_file
            )
        else:
            self.camera_wl = self.import_camera_dict()
            self.plot_export_path = Path.cwd() / "data" / "image_export"
            self.ground_truth_path = (
                Path.cwd() / "data" / "raw" / "ground_truth.csv"
            )
            self.data_import_path = (
                Path.cwd() / "data" / "raw" / {self.data_acquisition_date}
            )
            self.data_export_path = (
                Path.cwd() / "data" / "processed" / {self.data_acquisition_date}
            )
            self.config_file = (
                f"{self.project_stub}_{self.data_acquisition_date}_config.yaml"
            )

    def export_config(self):
        """ convert the self to dict and dump to yaml """
        export_config_path = Path.cwd() / "config" / self.config_file

        with open(export_config_path, "w",) as file:
            # print(yaml.dump(asdict(self), file))
            documents = yaml.dump(asdict(self), file)

    def import_camera_dict(self):
        camera_dict_path = Path.cwd() / "config" / f"{self.camera}.yaml"
        with open(camera_dict_path, "r") as stream:
            try:
                imported_config = yaml.load(stream, Loader=yaml.FullLoader)

            except yaml.YAMLError as exc:
                print(exc)
        return imported_config

    def import_config(self, config_path):
        """ will load dict from yaml file, then set config attributes to match key/values"""
        with open(config_path, "r") as stream:
            try:
                imported_config = yaml.load(stream, Loader=yaml.FullLoader)

                for key in imported_config:
                    setattr(self, key, imported_config[key])

            except yaml.YAMLError as exc:
                print(exc)


class CreateConfigurationYAMLExamples:
    def __init__(self):
        config = Config(
            working_dir=os.getcwd(),
            project_stub="Potato_Fertilizer_Othello",
            data_acquisition_date="Jun22_2020",
            NDVI_TH=0.3,
            field_origin=(2400, 1460),  # (y,x)
            field_dim=(7400, 1590),  # (y,x)
            field_angle=182.4,
            plot_shape=(200, 492),  # (x,y) TODO: change the order below
            edge_buf=40,
            num_ranges=13,
            plot_offset=(-50, 0),  # (y, x) begin plot offset from cropped image
        )
        config.export_config()

        config = Config(
            working_dir=os.getcwd(),
            project_stub="Potato_Fertilizer_Othello",
            data_acquisition_date="Jul08_2020",
            NDVI_TH=0.3,
            field_origin=(1350, 1445),  # (y,x)
            field_dim=(8750, 3025),  # (y,x)
            field_angle=182.4,
            plot_shape=(200, 492),  # (x,y) TODO: change the order below
            edge_buf=40,
            num_ranges=13,
            plot_offset=(-20, 6),  # (y, x) begin plot offset from cropped image
        )
        config.export_config()

        config = Config(
            working_dir=os.getcwd(),
            project_stub="Potato_Fertilizer_Othello",
            data_acquisition_date="Jul21_2020",
            NDVI_TH=0.3,
            field_origin=(1350, 1445),  # (y,x)
            field_dim=(8750, 3025),  # (y,x)
            field_angle=182.4,
            plot_shape=(200, 492),  # (x,y) TODO: change the order below
            edge_buf=40,
            num_ranges=13,
            plot_offset=(-20, 6),  # (y, x) begin plot offset from cropped image
        )
        config.export_config()
