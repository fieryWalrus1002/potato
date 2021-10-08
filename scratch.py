from pathlib import Path
from datetime import datetime


today = datetime.today()
sales_file = (
    Path.cwd()
    / "data"
    / "raw"
    / "Jun22_2020"
    / "Potato_Fertilizer_Othello_Jun22_M10_transparent_reflectance_blue-444.tif"
)
# summary_file = Path.cwd() / "data" / "processed" / f"summary_{today:%b-%d-%Y}.pkl"
print(sales_file)
