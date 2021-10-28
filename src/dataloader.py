from pathlib import Path
import pandas as pd


class DataLoader:
    """ loads processed datafiles from a directory for testing validation methods"""

    def __init__(self):
        dates = ["Jun22_2020", "Jul08_2020", "Jul21_2020"]
        self.df = {date: self.get_data_frame(date) for date in dates}

    def get_data_frame(self, date):
        """ returns df without weird index column and plot_id, just yield and ndsi_mean columns """
        return pd.read_csv(self.get_data_path(date)).iloc[:, 2:]

    def get_data_path(self, datestr: str) -> Path:
        return Path.cwd() / "data" / "processed" / datestr / f"{datestr}_df.csv"

    def get_dataframe(self, df_date: str) -> pd.DataFrame:
        return self.df[df_date]


# ######################################## Data import #########################################

# # data path
# df_path = Path.cwd() / "data" / "processed" / "Jun22_2020" / "Jun22_2020_df.csv"

# # Test/Train/Val Split

# # X and y
# X = df.iloc[:, 1:].values
# y = df["yield"].values

# # train test Split (0.7/0.3)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, train_size=0.7, random_state=26, shuffle=True
# )

# # test validate split (0.3 split into 0.15/0.15)
# X_test, X_val, y_test, y_val = train_test_split(
#     X_test, y_test, train_size=0.5, random_state=26, shuffle=True
# )

# #  train the scaler ONLY on the training set. Then use it to scale train/test/val
# scaler = preprocessing.StandardScaler()
# X_train = scaler.fit_transform(
#     X_train
# )  # trains the scaler using fit on X_train, then transforms X_train as well
# X_test = scaler.transform(
#     X_test
# )  # no fit, transforms using data from fit() stored in the scaler
# X_val = scaler.transform(X_val)

# # convert variables to PyTorch tensor
# X_train = torch.tensor(X_train, dtype=torch.float32)
# X_test = torch.tensor(X_test, dtype=torch.float32)
# X_val = torch.tensor(X_val, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32)
# y_test = torch.tensor(y_test, dtype=torch.float32)
# y_val = torch.tensor(y_val, dtype=torch.float32)
# print(X_train.shape)
