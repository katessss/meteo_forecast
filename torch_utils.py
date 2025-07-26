from torch.utils.data import Dataset
import pandas as pd
import torch
import torch.nn.functional as F


class WeatherDataset1(Dataset):
    MONTHS, DAYS, HOURS, MINUTES = 12, 31, 24, 60

    def init(self, data: pd.DataFrame):
        self.data = data.copy()
        dt = pd.to_datetime(self.data["Период"])
        self.data["month"] = dt.dt.month
        self.data["day"] = dt.dt.day
        self.data["hour"] = dt.dt.hour
        self.data["minute"] = dt.dt.minute

        self.feature_cols = [
            c for c in self.data.select_dtypes("number").columns
            if c not in ("month", "day", "hour", "minute")
        ]
        self.pred_cols = self.data.columns[7:10]
        
        self.num_feats = len(self.feature_cols)

        feats = self.data[self.pred_cols].fillna(0).values
        
        self.mean = torch.tensor(feats.mean(0), dtype=torch.float32)
        self.std = torch.tensor(feats.std(0) + 1e-8, dtype=torch.float32)

    def len(self):
        return len(self.data) - 72 * 1 - 72

    def _one_hot_time(self, row: pd.Series) -> torch.Tensor:
        month = F.one_hot(torch.tensor((row["month"] - 1).to_list()).long(), self.MONTHS)
        day = F.one_hot(torch.tensor((row["day"] - 1).to_list()).long(), self.DAYS)
        hour = F.one_hot(torch.tensor((row["hour"]).to_list()).long(), self.HOURS)
        minute = F.one_hot(torch.tensor((row["minute"]).to_list()).long(), self.MINUTES)
        return torch.cat([month, day, hour, minute], dim=-1).float()

    def getitem(self, idx):
        row_x = self.data.iloc[idx: idx + 72 * 1]
        row_y = self.data.iloc[idx + 72 * 1: idx + 72 * 1 + 72][self.pred_cols]

        x_feats = torch.tensor(row_x[self.pred_cols].fillna(0).to_numpy(), dtype=torch.float32)
        y_feats = torch.tensor(row_y.fillna(0).to_numpy(), dtype=torch.float32)

        x_feats = (x_feats - self.mean) / self.std
        y_feats = (y_feats - self.mean) / self.std

        time_oh = self._one_hot_time(row_x)

        return {
            "time": time_oh,
            "x": torch.cat([x_feats, time_oh], dim=-1),
            "y": y_feats
        }


class WeatherDataset2(Dataset):
    MONTHS, DAYS, HOURS, MINUTES = 12, 31, 24, 60

    def init(self, data: pd.DataFrame):
        self.data = data.copy()
        dt = pd.to_datetime(self.data["Период"])
        self.data["month"] = dt.dt.month
        self.data["day"] = dt.dt.day
        self.data["hour"] = dt.dt.hour
        self.data["minute"] = dt.dt.minute

        self.feature_cols = [
            c for c in self.data.select_dtypes("number").columns
            if c not in ("month", "day", "hour", "minute")
        ]
        self.pred_cols = self.data.columns[10:12]
        
        self.num_feats = len(self.feature_cols)

        feats = self.data[self.pred_cols].fillna(0).values
        
        self.mean = torch.tensor(feats.mean(0), dtype=torch.float32)
        self.std = torch.tensor(feats.std(0) + 1e-8, dtype=torch.float32)

    def len(self):
        return len(self.data) - 72 * 1 - 72

    def _one_hot_time(self, row: pd.Series) -> torch.Tensor:
        month = F.one_hot(torch.tensor((row["month"] - 1).to_list()).long(), self.MONTHS)
        day = F.one_hot(torch.tensor((row["day"] - 1).to_list()).long(), self.DAYS)
        hour = F.one_hot(torch.tensor((row["hour"]).to_list()).long(), self.HOURS)
        minute = F.one_hot(torch.tensor((row["minute"]).to_list()).long(), self.MINUTES)
        return torch.cat([month, day, hour, minute], dim=-1).float()

    def getitem(self, idx):
        row_x = self.data.iloc[idx: idx + 72 * 1]
        row_y = self.data.iloc[idx + 72 * 1: idx + 72 * 1 + 72][self.pred_cols]

        x_feats = torch.tensor(row_x[self.pred_cols].fillna(0).to_numpy(), dtype=torch.float32)
        y_feats = torch.tensor(row_y.fillna(0).to_numpy(), dtype=torch.float32)

        x_feats = (x_feats - self.mean) / self.std
        y_feats = (y_feats - self.mean) / self.std
        time_oh = self._one_hot_time(row_x)
        return {
            "time": time_oh,
            "x": torch.cat([x_feats, time_oh], dim=-1),
            "y": y_feats
        }

class WeatherDatasetInference(Dataset):
    MONTHS, DAYS, HOURS, MINUTES = 12, 31, 24, 60

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        dt = pd.to_datetime(self.data["Период"])
        self.data["month"] = dt.dt.month
        self.data["day"] = dt.dt.day
        self.data["hour"] = dt.dt.hour
        self.data["minute"] = dt.dt.minute

        
        self.pred_cols1 = self.data.columns[7:10]
        self.pred_cols2 = self.data.columns[10:12]
        
        feats = self.data[self.data.columns[7:12]].fillna(0).values
        
        self.mean = torch.tensor(feats.mean(0), dtype=torch.float32)
        self.std = torch.tensor(feats.std(0) + 1e-8, dtype=torch.float32)

    def __len__(self):
        return len(self.data) - 71

    def _one_hot_time(self, row: pd.Series) -> torch.Tensor:
        month = F.one_hot(torch.tensor((row["month"] - 1).to_list()).long(), self.MONTHS)
        day = F.one_hot(torch.tensor((row["day"] - 1).to_list()).long(), self.DAYS)
        hour = F.one_hot(torch.tensor((row["hour"]).to_list()).long(), self.HOURS)
        minute = F.one_hot(torch.tensor((row["minute"]).to_list()).long(), self.MINUTES)
        return torch.cat([month, day, hour, minute], dim=-1).float()

    def __getitem__(self, idx: int):
        row_x = self.data.iloc[idx: idx + 72]

        x_feats = torch.tensor(row_x[self.pred_cols1].fillna(0).to_numpy(), dtype=torch.float32)
        x_feats = (x_feats - self.mean[:3]) / self.std[:3]

        x2_feats = torch.tensor(row_x[self.pred_cols2].fillna(0).to_numpy(), dtype=torch.float32)
        x2_feats = (x2_feats - self.mean[3:]) / self.std[3:]

        time_oh = self._one_hot_time(row_x)

        return {
            "time": time_oh,
            "x": torch.cat([x_feats, time_oh], dim=-1),
            "x2": torch.cat([x2_feats, time_oh], dim=-1),
        }
    

mean_global = torch.Tensor([6.2421, 754.4582,  64.9654, 0.7424, 197.6225])
std_global = torch.Tensor([8.0054,  6.6253, 19.3245, 0.7877, 75.2531])