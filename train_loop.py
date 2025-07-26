import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from tqdm import tqdm
from math import sqrt
import torch
import torch.nn as nn
from typing import Any, Dict, List


def plot_feature_metrics_and_loss(y_true: pd.Series | np.ndarray, 
                                  y_pred: pd.Series | np.ndarray, 
                                  feature_names: List[str],
                                  return_metrics: bool = False) -> Dict[str, float] | None:
    metrics = {}

    for i, feat in enumerate(feature_names):
        true_values = y_true[:, i]
        pred_values = y_pred[:, i]
        losses = true_values - pred_values

        # График предсказания vs. истинные значения
        plt.figure(figsize=(10, 4))
        plt.plot(true_values, label='True', alpha=0.7)
        plt.plot(pred_values, label='Predicted', alpha=0.7)
        plt.title(f"{feat} - True vs Predicted")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()

        # График остатков
        plt.figure(figsize=(10, 4))
        plt.plot(losses, label='Loss', color='red')
        plt.axhline(0, linestyle='--', color='black')
        plt.title(f"{feat} - Loss")
        plt.xlabel("Time")
        plt.ylabel("Error")
        plt.grid(True)  
        plt.tight_layout()
        plt.show()
        plt.close()

        # метрики
        mae = mean_absolute_error(true_values, pred_values)
        mse = mean_squared_error(true_values, pred_values)
        rmse = sqrt(mean_squared_error(true_values, pred_values))
        r2 = r2_score(true_values, pred_values)
        mape = mean_absolute_percentage_error(true_values, pred_values)

        if return_metrics:
            metrics[feat] = {
                "MAE": round(mae, 4),
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "R2": round(r2, 4),
                "MAPE": round(mape, 4)
            }

    for name, m in metrics.items():
        print(f"{name}: MAE={m['MAE']}, MSE={m['MSE']}, RMSE={m['RMSE']}, R2={m['R2']}, MAPE={m['MAPE']}")

    return metrics


def train_loop(model: torch.nn.Module, device: str, train_loader: torch.utils.data.DataLoader, 
               val_loader: torch.utils.data.DataLoader, optimizer: torch.optim, epochs: int = 20, loss_fn: Any = nn.MSELoss(), 
               plot_epoch_loss: bool = True, metrics_and_grafics: bool = True, cols: List[str] = None, return_metrics: bool = False) -> pd.DataFrame | None:
    # задаем девайс, если требуется
    if not device:
        device = 'cpu'

    model.to(device)
    losses = []
    val_losses = []

    # тренировочный цикл
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for batch in (bar := tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            time = batch["time"].to(device)
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            output = model(time, x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            bar.set_description(f'Epoch: {epoch+1}/{epochs} loss: {sqrt(loss.item()):.4f}')
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        model.eval()
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            epoch_val_loss = 0
            for batch in tqdm(val_loader):
                time_val = batch["time"].to(device)
                x_val = batch["x"].to(device)
                y_val = batch["y"].to(device)
                output_val = model(time_val, x_val)
                val_loss = loss_fn(output_val, y_val)
                epoch_val_loss += val_loss.item()

                all_preds.append(output_val)
                all_trues.append(y_val)


        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        model.train()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Val_loss: {avg_val_loss}")


    if plot_epoch_loss:
        plt.plot(val_losses, marker='o')
        plt.title("Loss по эпохам")
        plt.xlabel("Эпоха")
        plt.ylabel("Validation Loss")
        plt.grid(True)
        plt.show()
        plt.close()
    
    if metrics_and_grafics and cols is not None:
        y_pred = torch.cat([p[:, -1, :] for p in all_preds], dim=0).cpu().numpy()
        y_true = torch.cat([t[:, -1, :] for t in all_trues], dim=0).cpu().numpy()

        plot_feature_metrics_and_loss(
            y_true=y_true,
            y_pred=y_pred,
            feature_names=cols
        )

    if return_metrics:
        y_pred = torch.cat(all_preds).detach().cpu().numpy()
        y_true = torch.cat(all_trues).detach().cpu().numpy()

        # формируем метрики по строкам
        mae = [mean_absolute_error(y_true[:, i, :].reshape(-1, 5), y_pred[:, i, :].reshape(-1, 5)) for i in range(72)]
        mse = [mean_squared_error(y_true[:, i, :].reshape(-1, 5), y_pred[:, i, :].reshape(-1, 5)) for i in range(72)]
        rmse = [sqrt(mean_squared_error(y_true[:, i, :].reshape(-1, 5), y_pred[:, i, :].reshape(-1, 5))) for i in range(72)]
        r2 = [r2_score(y_true[:, i, :].reshape(-1, 5), y_pred[:, i, :].reshape(-1, 5)) for i in range(72)]
        mape = [mean_absolute_percentage_error(y_true[:, i, :].reshape(-1, 5), y_pred[:, i, :].reshape(-1, 5)) for i in range(72)]

        item_metrix = pd.DataFrame({
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        })

        return item_metrix
