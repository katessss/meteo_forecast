import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from torch_utils import WeatherDatasetInference, mean_global, std_global
from model import Model


def decode_one_hot_time(one_hot_vector: torch.Tensor) -> pd.Timestamp:
    '''
    Делает разбиение временной метки на отдельные компоненты, применяет one-hot-encoding к ним
    '''
    month = torch.argmax(one_hot_vector[:12]).item() + 1
    day = torch.argmax(one_hot_vector[12:12+31]).item() + 1
    hour = torch.argmax(one_hot_vector[12+31:12+31+24]).item()
    minute = torch.argmax(one_hot_vector[12+31+24:]).item()

    return pd.Timestamp(year=2025, month=month, day=day, hour=hour, minute=minute)



def inference(data: pd.DataFrame, return_csv: bool = False):
    '''
    Полное обращение к модели: загрузка модели -> генерация признаков -> подача в модель -> получение предсказаний
    '''

    model = Model('model2.pth', 'model.pth')

    # Сделаем фичи
    data = WeatherDatasetInference(data)
    data.mean, data.std = mean_global, std_global
    data_loader = DataLoader(data)

    # Получим прдсказания
    results = []

    dataset = data_loader.dataset
    orig_cols = dataset.data.columns.tolist()
    cols_filtered = [col for col in orig_cols if col not in ['Период', 'Пост мониторинга']]
    selected_features = cols_filtered[5:10]

    with torch.no_grad():
        for batch in data_loader:
            print(batch["time"].shape, batch['x2'].shape, batch['x'].shape)
            output = model(batch["time"], batch['x2'], batch['x'])
            batch_size = output.size(0)

            for i in range(batch_size):
                y_vals = torch.reshape(output[i], (72, -1))
                y_denorm = y_vals * std_global[:y_vals.shape[1]].unsqueeze(0) + mean_global[:y_vals.shape[1]].unsqueeze(0)

                df_y = pd.DataFrame(
                    y_denorm.cpu().numpy(),
                    columns=selected_features
                )

                last_time_oh = batch['time'][i, -1]
                last_time = decode_one_hot_time(last_time_oh)
                # Генерируем временные метки для каждого шага предсказания
                time_steps = [last_time + pd.Timedelta(minutes=20 * (j + 1)) for j in range(y_vals.shape[0])]
                df_y['Период'] = time_steps
                # Перемещаем 'Период' в начало
                df_y = df_y[['Период'] + [col for col in df_y.columns if col != 'Период']]

                results.append(df_y)
                data_export = pd.concat(results, ignore_index=True)
                
    if return_csv:
        return data_export.to_csv('pred_data', index=False)
    else:
        return data_export