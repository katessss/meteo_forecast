from inference import results_preprocessing
from torch_utils import WeatherDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import mean_absolute_percentage_error


def post_training(data, model, device, optimizer, epochs, loss_fn=nn.MSELoss()):
    model = torch.load('model.pth', weights_only=False, map_location=torch.device('cpu'))

    mean_global = torch.Tensor([ 4.3823e+01,  2.1919e+01,  5.0122e-01,  3.8614e+02,  7.2370e-01,
         4.8215e+00,  7.5505e+02,  6.5823e+01,  7.2626e-01,  1.9657e+02,
         9.8655e+00,  5.8549e+00, -3.6346e+00,  2.5189e+00,  5.8823e+01,
         3.0133e+00, -2.7793e-01,  4.5310e+02,  3.0537e-01,  4.0661e-02,
         2.0661e+00,  1.1374e-02,  1.3287e+00,  2.2881e-02,  1.0722e+00,
         1.1065e+01,  6.1553e-03,  2.9058e+00,  1.3441e-02,  1.4021e+00,
         2.2021e-02,  7.7033e-01,  3.3416e-02,  7.3167e+00,  2.0553e-02,
         4.7305e+00,  2.4195e-02,  3.6044e+00,  3.2885e-02,  2.5822e-01,
         2.6400e-02,  1.9170e-01,  4.3531e-02,  1.5041e-01])
    
    std_global = torch.Tensor([1.1318e+01, 1.2209e+01, 8.0821e-01, 1.1654e+02, 1.9080e+00, 7.9652e+00,
        7.1513e+00, 1.9058e+01, 7.9581e-01, 7.5549e+01, 5.8727e+00, 2.6049e+00,
        1.4003e+01, 9.1248e+00, 3.1930e+01, 8.5548e+00, 6.9304e-01, 1.1553e+02,
        1.4414e-01, 1.4387e-02, 7.3274e-01, 1.2159e-02, 3.3691e-01, 1.4902e-02,
        3.4913e-01, 5.1375e+00, 1.0802e-03, 1.1252e+00, 3.0333e-03, 5.1021e-01,
        5.8717e-03, 2.9403e-01, 1.8157e-02, 2.3287e+00, 1.5006e-02, 1.7912e+00,
        1.2269e-02, 9.9224e-01, 1.8295e-02, 1.0020e-01, 2.6542e-02, 8.3123e-02,
        1.5128e-02, 6.2076e-02])

    y_mean_global = torch.Tensor([4.8215e+00, 7.5505e+02, 6.5823e+01, 7.2626e-01, 1.9657e+02])
    y_std_global = torch.Tensor([ 7.9652,  7.1513, 19.0576,  0.7958, 75.5488])

    data = results_preprocessing(data, is_Fourie=True)
    data = WeatherDataset(data)
    data.mean, data.std, data.y_mean, data.y_std = mean_global, std_global, y_mean_global, y_std_global
    data = DataLoader(data)

    model.to(device)
    last_mape = 0

    for _ in range(epochs):
        epoch_mapes = []

        for batch in data:
            time = batch["time"].to(device)
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            output = model(time, x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            epoch_mapes.append(mean_absolute_percentage_error(
                y.detach().cpu().numpy(),
                output.detach().cpu().numpy()
            ))

        last_mape = epoch_mapes / len(epoch_mapes)

    return model, f'last epoch mape: {last_mape}'
