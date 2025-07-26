import torch
import torch.nn as nn

class SuperSimpleTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        time_input_dim,
        dim_bow_tie,
        d_model,
        n_heads,
        anomaly_attention_n_heads,
        atttention_d_model,
        dim_feedforward,
        n_blocks,
        head_layers_dims,
        n_out_features
    ):
        super().__init__()

        self.bnorm = nn.LazyBatchNorm1d()

        self.time_encoding = nn.Linear(time_input_dim, d_model)
        self.features_encoding = nn.Linear(input_dim, d_model)

        self.days_to_predict = nn.Parameter(torch.randn(1, 72, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, n_blocks
        )

        temp_layers = [(nn.Linear(d1, d2), nn.ReLU()) for d1, d2 in zip(head_layers_dims[:-1], head_layers_dims[1:])]

        self.temp_head = nn.Sequential(
            nn.Linear(d_model, head_layers_dims[0]),
            nn.ReLU(),
            *[layer for pair in temp_layers for layer in pair],
            nn.Linear(head_layers_dims[-1], n_out_features)
        )

        pres_layers = [(nn.Linear(d1, d2), nn.ReLU()) for d1, d2 in zip(head_layers_dims[:-1], head_layers_dims[1:])]

        self.pres_head = nn.Sequential(
            nn.Linear(d_model, head_layers_dims[0]),
            nn.ReLU(),
            *[layer for pair in pres_layers for layer in pair],
            nn.Linear(head_layers_dims[-1], n_out_features)
        )

        hum_layers = [(nn.Linear(d1, d2), nn.ReLU()) for d1, d2 in zip(head_layers_dims[:-1], head_layers_dims[1:])]

        self.hum_head = nn.Sequential(
            nn.Linear(d_model, head_layers_dims[0]),
            nn.ReLU(),
            *[layer for pair in hum_layers for layer in pair],
            nn.Linear(head_layers_dims[-1], n_out_features)
        )

        speed_layers = [(nn.Linear(d1, d2), nn.ReLU()) for d1, d2 in zip(head_layers_dims[:-1], head_layers_dims[1:])]

        self.speed_head = nn.Sequential(
            nn.Linear(d_model, head_layers_dims[0]),
            nn.ReLU(),
            *[layer for pair in speed_layers for layer in pair],
            nn.Linear(head_layers_dims[-1], n_out_features)
        )

        direct_layers = [(nn.Linear(d1, d2), nn.ReLU()) for d1, d2 in zip(head_layers_dims[:-1], head_layers_dims[1:])]

        self.direct_head = nn.Sequential(
            nn.Linear(d_model, head_layers_dims[0]),
            nn.ReLU(),
            *[layer for pair in direct_layers for layer in pair],
            nn.Linear(head_layers_dims[-1], n_out_features)
        )

    def __call__(self, batch_time, batch_features):
        batch = self.features_encoding(batch_features)

        time_encoding = self.time_encoding(batch_time)
        batch += time_encoding

        batch = torch.cat([batch, self.days_to_predict.expand(batch.shape[0], -1, -1)], dim=-2)
        batch = self.transformer_encoder(batch)[:, -72:, :]

        temp = self.temp_head(batch)
        pres = self.pres_head(batch)
        hum = self.hum_head(batch)
        batch = torch.cat([temp, pres, hum], dim=-1)

        return batch


class SuperSimpleTransformer2(nn.Module):
    def __init__(
        self,
        input_dim,
        time_input_dim,
        dim_bow_tie,
        d_model,
        n_heads,
        anomaly_attention_n_heads,
        atttention_d_model,
        dim_feedforward,
        n_blocks,
        head_layers_dims,
        n_out_features
    ):
        super().__init__()

        self.bnorm = nn.LazyBatchNorm1d()

        self.time_encoding = nn.Linear(time_input_dim, d_model)
        self.features_encoding = nn.Linear(input_dim, d_model)

        self.days_to_predict = nn.Parameter(torch.randn(1, 72, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, n_blocks
        )

        speed_layers = [(nn.Linear(d1, d2), nn.ReLU()) for d1, d2 in zip(head_layers_dims[:-1], head_layers_dims[1:])]

        self.speed_head = nn.Sequential(
            nn.Linear(d_model, head_layers_dims[0]),
            nn.ReLU(),
            *[layer for pair in speed_layers for layer in pair],
            nn.Linear(head_layers_dims[-1], n_out_features)
        )

        direct_layers = [(nn.Linear(d1, d2), nn.ReLU()) for d1, d2 in zip(head_layers_dims[:-1], head_layers_dims[1:])]

        self.direct_head = nn.Sequential(
            nn.Linear(d_model, head_layers_dims[0]),
            nn.ReLU(),
            *[layer for pair in direct_layers for layer in pair],
            nn.Linear(head_layers_dims[-1], n_out_features)
        )

    def __call__(self, batch_time, batch_features):
        batch = self.features_encoding(batch_features)

        time_encoding = self.time_encoding(batch_time)
        batch += time_encoding

        batch = torch.cat([batch, self.days_to_predict.expand(batch.shape[0], -1, -1)], dim=-2)
        batch = self.transformer_encoder(batch)[:, -72:, :]

        speed = self.speed_head(batch)
        direct = self.direct_head(batch)

        batch = torch.cat([speed, direct], dim=-1)

        return batch


class Model():
    def __init__(self, wind_tr_path, meteo_tr_path):
        self.wind_transformer = torch.load(wind_tr_path, weights_only=False, map_location=torch.device('cpu'))
        self.meteo_transformer = torch.load(meteo_tr_path, weights_only=False, map_location=torch.device('cpu'))

    def __call__(self, time, data_wind, data_meteo):
        wind_preds = self.wind_transformer(time, data_wind)
        meteo_preds = self.meteo_transformer(time, data_meteo)
        return torch.cat([meteo_preds, wind_preds], dim=-1)