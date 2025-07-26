import numpy as np
import pandas as pd

def rain_proba(row) -> float:
        '''
        Функция для подсчета вероятности дождя
        '''

        score = 0
        if row['Влажность, %'] > 92.5: score += 0.4
        elif row['Влажность, %'] > 90: score += 0.3
        elif row['Влажность, %'] > 85: score += 0.2
        elif row['Влажность, %'] > 80: score += 0.1

        if abs(row['Температура, °С'] - row['dew_point']) < 1:
            score += 0.2
        elif abs(row['Температура, °С'] - row['dew_point']) < 2:
            score += 0.1

        if row['Давление, мм рт. ст.'] < 750: score += 0.15
        elif row['Давление, мм рт. ст.'] < 760: score += 0.1

        if row['dispersion_potential'] < 1.2: score += 0.15
        elif row['dispersion_potential'] < 2.1: score += 0.1

        if row['total_pollution'] > 1250: score -= 0.2
        elif row['total_pollution'] > 600: score -= 0.1

        return max(0, min(score, 1))

def preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    data['pressure_hpa'] = data['Давление, мм рт. ст.'] * 1.333
    data['e_s'] = 6.11 * 10 ** (7.5 * data['Температура, °С'] / (237.7 + data['Температура, °С']))
    data['e'] = data['Влажность, %'] / 100 * data['e_s']

    data['dew_point'] = 237.7 * np.log(data['e'] / 6.11) / (7.5 - np.log(data['e'] / 6.11))
    data['humidex'] = data['Температура, °С'] + 0.5555 * (data['e'] - 10)
    data['temp_f'] = data['Температура, °С'] * 9/5 + 32

    data['heat_index_f'] = (
        -42.379 + 2.04901523 * data['temp_f'] + 10.14333127 * data['Влажность, %']
        - 0.22475541 * data['temp_f'] * data['Влажность, %']
        - 6.83783e-3 * data['temp_f']**2 - 5.481717e-2 * data['Влажность, %']**2
        + 1.22874e-3 * data['temp_f']**2 * data['Влажность, %']
        + 8.5282e-4 * data['temp_f'] * data['Влажность, %']**2
        - 1.99e-6 * data['temp_f']**2 * data['Влажность, %']**2
    )

    data['heat_index_c'] = (data['heat_index_f'] - 32) * 5 / 9
    data = data.drop(columns=['pressure_hpa', 'temp_f', 'heat_index_f'])

    data['Ощущаемая_температура'] = (
        -2.7 + 1.04 * data['Температура, °С'] + 2.0 * (data['e'] / 10)
        - 0.65 * data['Скорость ветра, м/с']
    )

    data['dispersion_potential'] = data['Скорость ветра, м/с'] * np.cos(np.radians(data['Направление ветра, °']))
    data['total_pollution'] = data['NO2'] + data['O3'] + data['H2S'] + data['CO'] + data['SO2']

    data['rain_probability'] = data.apply(rain_proba, axis=1)

    return data
