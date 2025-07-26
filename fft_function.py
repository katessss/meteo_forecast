import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode


def analyze_fft_spectrum(signal: np.ndarray, sampling_interval: float, column: str) -> None:
    """
    Выполняет анализ спектра Фурье для временного ряда, визуализирует амплитудный спектр и возвращает топ частот.

    Parameters:
    - signal: np.ndarray — одномерный временной ряд
    - sampling_interval: float — шаг между измерениями (например, 1 день)
    - column: str — имя признака для отображения на графике
    """
    # сделаем преобразование Фурье
    N = len(signal)
    fft_result = np.fft.fft(signal)
    amplitudes = np.abs(fft_result) / N
    frequencies = np.fft.fftfreq(N, d=sampling_interval)

    # Позитивная часть спектра
    pos_mask = frequencies > 0
    pos_freqs = frequencies[pos_mask]
    pos_amplitudes = amplitudes[pos_mask]

    # Визуализация
    plt.figure(figsize=(12, 5))
    plt.plot(pos_freqs, pos_amplitudes)
    plt.title(f"Анализ Фурье: {column}")
    plt.xlabel("Частота (циклы в единицу времени)")
    plt.ylabel("Амплитуда")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def analyze_fft_by_sliding_window(data: pd.DataFrame, column: str, sampling_interval: float = 0.3, window_days: int = 30, stride: int = 30) -> None:
    """
    Выполняет скользящий анализ Фурье по временным окнам длиной в window_days.

    Parameters:
    - data: DataFrame с колонкой column
    - column: название колонки с временным рядом 
    - sampling_interval: шаг между измерениями, в часах
    - window_days: размер окна в днях (по умолчанию 30)
    - stride: временной шаг в днях
    """
     
    points_per_day = int(24 / sampling_interval)
    window_size = int(window_days * points_per_day)
    stride_points = int(stride * points_per_day)


    signal = data[column]
    for start in range(0, len(signal) - window_size, stride_points):
        segment = signal[start:start + window_size]

        if len(segment) >= 3 * 24 * (1 / sampling_interval): # хотя бы 3 суток для анализа
            analyze_fft_spectrum(
                signal=segment,
                sampling_interval=sampling_interval,
                column=f"{column} окно {window_days} дней"
            )


def extract_fft_features_by_window(
    data: pd.DataFrame,
    column: str,
    time_column: str,
    sampling_interval: float = 0.3,
    window_days: int = 7,
    top_k: int = 3,
    fft_freq: bool = True,
    fft_amp: bool = True,
    total_power: bool = True,
) -> pd.DataFrame:
    """
    Извлекает FFT-признаки по окнам и заполняет ими каждую строку в пределах окна (без NaN).

    Parameters:
    - data: DataFrame с временным рядом
    - column: имя числовой колонки (например, "Температура, °С")
    - time_column: имя колонки со временем (datetime)
    - sampling_interval: шаг между измерениями (в часах)
    - window_days: размер окна в днях
    - top_k: сколько частот извлекать
    - fft_freq, fft_amp, total_power: какие признаки извлекать
    - merge_back: если True — объединяет признаки с оригинальным DataFrame
    """

    df = data.copy()
    df[time_column] = pd.to_datetime(df[time_column])
    points_per_day = int(24 / sampling_interval)
    window_size = int(window_days * points_per_day)

    # Инициализируем фичи
    for i in range(top_k):
        if fft_freq:
            df[f"fft_freq_{column}_{i+1}"] = np.nan
        if fft_amp:
            df[f"fft_amp_{column}_{i+1}"] = np.nan
    if total_power:
        df["fft_power_total_{column}"] = np.nan

    series = df[column].astype(float).values

    for i in range(0, len(series) - window_size + 1, window_size):
        segment = series[i:i + window_size]

        if len(segment) < window_size:
            continue

        N = len(segment)
        fft_result = np.fft.fft(segment)
        amplitudes = np.abs(fft_result) / N
        frequencies = np.fft.fftfreq(N, d=sampling_interval)

        pos_mask = frequencies > 0
        pos_freqs = frequencies[pos_mask]
        pos_amplitudes = amplitudes[pos_mask]

        top_idx = np.argsort(pos_amplitudes)[-top_k:][::-1]
        top_freqs = pos_freqs[top_idx]
        top_amps = pos_amplitudes[top_idx]

        for j in range(top_k):
            if fft_freq:
                df.loc[i:i + window_size - 1, f"fft_freq_{column}_{j+1}"] = top_freqs[j] #if j < len(top_freqs) else 0
            if fft_amp:
                df.loc[i:i + window_size - 1, f"fft_amp_{column}_{j+1}"] = top_amps[j] #if j < len(top_amps) else 0

        if total_power:
            total = np.sum(pos_amplitudes**2)
            df.loc[i:i + window_size - 1, "fft_power_total_{column}"] = total


     # Заполнение NaN модой, иначе нулём
        window_slice = df.iloc[i:i + len(segment)]
        for col in df.columns:
            if col.startswith("fft_") and window_slice[col].isna().any():
                m = mode(window_slice[col].dropna(), keepdims=False)
                mode_val = np.atleast_1d(m.mode)[0]  # безопасно получаем значение
                if not pd.isna(mode_val):
                        value_to_fill = float(mode_val)
                        df.loc[i:i + len(segment) - 1, col] = df.loc[i:i + len(segment) - 1, col].fillna(value_to_fill)
                else:
                        df.loc[i:i + len(segment) - 1, col] = df.loc[i:i + len(segment) - 1, col].fillna(0)
    df = df.fillna(0)

    return df
