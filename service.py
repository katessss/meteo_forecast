import pandas as pd
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import os

from datetime import datetime
from inference import inference
from model import SuperSimpleTransformer2, SuperSimpleTransformer, Model
from typing import Tuple

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

class ClimaScopeX():

    def __init__(self):
        """
        Загружает данные из CSV-файла и выполняет предварительную обработку.

        Args:
            path (str): Путь к CSV-файлу. По умолчанию "20250520_120751.csv".

        Raises:
            FileNotFoundError: Если файл не найден.
            pd.errors.ParserError: Если произошла ошибка парсинга CSV.
            ValueError: Если DataFrame пустой после загрузки или обработки.
        """
        
        try:
            self.weather_data = pd.read_csv("20250522_023022_month.csv")

            self.weather_data['Период'] = pd.to_datetime(
                self.weather_data['Период'], errors='coerce'
            )

            if self.weather_data.empty:
                raise ValueError("DataFrame пустой после загрузки и обработки. ")

        except (FileNotFoundError, pd.errors.ParserError, ValueError) as e:
            self.weather_data = None
            raise e
        
        try:
            self.source_day_end_time = self.weather_data["Период"].max()
            self.source_day_start_time = self.source_day_end_time - pd.Timedelta(hours=24)

            self.predicted_data = self.get_predicted_data()

        except Exception as e:
            print(f"Предупреждение ! : не удалось получить предсказанные данные.\n Ошибка: {e}")
            raise e

    def get_predicted_data(self) -> pd.DataFrame:
        """
        Получение предсказанных данных для дня, следующего за последней доступной датой.
    
        Returns:
            pd.DataFrame: Таблица с предсказанными данными, полученными через функцию inference.
        """
        print(self.weather_data)

        data = inference(
            self.weather_data[
                (self.weather_data["Период"] >= self.source_day_start_time) &
                (self.weather_data["Период"] < self.source_day_end_time)
            ]
        )

        return data

    def update_plots(self, column: str, time: float) -> Tuple[plt.Figure, plt.Figure]:
        """ 
        Обновление графиков

        Args:
            column (str): Название признака для построения графика.
            time (float): Временная отметка в формате UNIX timestamp.

        Returns:
            Tuple[plt.Figure, plt.Figure]: Два объекта графиков:
                - График с данными текущего дня.
                - График с данными следующего дня, наложенными предсказанными значениями.
        """

        # Устанавливаем начальное и конечное время для выбранного дня
        self.source_day_start_time = pd.Timestamp(datetime.fromtimestamp(time))
        self.source_day_end_time = self.source_day_start_time + pd.Timedelta(hours=24)

        # Если выбранного дня нет в датасете
        if self.source_day_start_time.date() not in self.weather_data['Период'].dt.date.unique():
            print(f"Нет данных для даты: {self.source_day_start_time.strftime('%Y-%m-%d')}")

            fig1, ax1 = plt.subplots(figsize=(12, 6))
            ax1.text(0.5, 0.5, f"Нет данных для даты: {self.source_day_start_time.strftime('%Y-%m-%d')}",
                    horizontalalignment='center', verticalalignment='center')

            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.text(0.5, 0.5, "Нет данных", horizontalalignment='center', verticalalignment='center')

            return fig1, fig2

        # Забираем данные необходимые
        df_source_day = self.weather_data[
            (self.weather_data["Период"] >= self.source_day_start_time) & (self.weather_data["Период"] < self.source_day_end_time)
        ]

        # --- Первый график ---
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df_source_day["Период"], df_source_day[column], color='blue', label="Фактические значения")
        ax1.set_title(f"{column} на выбранный день")
        ax1.set_xlabel("Время")
        ax1.set_ylabel(column)
        ax1.legend()
        ax1.grid()

        # Форматирование оси X
        ax1.xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=1))
        ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

        fig1.tight_layout()

        # --- Второй график ---
        next_day_start_time = self.source_day_end_time
        next_day_end_time = next_day_start_time + pd.Timedelta(hours=24)

        self.predicted_data = self.get_predicted_data()

        df_next_day = self.weather_data[
            (self.weather_data['Период'] >= next_day_start_time) & (self.weather_data['Период'] < next_day_end_time)
        ]

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(
            df_next_day["Период"],
            df_next_day[column],
            color="blue",
            label="Фактические значения"
        )
        ax2.plot(
            self.predicted_data["Период"],
            self.predicted_data[column],
            color="orange",
            linestyle="--",
            label=f"Предсказанные значения"
        )
        ax2.set_title(f"{column} на следующий день")
        ax2.set_xlabel("Время")
        ax2.set_ylabel(column)
        ax2.legend()
        ax2.grid()

        # Форматирование оси X
        ax2.xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=1))
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

        fig2.tight_layout()

        return fig1, fig2
    
    def reader_csv(self, file) -> pd.DataFrame:
        """
        Чтение csv файла под обучение или дообучение.

        Args:
            file: Загруженный файл.
            flag: Флаг, указывающий, для обучения или дообучения.
        """

        try:
            self.weather_data = pd.read_csv(file.name)

            required_columns = ['Период', 'Температура, °С', 'Давление, мм рт. ст.', 'Влажность, %', 'Скорость ветра, м/с', 'Направление ветра, °']
            missing_columns = [col for col in required_columns if col not in self.weather_data.columns]

            if missing_columns:
                raise ValueError(f"Отсутствуют необходимые признаки : {missing_columns}")

            self.weather_data['Период'] = pd.to_datetime(self.weather_data['Период'], errors='coerce')

            if self.weather_data.empty:
                raise ValueError("DataFrame пустой после загрузки и обработки.")

            self.source_day_end_time = self.weather_data["Период"].max()
            self.source_day_start_time = self.source_day_end_time - pd.Timedelta(hours=24)

        except (FileNotFoundError, pd.errors.ParserError, ValueError) as e:
            self.weather_data = None
            raise e

        try:
            self.predicted_data = self.get_predicted_data()
            
        except Exception as e:
            print(f"Предупреждение: не удалось получить предсказанные данные. Ошибка: {e}")
            raise e

    def export_csv(self, file_name: str = "baseline.csv") -> str:
        """
        Экспортирует предсказанные значения в файл .csv.

        Args:
            file_name (str): Имя файла для сохранения. По умолчанию "baseline.csv".

        Returns:
            str: Полный путь к сохранённому файлу.

        Raises:
            ValueError: Если в `self.predicted_data` нет данных для экспорта.
            OSError: Если произошла ошибка при записи файла.
        """

        if self.predicted_data is None or self.predicted_data.empty:
            raise ValueError("Данные для экспорта отсутствуют. Убедитесь, что predicted_data заполнен.")

        self.predicted_data.insert(0, "station", "21 414")

        rename_columns = {
            'Период': 'time',
            'Температура, °С': 'temperature',
            'Давление, мм рт. ст.': 'pressure',
            'Влажность, %': 'humidity',
            'Скорость ветра, м/с': 'wind speed',
            'Направление ветра, °': 'wind direction'
        }
        self.predicted_data.rename(columns=rename_columns, inplace=True)

        # Если нету .csv в названии файла
        if not file_name.endswith(".csv"):
            file_name += ".csv"

        file_path = os.path.abspath(file_name)

        try:
            self.predicted_data.to_csv(file_path, index=False)
        except Exception as e:
            raise OSError(f"Ошибка при записи файла: {e}")

        return file_path

    def set_bool_value_flag(self, flag : str) -> bool:

        if (flag == "Обучение"):
            return False
        else:
            return True


    def launch(self):
        """
        Запуск сервиса
        """

        custom_css = """
        .radio-group .wrap {
            display: grid !important;
            grid-template-columns: 1fr 1fr;
        }

        .radio-group label {
            font-weight: bold;
            font-size: 16px;
        }
        
        """

        with gr.Blocks(title="ClimaScopeX", css=custom_css, theme=gr.themes.Ocean()) as application:

            with gr.Row():
                with gr.Column(scale=2):
                    column_selector = gr.Radio(
                        elem_classes="radio-group",
                        choices=[
                            'Температура, °С', 
                            'Давление, мм рт. ст.', 
                            'Влажность, %', 
                            'Скорость ветра, м/с', 
                            'Направление ветра, °'
                        ],
                        value='Влажность, %',
                        label="Признак для визуализации : ",
                    )

                    input_date = gr.DateTime(
                        label="Дата : ",
                        value = self.source_day_start_time
                    )

                    with gr.Blocks():

                        input_file = gr.File(
                            file_types=[".csv"],
                            label="Загрузка"
                        )

                        download_button = gr.Button(value="Экспорт данных")
                        download_output = gr.File()

                with gr.Column(scale=3):
                    with gr.Row():
                        source_graph = gr.Plot()
                    with gr.Row():
                        prediction_graph = gr.Plot()

            column_selector.change(
                fn=self.update_plots,
                inputs=[column_selector, input_date],
                outputs=[source_graph, prediction_graph]
            )

            input_file.upload(
                fn=self.reader_csv,
                inputs=[input_file],
                outputs=[]
            )

            download_button.click(
                fn=self.export_csv,
                inputs=[],
                outputs=download_output,
            )

        application.launch()


app = ClimaScopeX()
app.launch()