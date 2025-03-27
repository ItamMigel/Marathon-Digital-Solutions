import pandas as pd

def process_data(file_path: str = 'data/Телеметрия.xlsx', output_path: str = 'data/all_data.parquet'):
    """
        Обработка данных из двух учатсков в один, объединение по столбцам и сохранение в parquet файл

        Args:
            file_path: str - путь к файлу с данными (по умолчанию 'data/Телеметрия.xlsx')
            output_path: str - путь к файлу для сохранения объединенных данных (по умолчанию 'data/all_data.parquet')
    """
    print(f'Processing data from {file_path} to {output_path}')
    df = pd.read_excel(file_path)
    print(f'Data loaded')
    df_columns = df.columns
    df_area_1 = df[[col for col in df_columns if not col.endswith('2')]]
    df_area_2 = df[[col for col in df_columns if not col.endswith('1')]]

    df_area_1.columns = [col.replace('1', '') for col in df_area_1.columns]
    df_area_2.columns = [col.replace('2', '') for col in df_area_2.columns]

    df_all = pd.concat([df_area_1, df_area_2], axis=0)
    print(f'Data concatenated')
    df_all.to_parquet(output_path, index=False)
    print(f'Data saved to {output_path}')

if __name__ == '__main__':
    process_data()
