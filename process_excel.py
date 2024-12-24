import pandas as pd
from normalize import normalize_name
from utils import clear_and_display_dataframe

def process_excel_column(excel_file, column_name):
    """Обрабатывает указанный столбец в Excel-файле и применяет нормализацию."""
    try:
        df = pd.read_excel(excel_file)
    except Exception as e:
        raise ValueError(f"Ошибка при чтении Excel файла: {e}")

    if column_name not in df.columns:
        raise ValueError(f"Столбец '{column_name}' не найден в файле.")

    processed_rows = []

    for index, row in df.iterrows():
        normalized_data = normalize_name(row[column_name])
        processed_row = {
            column_name: row[column_name],
            'Наименование(нормализованное)': normalized_data['Наименование(нормализованное)'],
            'Ошибка': 'ИСТИНА' if normalized_data['Modified'] else 'ЛОЖЬ'
        }
        processed_rows.append(processed_row)

    # Create DataFrame after processing all rows
    current_df = pd.DataFrame(processed_rows)
    clear_and_display_dataframe(current_df)

    return current_df
