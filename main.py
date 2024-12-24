import pandas as pd
from config import logger
from process_excel import process_excel_column
from utils import calculate_percentage
import os

def main():
    """Основная функция для выполнения обработки данных."""
    excel_file = '10записей.xlsx'
    column_name = 'Наименование(ненормализованное)'

    # Input validation
    if not os.path.isfile(excel_file):
        logger.error(f"Файл {excel_file} не найден.")
        return

    if not column_name:
        logger.error("Имя столбца не может быть пустым.")
        return

    try:
        result_df = process_excel_column(excel_file, column_name)
    except Exception as e:
        logger.error(f"Ошибка при обработке файла: {e}")
        return

    total_records = len(result_df)
    correct_original_names = result_df['Ошибка'].value_counts().get('ИСТИНА', 0)
    percentage_original = calculate_percentage(correct_original_names, total_records)

    result_df.loc[len(result_df)] = [f"Процент корректно нормализованных записей (оригинальные): {percentage_original:.2f}%", "", ""]
    result_df.to_excel('normalized_names.xlsx', index=False)
    logger.info(result_df)

if __name__ == "__main__":
    main()
