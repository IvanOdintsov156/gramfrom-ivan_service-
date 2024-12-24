import pandas as pd
import logging

# Настройки для pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Настройки для логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Параметры
MAX_LENGTH = 512
spell_check_dict = {
    "Риле": "Реле",
    "кантакта": "контакта",
    "Релень": "Ремень",
    "Хамут": "Хомут",
    "Релень": "Ремень"
}
