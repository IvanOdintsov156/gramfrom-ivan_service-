import logging
import time

# Настройка логирования
logging.basicConfig(
    filename="log_file.log",  # Имя файла для сохранения логов
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)

# Логируем информационное сообщение
def log_info(message, value=None):
    if value is not None:
        logging.info(f"{message}: {value}")
    else:
        logging.info(message)

# Измеряем продолжительность времени
def measure_time(start_time, counter=0):
    elapsed_time = time.time() - start_time
    if counter > 0:
        elapsed_time /= counter
    if elapsed_time < 0.01:
        return f"{elapsed_time * 1_000_000:.2f} µs"
    if elapsed_time < 1:
        return f"{elapsed_time * 1000:.2f} ms"
    elif elapsed_time < 60:
        return f"{elapsed_time:.2f} s"
    else:
        return f"{elapsed_time / 60:.2f} m"
