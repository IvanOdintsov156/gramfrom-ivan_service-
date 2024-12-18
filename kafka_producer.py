# kafka_logger.py
from confluent_kafka import Producer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Kafka configuration
kafka_conf = {
    'bootstrap.servers': 'localhost:9092',  # Замените на адрес вашего Kafka сервера
}
producer = Producer(kafka_conf)

def send_log_to_kafka(message):
    """Отправляет сообщение в Kafka."""
    producer.produce('logs', message)  # 'logs' - название топика для логов
    producer.flush()

def log_event(event_description):
    """Логирует событие и отправляет его в Kafka."""
    logger.info(event_description)
    send_log_to_kafka(event_description)
