# kafka_logger.py
from confluent_kafka import Producer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KafkaLogger:
    def __init__(self, kafka_conf, topic):
        self.producer = Producer(**kafka_conf)
        self.topic = topic

    def send_log(self, event, duration):
        """Отправляет лог события в Kafka."""
        message = {
            'event': event,
            'duration_ns': duration
        }
        self.producer.produce(self.topic, value=str(message))
        self.producer.flush()
        logger.info(f"Лог отправлен в Kafka: {message}")

# Пример использования
# kafka_conf = {'bootstrap.servers': 'localhost:9092'}
# logger = KafkaLogger(kafka_conf, 'logs_topic')
# logger.send_log('Пример события', 123456789)
