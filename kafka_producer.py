# kafka_producer.py

from confluent_kafka import Producer
import logging

class KafkaLogger:
    def __init__(self, bootstrap_servers='localhost:9092', client_id='data-processor'):
        self.kafka_conf = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': client_id
        }
        self.producer = Producer(self.kafka_conf)
        self.logger = logging.getLogger(__name__)

    def send_log(self, event_type, details, duration=None):
        """Отправляет лог-сообщение в Kafka."""
        message = {
            'event_type': event_type,
            'details': details,
            'duration_ns': duration
        }
        self.producer.produce('logs_topic', value=str(message))  # Убедитесь, что 'logs_topic' существует
        self.producer.flush()

# Пример использования:
# kafka_logger = KafkaLogger()
# kafka_logger.send_log('event_type', 'details', duration)
