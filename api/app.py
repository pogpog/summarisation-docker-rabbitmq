import json
import re
import uuid
from typing import Dict

from utils import log, print_debug
import pika
from flask import Flask, request

app = Flask(__name__)
queue: dict = {}


@app.route("/")
def index():
    """Test connection"""
    return "OK", 200


@app.route("/highlights", methods=["POST"])
def add():

    request.get_data()
    cmd = get_input()
    print_debug(f" [x] Requesting {cmd}")
    worker = WorkerClient()
    response = worker.call(cmd)

    return response


@app.route("/results")
def send_results():

    return str(queue.items())


def get_input() -> str:
    """Retrieves the text from the request data, or a default text if no data is provided.

    Returns:
        str: The cleaned text from the request data, or the default text.
    """
    if request.data:
        data = request.get_json()
        text = json.loads(json.dumps(data))["text"]
        return clean_text(text)

    return ""


def clean_text(text: str) -> str:
    """Clean text to remove unwanted characters

    Args:
        text (str): Text to clean.

    Returns:
        str: Cleaned text.
    """
    text = text.replace('"', '\\"')
    text = re.sub(r"http[s]?://\S+", "", text)
    text = text.replace("", "")
    text = text.replace("\n", " ")
    text = text.strip()
    return text


class WorkerClient:

    def __init__(self):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host="rabbitmq"))

        self.channel = self.connection.channel()

        result = self.channel.queue_declare("", exclusive=True)
        self.callback_queue = result.method.queue
        self.channel.basic_consume(queue=self.callback_queue, on_message_callback=self.on_response, auto_ack=True)

        self.response = None
        self.corr_id = None

    def on_response(self, ch, method, props, body):
        """
        Callback function that is called when a response is received from the RabbitMQ queue.

        Args:
            ch (pika.channel.Channel): The RabbitMQ channel.
            method (pika.spec.Basic.Deliver): The delivery information.
            props (pika.spec.BasicProperties): The message properties.
            body (bytes): The response message body.

        Returns:
            None
        """
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, cmd: str):
        """
        Call a RabbitMQ task queue and wait for the response.

        Args:
            cmd (str): The command to send to the task queue.

        Returns:
            str: The response from the task queue.
        """
        self.response = None
        self.corr_id = str(uuid.uuid4())
        queue[self.corr_id] = None
        self.channel.basic_publish(
            exchange="",
            routing_key="task_queue",
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=cmd,
        )
        while self.response is None:
            self.connection.process_data_events(time_limit=None)
        queue[self.corr_id] = self.response
        print_debug(self.response)
        self.connection.close()
        return str(self.response)


# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port="5002")
