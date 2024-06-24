import time
from multiprocessing import Process

import pika
import torch.multiprocessing as mp
from run import run

from utils import log, print_debug

# Short sleep avoids race conditions with server/RabbitMQ
sleepTime = 10
print_debug(f" [x] Sleeping for {sleepTime} seconds.")
time.sleep(sleepTime)

print_debug(" [x] Connecting to server ...")
connection = pika.BlockingConnection(pika.ConnectionParameters(host="rabbitmq"))
channel = connection.channel()
channel.queue_declare(queue="task_queue", durable=True)


def on_request(ch, method, props, body):
    """
    Handles a request received from the RabbitMQ queue.

    This function is called when a message is received from the "task_queue" queue.
    It decodes the message body, runs the corresponding command using the `run()` function,
    and publishes the response back to the RabbitMQ exchange with the original correlation ID.

    Args:
        ch (pika.channel.Channel): The RabbitMQ channel.
        method (pika.spec.Basic.Deliver): The delivery information.
        props (pika.spec.BasicProperties): The message properties.
        body (bytes): The message body.
    """
    print_debug(f"{ch =}")
    print_debug(f"{method =}")
    print_debug(f"{props =}")
    print_debug(f"{body =}")
    print_debug(" [x] Received %s" % body)

    cmd = body.decode()
    response = run(cmd)

    print_debug(f" [x] Sending {response}")
    print_debug(" [x] Done")

    ch.basic_publish(
        exchange="",
        routing_key=props.reply_to,
        properties=pika.BasicProperties(correlation_id=props.correlation_id),
        body=str(response),
    )

    ch.basic_ack(delivery_tag=method.delivery_tag, multiple=True)


class Worker:
    def __init__(self):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host="rabbitmq"))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue="task_queue", durable=True)
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue="task_queue", on_message_callback=on_request)

    def run(self):
        print_debug(" [x] Waiting for messages.")
        self.channel.start_consuming()


def worker_process():
    """Runs a worker process that listens for tasks on the RabbitMQ queue and executes them."""
    worker = Worker()
    worker.run()


if __name__ == "__main__":
    # Choose the method of multiprocessing to use
    # TODO: Multi isn't working for GPU.
    method = "single"

    if method == "multi-cuda":
        print_debug(f"~~~ METHOD: {method}")
        # NOTE: this is required for the ``fork`` method to work
        # worker.share_memory()
        nproc = 1
        processes = []
        worker = Worker()
        mp.set_start_method("spawn")
        for _ in range(nproc):
            p = mp.Process(target=worker_process)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    elif method == "multi-cpu":
        # Start multiple worker processes
        print_debug(f"~~~ METHOD: {method}")
        process_list = []
        nproc = 1
        for _ in range(nproc):
            process = Process(target=worker_process)
            process.start()
            process_list.append(process)

        # Join all processes at the end
        for process in process_list:
            process.join()

    elif method == "single":
        print_debug(f"~~~ METHOD: {method}")
        worker = Worker()
        worker.run()
