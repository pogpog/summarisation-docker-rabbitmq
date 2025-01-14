import time
from multiprocessing import Process

import pika
import torch.multiprocessing as mp
from run import run

sleepTime = 10
print(" [x] Sleeping for ", sleepTime, " seconds.")
time.sleep(sleepTime)

print(" [x] Connecting to server ...")
connection = pika.BlockingConnection(pika.ConnectionParameters(host="rabbitmq"))
channel = connection.channel()
channel.queue_declare(queue="task_queue", durable=True)


def on_request(ch, method, props, body):
    print(" [x] Received %s" % body)
    cmd = body.decode()

    # response = my_crazy_function(cmd)
    response = run(cmd)

    print(f" [x] Sending {response}")
    print(" [x] Done")

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
        print(" [x] Waiting for messages.")

    def run(self):
        print(" [x] Waiting for messages.")
        self.channel.start_consuming()


def worker_process():
    """Runs a worker process that listens for tasks on the RabbitMQ queue and executes them."""
    worker = Worker()
    worker.run()


if __name__ == "__main__":
    method = "cuda-single"

    if method == "cuda-multi":
        print(f"METHOD: {method}")
        # NOTE: this is required for the ``fork`` method to work
        # worker.share_memory()
        num_processes = 1
        processes = []
        worker = Worker()
        mp.set_start_method("spawn")
        for _ in range(num_processes):
            p = mp.Process(target=worker_process)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    elif method == "cpu-multi":
        # Start multiple worker processes
        print(f"METHOD: {method}")
        process_list = []
        worker = Worker()
        for _ in range(2):
            process = Process(target=worker.run)
            process.start()
            process_list.append(process)

        # Join all processes at the end
        for process in process_list:
            process.join()

    elif method == "cuda-single":
        worker = Worker()
        worker.run()
