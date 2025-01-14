# Background Processing with RabbitMQ, Python and Flask

A simple job worker that uses RabbitMQ for job management. Full doc and explanation [here](https://medium.com/@naveed125/background-processing-with-rabbitmq-python-and-flask-5ca62acf409c).

# Quick start

## Start processes

Use one of the following commands to start all three processes:

```bash
$ docker-compose -f docker-compose.cpu.yml up -d # CPU
$ docker-compose -f docker-compose.cuda.yml up -d # GPU

Starting rabbitmq-job-worker_worker_1 ... done
Starting rabbitmq-job-worker_rabbitmq_1 ... done
Starting rabbitmq-job-worker_server_1 ... done
```

## Run jobs

To see all of this in action, just hit the /add-job/hey or /add-job/hello end-point on your localhost and you will see the messages flowing through.

```bash
$ curl localhost:5002/add-job/hey
[x] Sent: hey
```

## Check Jobs

The worker container logs should show the job being executed:

```bash
% docker logs rabbitmq-job-worker_worker_1
...
[*] Sleeping for 10 seconds.
[*] Connecting to server ...
[*] Waiting for messages.
[x] Received b'hey'
hey there
[x] Done
```
