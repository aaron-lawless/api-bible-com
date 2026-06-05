import threading
import time

from fastapi import FastAPI
import uvicorn
from prometheus_client import Gauge, generate_latest
import psutil


prometheus_app = FastAPI()

@prometheus_app.get("/metrics")
async def metrics():
    """ Expose Prometheus metrics."""
    return generate_latest()

CPU_USAGE = Gauge("system_cpu_usage", "Current CPU usage percentage")
MEMORY_USAGE = Gauge("system_memory_usage", "Current Memory usage percentage")
DISK_USAGE = Gauge("system_disk_usage", "Current Disk usage percentage")

def collect_system_metrics():
    while True:
        # Collect CPU, Memory, and Disk usage metrics
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().percent)
        DISK_USAGE.set(psutil.disk_usage('/').percent)
        time.sleep(5)  # Collect metrics every 5 seconds

async def start_prometheus_app():
    # Prometheus is on a different port
    config = uvicorn.Config(prometheus_app, host='0.0.0.0', port=9090)
    server = uvicorn.Server(config)
    await server.serve()

# Start a background thread to collect system metrics
thread = threading.Thread(target=collect_system_metrics, daemon=True)
thread.start()
    