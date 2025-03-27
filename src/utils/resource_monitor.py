"""
Resource monitoring module for the Anti-Corruption RAG system.
Monitors CPU, RAM, and GPU usage.
"""
import os
import time
import logging
import psutil
import threading
from datetime import datetime

# Flag to determine if GPU monitoring is available
GPU_AVAILABLE = False

# Try to import GPU monitoring tools
try:
    import torch
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
except ImportError:
    pass


def get_system_info():
    """
    Get system information including CPU, RAM, and GPU (if available).
    
    Returns:
        dict: System information
    """
    # Get CPU info
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_count = psutil.cpu_count()
    
    # Get memory info
    memory = psutil.virtual_memory()
    memory_used_gb = memory.used / (1024 ** 3)
    memory_total_gb = memory.total / (1024 ** 3)
    memory_percent = memory.percent
    
    # Initialize GPU info
    gpu_info = {"available": GPU_AVAILABLE}
    
    # Get GPU info if available
    if GPU_AVAILABLE:
        try:
            gpu_info["count"] = torch.cuda.device_count()
            
            # Get per-GPU info
            gpu_info["devices"] = []
            for i in range(gpu_info["count"]):
                device = torch.cuda.get_device_properties(i)
                gpu_memory = {}
                
                try:
                    # Try to get memory info
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                    memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # GB
                    memory_free = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3) - memory_reserved  # GB
                    
                    gpu_memory = {
                        "allocated_gb": memory_allocated,
                        "reserved_gb": memory_reserved,
                        "free_gb": memory_free,
                        "total_gb": torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                    }
                except Exception as e:
                    gpu_memory = {"error": str(e)}
                
                gpu_info["devices"].append({
                    "id": i,
                    "name": device.name,
                    "memory": gpu_memory
                })
        except Exception as e:
            gpu_info["error"] = str(e)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu": {
            "percent": cpu_percent,
            "count": cpu_count
        },
        "memory": {
            "used_gb": memory_used_gb,
            "total_gb": memory_total_gb,
            "percent": memory_percent
        },
        "gpu": gpu_info
    }


def log_memory_usage(logger=None):
    """
    Log current memory usage.
    
    Args:
        logger (logging.Logger, optional): Logger to use. If None, use print.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Get system info
    info = get_system_info()
    
    # Log CPU and RAM
    logger.info(f"CPU: {info['cpu']['percent']:.1f}% | "
                f"RAM: {info['memory']['used_gb']:.2f}GB/{info['memory']['total_gb']:.2f}GB "
                f"({info['memory']['percent']:.1f}%)")
    
    # Log GPU if available
    if info['gpu']['available']:
        for device in info['gpu']['devices']:
            try:
                memory = device['memory']
                logger.info(f"GPU {device['id']} ({device['name']}): "
                           f"{memory['allocated_gb']:.2f}GB allocated, "
                           f"{memory['free_gb']:.2f}GB free")
            except KeyError:
                logger.info(f"GPU {device['id']} ({device['name']}): Info unavailable")


class ResourceMonitor:
    """Monitor system resources periodically."""
    
    def __init__(self, interval=30, logger=None):
        """
        Initialize resource monitor.
        
        Args:
            interval (int): Monitoring interval in seconds
            logger (logging.Logger, optional): Logger to use
        """
        self.interval = interval
        self.logger = logger or logging.getLogger(__name__)
        self.running = False
        self.thread = None
    
    def _monitor_loop(self):
        """Internal monitoring loop."""
        while self.running:
            log_memory_usage(self.logger)
            time.sleep(self.interval)
    
    def start(self):
        """Start resource monitoring."""
        if self.running:
            self.logger.warning("Resource monitor already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        self.logger.info(f"Resource monitoring started (interval: {self.interval}s)")
    
    def stop(self):
        """Stop resource monitoring."""
        if not self.running:
            self.logger.warning("Resource monitor not running")
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        self.logger.info("Resource monitoring stopped")
    
    def get_current_status(self):
        """
        Get current resource status.
        
        Returns:
            dict: Current system information
        """
        return get_system_info()
