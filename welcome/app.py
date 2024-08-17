import os
import platform
import psutil
import GPUtil
import time
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from datetime import datetime

# Initialize console
console = Console()

# Define the log file location
log_file_path = r"C:\Users\orior\VideoImageEnhancementApp\Logs\log.txt"


def get_system_info():
    system = platform.system()
    release = platform.release()
    version = platform.version()
    machine = platform.machine()
    node = platform.node()
    processor = platform.processor()

    # Get GPU information
    gpus = GPUtil.getGPUs()
    gpu_info = ", ".join([gpu.name for gpu in gpus])

    return {
        "os": f"{system} {release} {machine}",
        "host": node,
        "kernel": version,
        "cpu": processor,
        "gpu": gpu_info if gpu_info else "No GPU detected",
        "memory": f"{psutil.virtual_memory().total // (1024 ** 2)}MiB",
        "log_file_path": r"C:\Users\orior\VideoImageEnhancementApp\Logs",
    }


def log_message(message):
    # Log the message to a file
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"{datetime.now()}: {message}\n")


def display_banner():
    # Clear the screen
    os.system('cls' if os.name == 'nt' else 'clear')

    # Large ASCII "L" and Banner Text
    ascii_art = [
        "[bold red] LLLLLLL[/bold red]",
        "[bold red] LLLLLLL[/bold red]",
        "[bold red] LLLLLLL[/bold red]",
        "[bold red] LLLLLLL[/bold red]",
        "[bold red] LLLLLLL[/bold red]",
        "[bold red] LLLLLLL[/bold red]",
        "[bold red] LLLLLLL[/bold red]",
        "[bold red] LLLLLLL[/bold red]",
        "[bold red] LLLLLLL[/bold red]",
        "[bold red] LLLLLLL[/bold red]",
        "[bold red] LLLLLLLLLLLLLLLLLLLLL[/bold red]",
        "[bold red] LLLLLLLLLLLLLLLLLLLLL[/bold red]",
        "[bold red] LLLLLLLLLLLLLLLLLLLLL[/bold red]",
        "[bold red] LLLLLLLLLLLLLLLLLLLLL[/bold red]         [bold yellow on black]Welcome to Algolight Enhancement App[/bold yellow on black]",
        "                      ",
        "[cyan] OS: {os}[/cyan]              ",
        "[cyan] Host: {host}[/cyan]           ",
        "[cyan] Kernel: {kernel}[/cyan]      ",
        "[cyan] Uptime: {uptime}[/cyan]      ",
        "[cyan] CPU: {cpu}[/cyan]            ",
        "[cyan] GPU: {gpu}[/cyan]            ",
        "[cyan] Memory: {memory}[/cyan]      ",
        "[green]Log file will be located in {log_file_path}[/green]",

    ]

    # Get system information
    info = get_system_info()

    # Calculate uptime
    uptime_seconds = time.time() - psutil.boot_time()
    uptime_string = time.strftime("%H hours, %M mins", time.gmtime(uptime_seconds))
    info["uptime"] = uptime_string

    # Display ASCII art with system information
    for line in ascii_art:
        formatted_line = line.format(**info)
        console.print(formatted_line)
        log_message(formatted_line)

    # Pause to keep the window open
    # input("\nPress Enter to continue...")
    print("Please wait while the program is loading....")

if __name__ == "__main__":
    display_banner()
