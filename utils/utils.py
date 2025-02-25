from pathlib import Path
import logging

def get_root(num_parents=2):
    BASE_PATH = Path(__file__).resolve()

    for _ in range(num_parents):
        BASE_PATH = BASE_PATH.parent

    return BASE_PATH

def get_path(directory, dataset_size, dataset_name, num_parents=2):
    return get_root(num_parents=2) / directory / dataset_size / dataset_name

def get_directory(directory, dataset_size, num_parents=2):
    return get_root(num_parents) / directory / dataset_size

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("my_logger")
