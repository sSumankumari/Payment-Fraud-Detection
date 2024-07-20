import os
from dotenv import load_dotenv
import kaggle

# Load environment variables from .env file
load_dotenv()

# Ensure Kaggle API credentials are set from environment variables
kaggle_username = os.getenv('KAGGLE_USERNAME')
kaggle_key = os.getenv('KAGGLE_KEY')

if not kaggle_username or not kaggle_key:
    raise ValueError("KAGGLE_USERNAME and KAGGLE_KEY must be set in the .env file")


# Download dataset
def download_dataset():
    dataset_name = 'ealaxi/paysim1'
    dataset_dir = 'datasets/paysim1'

    # Ensure the dataset directory exists
    os.makedirs(dataset_dir, exist_ok=True)

    # Download the dataset
    kaggle.api.dataset_download_files(dataset_name, path=dataset_dir, unzip=True)


if __name__ == "__main__":
    download_dataset()
