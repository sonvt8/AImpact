import os
import requests
import gzip
import shutil

# Define the URL for the Vietnamese fasttext model
FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.bin.gz"
MODEL_DIR = "models"
MODEL_FILENAME = "cc.vi.300.bin.gz"
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "cc.vi.300.bin")

def download_file(url: str, dest_path: str):
    """Download a file from a URL and save it to the specified path."""
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded to {dest_path}")

def decompress_gz(gz_path: str, dest_path: str):
    """Decompress a .gz file and save it to the specified path."""
    print(f"Decompressing {gz_path}...")
    with gzip.open(gz_path, "rb") as f_in:
        with open(dest_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Decompressed to {dest_path}")

def main():
    # Create the models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Define paths
    gz_path = os.path.join(MODEL_DIR, MODEL_FILENAME)

    # Download the model
    try:
        download_file(FASTTEXT_MODEL_URL, gz_path)
    except Exception as e:
        print(f"Error downloading model: {e}")
        return

    # Decompress the model
    try:
        decompress_gz(gz_path, FINAL_MODEL_PATH)
    except Exception as e:
        print(f"Error decompressing model: {e}")
        return

    # Remove the .gz file
    os.remove(gz_path)
    print(f"Model saved to {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    main()