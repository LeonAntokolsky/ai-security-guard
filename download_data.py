import os
import requests
import zipfile


def download_dataset_direct():
    # Define dataset details and paths
    dataset_owner = "naveenk903"
    dataset_name = "movies-fight-detection-dataset"
    output_zip = "data/dataset.zip"
    extract_dir = "data/movies_fight_dataset"

    print(f"[INFO] Initializing direct web download for: {dataset_owner}/{dataset_name}")

    # Ensure the destination data directory exists
    os.makedirs("data", exist_ok=True)

    # Read the Kaggle access token from the hidden directory
    token_path = os.path.expanduser("~/.kaggle/access_token")
    if not os.path.exists(token_path):
        print("[ERROR] Kaggle access token file not found at ~/.kaggle/access_token")
        print("[HELP] Please make sure you created the token file correctly.")
        return

    with open(token_path, "r") as f:
        token = f.read().strip()

    # Construct the raw Kaggle API endpoint URL for dataset download
    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_owner}/{dataset_name}"

    # Apply the modern KGAT token authentication header
    headers = {
        "Authorization": f"Bearer {token}"
    }

    print("[INFO] Connecting to Kaggle servers and streaming the file...")
    try:
        # Request the file as a stream to handle large files efficiently
        response = requests.get(url, headers=headers, stream=True)

        # Check if the token was accepted (HTTP 200)
        if response.status_code != 200:
            print(f"[ERROR] Connection failed. HTTP Status: {response.status_code}")
            print(f"[RESPONSE] Server message: {response.text}")
            return

        # Write the incoming data chunks into the zip file
        with open(output_zip, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"[SUCCESS] Archive downloaded completely and saved to: {output_zip}")

        # Unzip the downloaded files automatically
        print("[INFO] Extracting zip file contents into the project directory...")
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Clean up the zip file to save disk space
        os.remove(output_zip)
        print(f"[SUCCESS] Extraction complete! Dataset folder is located at: {extract_dir}")

    except Exception as e:
        print(f"[CRITICAL ERROR] Pipeline execution failed: {e}")


if __name__ == "__main__":
    download_dataset_direct()