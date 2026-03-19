"""Download Common Voice 24.0 Belarusian dataset from Mozilla Data Collective."""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from datacollective import download_dataset, get_dataset_details

# Common Voice Scripted Speech 24.0 - Belarusian
DATASET_ID = "cmj8u3oug0029nxxboll1mh9e"


def main():
    if not os.environ.get("MDC_API_KEY"):
        print("ERROR: MDC_API_KEY environment variable not set.")
        print("Sign up at https://datacollective.mozillafoundation.org")
        print("Then add MDC_API_KEY to .env file")
        sys.exit(1)

    print(f"Download path: {os.environ.get('MDC_DOWNLOAD_PATH', '~/.mozdata/datasets')}")

    print("Fetching dataset details...")
    info = get_dataset_details(DATASET_ID)
    print(f"Dataset: {info}")

    print("\nDownloading Common Voice 24.0 Belarusian...")
    print("This may take a while depending on your connection.")
    dataset = download_dataset(DATASET_ID)
    print(f"\nDownload complete: {dataset}")


if __name__ == "__main__":
    main()
