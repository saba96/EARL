import os
import urllib.request
import shutil
import tarfile
import argparse
from urllib.parse import urlparse, unquote

def get_filename_from_url(url):
    """Extract filename from URL, handling query parameters and fragments."""
    # Parse the URL and get the path
    parsed = urlparse(url)
    # Split the path by '/' and get the last non-empty part
    path_parts = [p for p in parsed.path.split('/') if p]
    if not path_parts:
        return 'downloaded_file.tar.gz'
    
    # Get the last part and decode it
    filename = unquote(path_parts[-1])
    
    # If filename ends with a query parameter or fragment, remove it
    if '?' in filename:
        filename = filename.split('?')[0]
    if '#' in filename:
        filename = filename.split('#')[0]
        
    # If we ended up with an empty string or no filename, use default
    if not filename:
        filename = 'downloaded_file.tar.gz'
        
    return filename

def download_and_extract(url, output_dir, output_file):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Download the file
    print(f"Downloading dataset from {url}...")
    urllib.request.urlretrieve(url, output_file)
    print(f"Dataset downloaded and saved to {output_file}")

    # Extract the tar.gz file
    print(f"Extracting {output_file} to {output_dir}...")
    with tarfile.open(output_file, "r:gz") as tar:
        tar.extractall(path=output_dir)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Download and extract dataset')
    parser.add_argument('--base_dir', type=str, required=True, 
                       help='Base directory where the data will be stored')
    parser.add_argument('--url', type=str, required=True,
                       help='URL for the dataset')
    args = parser.parse_args()

    # Set up output directories and files
    output_dir = os.path.join(args.base_dir, "data")
    # Extract filename from URL
    filename = get_filename_from_url(args.url)
    output_file = os.path.join(output_dir, filename)

    # Download and extract data
    download_and_extract(args.url, output_dir, output_file)

if __name__ == "__main__":
    main()
