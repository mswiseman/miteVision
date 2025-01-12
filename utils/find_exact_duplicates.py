import os
import hashlib
import argparse

def compute_md5(file_path, chunk_size=8192):
    """
    Compute MD5 hash of the specified file.

    Args:
        file_path (str): Path to the file.
        chunk_size (int): Number of bytes to read at a time.

    Returns:
        str: MD5 hexadecimal digest of the file.
    """
    md5 = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    return md5.hexdigest()

def find_exact_duplicates(directory, delete=False):
    """
    Find and optionally delete exact duplicate images in a directory.

    Args:
        directory (str): Path to the directory to scan.
        delete (bool): If True, delete duplicate files.
    """
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    # Dictionary to map file hash to file paths
    hash_dict = {}

    # Traverse the directory
    for root, dirs, files in os.walk(directory):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                file_path = os.path.join(root, filename)
                file_hash = compute_md5(file_path)
                if file_hash:
                    if file_hash in hash_dict:
                        hash_dict[file_hash].append(file_path)
                    else:
                        hash_dict[file_hash] = [file_path]

    # Identify duplicates
    duplicates = {hash_: paths for hash_, paths in hash_dict.items() if len(paths) > 1}

    if not duplicates:
        print("No exact duplicates found.")
        return

    print(f"Found {len(duplicates)} sets of duplicates.")

    # Iterate through duplicates and handle them
    for hash_, files in duplicates.items():
        print(f"\nDuplicate Set (MD5: {hash_}):")
        for i, file in enumerate(files):
            if i == 0:
                print(f"  [Keep]    {file}")
            else:
                print(f"  [Delete]  {file}")
                if delete:
                    try:
                        os.remove(file)
                        print(f"    Deleted: {file}")
                    except Exception as e:
                        print(f"    Error deleting {file}: {e}")

    if delete:
        print("\nDuplicate files have been deleted.")
    else:
        print("\nTo delete duplicates, run the script with the '--delete' flag.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and delete exact duplicate images in a directory.")
    parser.add_argument("directory", type=str, help="Path to the directory to scan for duplicates.")
    parser.add_argument("--delete", action="store_true", help="Delete duplicate files.")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: The directory '{args.directory}' does not exist or is not accessible.")
    else:
        find_exact_duplicates(args.directory, delete=args.delete)
# Usage example:
# python find_exact_duplicates.py /path/to/directory