import os
import shutil
from pathlib import Path

def move_files_and_directories(src_dir: str, dest_dir: str, items: list):
    """Move data files and directories from one directory to another"""
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    
    for item in items:
        src_path = os.path.join(src_dir, item)
        dest_path = os.path.join(dest_dir, item)
        
        # Check if the item exists in the source directory
        if os.path.exists(src_path):
            # Move file or directory
            shutil.move(src_path, dest_path)
            print(f"Moved {item} from {src_dir} to {dest_dir}.")
        else:
            print(f"Item {item} not found in {src_dir}.")

if __name__ == "__main__":
    src_directory = "dev"
    dest_directory = "app"
    items_to_move = ["table_index_dir", "table_info_directory", "transactions.db"]

    move_files_and_directories(src_directory, dest_directory, items_to_move)
