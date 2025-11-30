import os
import re
from pathlib import Path

def has_chinese(text):
    """Check if text contains Chinese characters"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def rename_files_in_directory(directory):
    """Rename files with Chinese characters to English equivalents"""
    directory = Path(directory)
    renamed_count = 0

    print(f"\nScanning directory: {directory}")
    print("=" * 80)

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if has_chinese(filename):
                old_path = Path(root) / filename

                name, ext = os.path.splitext(filename)

                numbers = re.findall(r'\d+', filename)
                if numbers:
                    new_name = f"defect_{numbers[0]}{ext}"
                else:
                    new_name = f"defect_{renamed_count + 1}{ext}"

                new_path = Path(root) / new_name

                counter = 1
                while new_path.exists():
                    name_part = new_name.rsplit('.', 1)[0]
                    new_path = Path(root) / f"{name_part}_{counter}{ext}"
                    counter += 1

                print(f"Renaming:")
                print(f"  FROM: {old_path}")
                print(f"  TO:   {new_path}")

                os.rename(old_path, new_path)
                renamed_count += 1

    print("=" * 80)
    print(f"\nTotal files renamed: {renamed_count}")
    return renamed_count

if __name__ == "__main__":
    data_root = "data/my_nut_dataset/nut"

    test_dirs = [
        os.path.join(data_root, "test", "crack"),
        os.path.join(data_root, "test", "scratch"),
        os.path.join(data_root, "test", "other_defect"),
        os.path.join(data_root, "test", "good"),
    ]

    total_renamed = 0
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            renamed = rename_files_in_directory(test_dir)
            total_renamed += renamed

    print(f"\n{'='*80}")
    print(f"TOTAL FILES RENAMED ACROSS ALL DIRECTORIES: {total_renamed}")
    print(f"{'='*80}")
