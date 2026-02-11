
import os
from tqdm import tqdm
from threading import Thread

import datasets
from datasets.utils.logging import disable_progress_bar, enable_progress_bar


TARGET_PATH = "/home/ubuntu/.cache/huggingface/hub/datasets--allenai--dolma3_longmino_mix-50B-1025/snapshots/8c0b3b265f95514c0f1b643c95da518e261a32a7/data"

SAVE_PATH = "data/longmino-50B"


def main():

    disable_progress_bar()

    dataset_list = []
    skipped_files = []

    folder_list = list(os.listdir(TARGET_PATH))
    folder_list.sort()

    for folder in tqdm(folder_list, desc="Folders"):
        folder_path = os.path.join(TARGET_PATH, folder)

        file_list = list(os.listdir(folder_path))
        file_list.sort()

        for file in tqdm(file_list, desc=f"{folder}"):
            if not file.endswith(".jsonl.zst"):
                print(f"Skipping {file}")
                continue

            try:
                data = datasets.load_dataset(
                    folder_path,
                    data_files=[file],
                )
            except KeyboardInterrupt as e:
                raise e
            except:
                print(f"Skipping {file} in {folder} due to read error.")
                skipped_files.append(os.path.join(folder_path, file))
                continue

            dataset_list.append(data)

    with open("skipped_files.txt", "w") as f:
        for file in skipped_files:
            f.write(file + "\n")

    enable_progress_bar()

    full_data = datasets.concatenate_datasets(dataset_list)
    full_data = full_data.shuffle(seed=42)

    full_data.save_to_disk(SAVE_PATH)
                

if __name__ == "__main__":
    main()
