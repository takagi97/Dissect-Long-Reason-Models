import os
import shutil

BASE_DIR = "/mnt/gemininjceph2/geminicephfs/pr-others-prctrans/jerrymu/r1/train/deepscaler/checkpoints/deepscaler/4.1.8to16k-A100-bsz64-baseline"

def find_and_delete_error_dirs():
    for root, dirs, files in os.walk(BASE_DIR):
        for dir_name in dirs:
            if dir_name == "aime":
                aime_path = os.path.join(root, dir_name)
                result_file = os.path.join(aime_path, "results_details.json")
                if os.path.exists(result_file):
                    try:
                        with open(result_file, "r", encoding="utf-8") as f:
                            content = f.read()
                            if "Error, Traceback (most recent call last):" in content:
                                print(f"[DELETING] Found error in: {result_file}")
                                # shutil.rmtree(aime_path)
                    except Exception as e:
                        print(f"[ERROR] Could not read {result_file}: {e}")

if __name__ == "__main__":
    find_and_delete_error_dirs()
