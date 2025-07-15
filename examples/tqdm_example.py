import time
from pathlib import Path

from tqdm import tqdm

fname = Path() / "output_2024-10-07T23-59-09.csv"
total_iterations = 11 * 11 * 6 * 12
with tqdm(total=total_iterations, desc="Simulating", unit="iteration") as pbar:
    progress = 0
    while True:
        if fname.exists():
            with open(fname, "r") as f:
                total = len(f.readlines())
            pbar.update(total - progress)
            progress = total
            if progress >= total_iterations:
                break
        time.sleep(5)
