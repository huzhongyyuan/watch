import argparse
from pathlib import Path
import subprocess
import os
from os.path import dirname as pathdir
import sys
sys.path.append(pathdir(pathdir(pathdir(os.path.abspath(__file__)))))
from core.utils.pylogger import Log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str)
    parser.add_argument("-d", "--output_root", type=str, default=None)
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip DPVO")
    parser.add_argument("-v", "--verbose", action="store_true", help="If true, draw 2d keypoints and bbox")
    args = parser.parse_args()

    folder = Path(args.folder)
    output_root = args.output_root

    # Run demo.py for each .mp4 file
    mp4_paths = sorted(list(folder.glob("*.mp4")))
    Log.info(f"Found {len(mp4_paths)} .mp4 files in {folder}")
    for mp4_path in mp4_paths:
        command = ["f:/VIDEO/GVHMR/venv/Scripts/python.exe", "scripts/demo/watch_demo.py", "--video", str(mp4_path)]
        if output_root is not None:
            command += ["--output_root", output_root]
        if args.static_cam:
            command += ["-s"]
        if args.verbose:
            command += ["--verbose"]
        Log.info(f"Running: {' '.join(command)}")
        subprocess.run(command, env=dict(os.environ), check=True)
