"""
Step 1 — Extract frames from video files using FFmpeg.
Walks a directory of movies/series and saves 1 frame/sec as JPEG.
"""

import os
import subprocess
import argparse


def extract_frames(video_path, output_dir, fps=1):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    movie_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(movie_output_dir, exist_ok=True)

    output_path = os.path.join(movie_output_dir, f"{video_name}_%03d.jpg")

    command = [
        "ffmpeg",
        "-i", video_path,
        "-sn",
        "-vf", f"fps={fps},scale=1280:-1",
        "-q:v", "20",
        output_path,
        "-hide_banner", "-loglevel", "error",
    ]

    print(f"Extracting frames from: {video_name}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"  [WARNING] FFmpeg returned non-zero exit code for {video_name}")
    else:
        print(f"  Saved to: {movie_output_dir}")


def process_videos(root_video_dir, root_output_dir, fps=1):
    supported = (".mp4", ".mkv", ".avi", ".mov")
    found = 0
    for dirpath, _, filenames in os.walk(root_video_dir):
        for file in filenames:
            if file.lower().endswith(supported):
                video_path = os.path.join(dirpath, file)
                rel_path = os.path.relpath(dirpath, root_video_dir)
                output_dir = os.path.join(root_output_dir, rel_path)
                extract_frames(video_path, output_dir, fps=fps)
                found += 1
    if found == 0:
        print("No video files found. Check --video_dir.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video files.")
    parser.add_argument("--video_dir", required=True, help="Root folder containing video files.")
    parser.add_argument("--output_dir", default="frames", help="Root folder to save extracted frames (default: ./frames).")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to extract (default: 1).")
    args = parser.parse_args()

    process_videos(args.video_dir, args.output_dir, fps=args.fps)
