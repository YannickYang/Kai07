#!/usr/bin/env python3
"""Interactive subtask annotation tool for LeRobot datasets.

Generates contact-sheet images for visual review, then collects
subtask boundary annotations via terminal input.

Uses PyAV for AV1 video decoding (cv2 doesn't support software AV1).

Usage:
    # Step 1: Generate contact sheets for all episodes
    python scripts/annotate_subtasks.py --data-dir /path/to/dataset --preview

    # Step 2: Annotate interactively (one episode at a time)
    python scripts/annotate_subtasks.py --data-dir /path/to/dataset --episode 0

    # Step 3: Annotate a range of episodes
    python scripts/annotate_subtasks.py --data-dir /path/to/dataset --range 0-9

    # Batch import from CSV
    python scripts/annotate_subtasks.py --data-dir /path/to/dataset --from-csv annotations.csv

    # Generate defaults (1 subtask = whole episode)
    python scripts/annotate_subtasks.py --data-dir /path/to/dataset --generate-defaults
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_episode_info(data_dir: str) -> list[dict]:
    """Load episode metadata from episodes.jsonl."""
    episodes_path = os.path.join(data_dir, "meta", "episodes.jsonl")
    episodes = []
    with open(episodes_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def read_frame_pyav(video_path: str, frame_idx: int) -> np.ndarray | None:
    """Read a single frame from video using PyAV (supports AV1)."""
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        # Seek to approximate position then decode
        fps = float(stream.average_rate)
        target_pts = int(frame_idx / fps / stream.time_base)
        container.seek(max(0, target_pts - int(1 / stream.time_base)), stream=stream)

        current_idx = 0
        # We need to count from 0 since seek might not be frame-accurate
        container.seek(0, stream=stream)
        for i, frame in enumerate(container.decode(video=0)):
            if i == frame_idx:
                img = frame.to_ndarray(format='rgb24')
                container.close()
                return img
            if i > frame_idx:
                break
        container.close()
    except Exception as e:
        print(f"  [WARN] Failed to read frame {frame_idx} from {video_path}: {e}")
    return None


def read_frames_pyav(video_path: str, frame_indices: list[int]) -> dict[int, np.ndarray]:
    """Read multiple frames efficiently using PyAV (sequential decode)."""
    frames = {}
    if not frame_indices:
        return frames

    target_set = set(frame_indices)
    max_idx = max(frame_indices)

    try:
        container = av.open(video_path)
        for i, frame in enumerate(container.decode(video=0)):
            if i in target_set:
                frames[i] = frame.to_ndarray(format='rgb24')
            if i >= max_idx:
                break
        container.close()
    except Exception as e:
        print(f"  [WARN] Failed to read frames from {video_path}: {e}")

    return frames


def get_video_info(video_path: str) -> dict:
    """Get video metadata using PyAV."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    info = {
        "width": stream.width,
        "height": stream.height,
        "frames": stream.frames,
        "fps": float(stream.average_rate),
        "codec": stream.codec_context.name,
    }
    container.close()
    return info


def get_video_path(data_dir: str, episode_idx: int, camera: str = "observation.images.head") -> str:
    chunk_idx = episode_idx // 1000
    return os.path.join(
        data_dir, "videos", f"chunk-{chunk_idx:03d}", camera, f"episode_{episode_idx:06d}.mp4"
    )


def make_contact_sheet(
    data_dir: str,
    episode_idx: int,
    total_frames: int,
    output_path: str,
    camera: str = "observation.images.head",
    stride: int = 50,
    thumb_w: int = 240,
    thumb_h: int = 180,
    cols: int = 10,
):
    """Generate a contact sheet image showing sampled frames with timestamps."""
    video_path = get_video_path(data_dir, episode_idx, camera)
    if not os.path.exists(video_path):
        print(f"  [WARN] Video not found: {video_path}")
        return

    frame_indices = list(range(0, total_frames, stride))
    print(f"  Extracting {len(frame_indices)} frames (every {stride} frames = {stride/30:.1f}s)...")

    frames_dict = read_frames_pyav(video_path, frame_indices)

    rows = (len(frame_indices) + cols - 1) // cols
    label_h = 22
    cell_h = thumb_h + label_h
    sheet = Image.new("RGB", (cols * thumb_w, rows * cell_h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)

    for i, fidx in enumerate(frame_indices):
        r, c = i // cols, i % cols
        x = c * thumb_w
        y = r * cell_h

        if fidx in frames_dict:
            thumb = Image.fromarray(frames_dict[fidx]).resize((thumb_w, thumb_h))
            sheet.paste(thumb, (x, y))

        time_sec = fidx / 30.0
        label = f"f{fidx} ({time_sec:.1f}s)"
        draw.text((x + 4, y + thumb_h + 3), label, fill=(220, 220, 100))

    sheet.save(output_path)
    print(f"  Saved contact sheet: {output_path}")


def annotate_episode_interactive(
    data_dir: str, episode_idx: int, episode_length: int, sheet_dir: str
) -> list[dict]:
    """Interactively annotate subtasks for one episode via terminal."""
    print(f"\n{'='*60}")
    print(f"  Episode {episode_idx}: {episode_length} frames ({episode_length/30:.1f}s)")
    print(f"{'='*60}")

    # Generate contact sheet if not present
    sheet_path = os.path.join(sheet_dir, f"ep{episode_idx:04d}.png")
    if not os.path.exists(sheet_path):
        make_contact_sheet(data_dir, episode_idx, episode_length, sheet_path)
    else:
        print(f"  Contact sheet: {sheet_path}")

    print(f"\n  查看缩略图后标注子任务。")
    print(f"  常用subtask: Grasp the cloth / Flatten the cloth / Fold the cloth")
    print(f"  输入格式: 结束帧号 + 子任务描述")
    print(f"  输入 'q' 跳过此episode\n")

    subtasks = []
    current_start = 0

    while current_start < episode_length:
        time_str = f"{current_start/30:.1f}s"
        print(f"  --- 从 frame {current_start} ({time_str}) 开始 ---")

        # Get end frame
        while True:
            prompt = f"  结束帧 [{current_start+1}-{episode_length}] (直接回车={episode_length}): "
            end_input = input(prompt).strip()

            if end_input.lower() == 'q':
                return []
            if end_input == "" or end_input.lower() == "end":
                end_frame = episode_length
                break
            try:
                end_frame = int(end_input)
                if current_start < end_frame <= episode_length:
                    break
                print(f"  无效: 需在 {current_start+1} - {episode_length} 之间")
            except ValueError:
                print("  请输入数字或直接回车")

        # Get subtask text
        text = input("  Subtask描述 (英文): ").strip()
        if not text:
            text = "complete task"

        subtasks.append({
            "start": current_start,
            "end": end_frame,
            "text": text,
        })

        end_time = f"{end_frame/30:.1f}s"
        print(f"  -> [{current_start}-{end_frame}] ({time_str}-{end_time}) \"{text}\"")

        current_start = end_frame
        if current_start >= episode_length:
            break

    return subtasks


def annotate_from_csv(csv_path: str) -> dict[str, list[dict]]:
    """Import annotations from CSV. Format: episode_idx,start_frame,end_frame,subtask_text"""
    annotations = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep_idx = str(row["episode_idx"])
            if ep_idx not in annotations:
                annotations[ep_idx] = []
            annotations[ep_idx].append({
                "start": int(row["start_frame"]),
                "end": int(row["end_frame"]),
                "text": row["subtask_text"].strip(),
            })
    return annotations


def generate_default_subtasks(episodes: list[dict], task_text: str) -> dict[str, list[dict]]:
    """Generate default: whole episode = one subtask."""
    annotations = {}
    for ep in episodes:
        ep_idx = str(ep["episode_index"])
        annotations[ep_idx] = [{
            "start": 0,
            "end": ep["length"],
            "text": task_text,
        }]
    return annotations


def main():
    parser = argparse.ArgumentParser(description="Annotate subtasks for LeRobot dataset")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--episode", type=int, default=None, help="Annotate one episode")
    parser.add_argument("--range", type=str, default=None, help="Annotate range, e.g. 0-9")
    parser.add_argument("--preview", action="store_true", help="Generate contact sheets only (no annotation)")
    parser.add_argument("--from-csv", type=str, default=None, help="Import from CSV")
    parser.add_argument("--generate-defaults", action="store_true", help="Generate default (1 subtask/ep)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--stride", type=int, default=50, help="Frame stride for contact sheets")
    parser.add_argument("--camera", type=str, default="observation.images.head")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_path = args.output or os.path.join(data_dir, "subtask_annotations.json")
    sheet_dir = os.path.join(data_dir, "contact_sheets")
    os.makedirs(sheet_dir, exist_ok=True)

    episodes = load_episode_info(data_dir)
    print(f"Loaded {len(episodes)} episodes from {data_dir}")

    # Task text
    tasks_path = os.path.join(data_dir, "meta", "tasks.jsonl")
    task_text = "complete task"
    if os.path.exists(tasks_path):
        with open(tasks_path, "r") as f:
            first = f.readline().strip()
            if first:
                task_text = json.loads(first).get("task", task_text)
    print(f"Task: {task_text}")

    # Load existing
    annotations = {}
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            annotations = json.load(f)
        print(f"Existing annotations: {len(annotations)} episodes")

    # === Preview mode: generate contact sheets only ===
    if args.preview:
        print(f"\nGenerating contact sheets to: {sheet_dir}")
        for ep in episodes:
            idx = ep["episode_index"]
            sheet_path = os.path.join(sheet_dir, f"ep{idx:04d}.png")
            if os.path.exists(sheet_path):
                continue
            make_contact_sheet(data_dir, idx, ep["length"], sheet_path, args.camera, args.stride)
        print("Done. View images in:", sheet_dir)
        return

    # === CSV import ===
    if args.from_csv:
        csv_anns = annotate_from_csv(args.from_csv)
        annotations.update(csv_anns)
        print(f"Imported {len(csv_anns)} episodes from CSV")

    # === Default generation ===
    elif args.generate_defaults:
        annotations = generate_default_subtasks(episodes, task_text)
        print(f"Generated defaults for {len(annotations)} episodes")

    # === Interactive annotation ===
    else:
        if args.episode is not None:
            ep_indices = [args.episode]
        elif args.range:
            lo, hi = args.range.split("-")
            ep_indices = list(range(int(lo), int(hi) + 1))
        else:
            ep_indices = [ep["episode_index"] for ep in episodes]

        annotated_count = 0
        for ep_idx in ep_indices:
            ep_info = next((e for e in episodes if e["episode_index"] == ep_idx), None)
            if ep_info is None:
                print(f"Episode {ep_idx} not found")
                continue

            ep_key = str(ep_idx)
            if ep_key in annotations:
                existing = annotations[ep_key]
                texts = [s["text"] for s in existing]
                print(f"\nEpisode {ep_idx} 已标注: {texts}")
                ans = input("  覆盖? [y/N]: ").strip().lower()
                if ans != "y":
                    continue

            subtasks = annotate_episode_interactive(
                data_dir, ep_idx, ep_info["length"], sheet_dir
            )

            if not subtasks:
                print(f"  跳过 Episode {ep_idx}")
                continue

            annotations[ep_key] = subtasks
            annotated_count += 1

            print(f"\n  Episode {ep_idx} 标注结果:")
            for st in subtasks:
                print(f"    [{st['start']:>5d} - {st['end']:>5d}] {st['text']}")

            # Save after each episode
            with open(output_path, "w") as f:
                json.dump(annotations, f, indent=2, ensure_ascii=False)

        print(f"\n本次标注: {annotated_count} episodes")

    # Final save
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    total = len(annotations)
    total_eps = len(episodes)
    print(f"\n已保存: {output_path}")
    print(f"标注进度: {total}/{total_eps} episodes ({100*total/total_eps:.0f}%)")


if __name__ == "__main__":
    main()
