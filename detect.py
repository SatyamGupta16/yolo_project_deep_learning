"""
DL-017: Object Detection with a Pretrained YOLO Model
-----------------------------------------------------
What this script does (maps to the problem statement):
  1. Loads PRETRAINED weights (YOLOv8n trained on the COCO dataset, 80 classes)
  2. Runs INFERENCE on every image in a folder (and optionally a short video)
  3. Repeats it for several CONFIDENCE THRESHOLDS so you can compare them
  4. Saves images with BOXES + LABELS + CONFIDENCE SCORES drawn on them
  5. Writes a CSV of all detections (use it to REVIEW ERRORS by hand)
  6. Makes Matplotlib charts for your report

Run:  python detect.py --source images
      python detect.py --source clip1.mp4
"""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # save figures to files; no pop-up window needed
import matplotlib.pyplot as plt
from ultralytics import YOLO

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args():
    p = argparse.ArgumentParser(description="Object detection with pretrained YOLO")
    p.add_argument("--source", default="images", help="folder with images/videos, or a single file")
    p.add_argument("--model", default="yolov8n.pt", help="pretrained weights (auto-downloaded first time)")
    p.add_argument("--conf", type=float, nargs="+", default=[0.10, 0.25, 0.50, 0.75],
                   help="confidence thresholds to compare")
    p.add_argument("--output", default="outputs", help="where results are saved")
    p.add_argument("--video-conf", type=float, default=0.25, help="threshold used for videos")
    p.add_argument("--max-frames", type=int, default=150, help="max frames to process per video")
    return p.parse_args()


def collect_files(source):
    src = Path(source)
    if not src.exists():
        raise SystemExit(f"'{source}' does not exist. Check the file/folder name "
                         f"(current folder: {Path.cwd()})")
    if src.is_file():
        files = [src]
    else:
        files = sorted(f for f in src.iterdir() if f.is_file())
    images = [f for f in files if f.suffix.lower() in IMAGE_EXT]
    videos = [f for f in files if f.suffix.lower() in VIDEO_EXT]
    return images, videos


def run_on_image(model, image_path, conf):
    """Run the model on one image. Returns (annotated BGR image, list of detections)."""
    result = model.predict(str(image_path), conf=conf, verbose=False)[0]
    detections = []
    for box, cls_id, score in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
        x1, y1, x2, y2 = [round(float(v), 1) for v in box]
        detections.append({
            "class": result.names[int(cls_id)],
            "confidence": round(float(score), 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    return result.plot(), detections  # result.plot() draws boxes + labels + scores


def process_images(model, images, confs, out_dir):
    rows = []
    summary = {c: {"count": 0, "conf_sum": 0.0} for c in confs}
    for conf in confs:
        folder = out_dir / "annotated" / f"conf_{conf:.2f}"
        folder.mkdir(parents=True, exist_ok=True)
        for img in images:
            annotated, dets = run_on_image(model, img, conf)
            cv2.imwrite(str(folder / img.name), annotated)
            for d in dets:
                rows.append({"image": img.name, "threshold": conf, **d,
                             "review (TP/FP)": ""})  # fill this in by hand later
                summary[conf]["count"] += 1
                summary[conf]["conf_sum"] += d["confidence"]
            print(f"[conf={conf:.2f}] {img.name}: {len(dets)} detections")
    return rows, summary


def save_csv(rows, out_dir):
    if not rows:
        print("No detections found - nothing to write to CSV.")
        return
    path = out_dir / "detections.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def plot_summary(summary, out_dir):
    confs = list(summary.keys())
    counts = [summary[c]["count"] for c in confs]
    avg_conf = [summary[c]["conf_sum"] / summary[c]["count"] if summary[c]["count"] else 0
                for c in confs]
    labels = [f"{c:.2f}" for c in confs]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, counts, color="steelblue")
    axes[0].set_title("Detections vs confidence threshold")
    axes[0].set_xlabel("Confidence threshold")
    axes[0].set_ylabel("Total detections")
    axes[1].plot(labels, avg_conf, marker="o", color="darkorange")
    axes[1].set_title("Average confidence of kept detections")
    axes[1].set_xlabel("Confidence threshold")
    axes[1].set_ylabel("Average confidence")
    axes[1].set_ylim(0, 1)
    plt.tight_layout()
    path = out_dir / "threshold_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_comparison(images, confs, out_dir):
    """Side-by-side view of the same image at every threshold."""
    for img in images[:3]:  # first 3 images keeps it small
        fig, axes = plt.subplots(1, len(confs), figsize=(4 * len(confs), 4))
        if len(confs) == 1:
            axes = [axes]
        for ax, conf in zip(axes, confs):
            shown = cv2.imread(str(out_dir / "annotated" / f"conf_{conf:.2f}" / img.name))
            ax.imshow(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB))  # OpenCV is BGR, Matplotlib wants RGB
            ax.set_title(f"conf >= {conf:.2f}")
            ax.axis("off")
        plt.tight_layout()
        path = out_dir / f"compare_{img.stem}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"Saved {path}")


def process_video(model, video_path, conf, max_frames, out_dir):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = out_dir / f"{video_path.stem}_detected.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_no, total = 0, 0
    while frame_no < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(frame, conf=conf, verbose=False)[0]
        total += len(result.boxes)
        writer.write(result.plot())
        frame_no += 1
    cap.release()
    writer.release()
    print(f"Video {video_path.name}: {frame_no} frames, {total} detections -> {out_path}")


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    images, videos = collect_files(args.source)
    if not images and not videos:
        raise SystemExit(f"No images or videos found in '{args.source}'")
    print(f"Found {len(images)} image(s) and {len(videos)} video(s)")

    model = YOLO(args.model)  # loads pretrained weights (downloads on first run)

    if images:
        rows, summary = process_images(model, images, args.conf, out_dir)
        save_csv(rows, out_dir)
        plot_summary(summary, out_dir)
        plot_comparison(images, args.conf, out_dir)

    for video in videos:
        process_video(model, video, args.video_conf, args.max_frames, out_dir)

    print("\nDone. Open the 'outputs' folder to see results.")


if __name__ == "__main__":
    main()