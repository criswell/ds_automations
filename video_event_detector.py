"""
Video audio event detector.
Scans MP4/MKV video files for coughing, laughing, and loud shouting,
then writes a combined timestamp report to a text file.

Usage:
    python video_event_detector.py <input_dir_or_file> <output.txt> [--threshold 0.3]
"""

import sys
import os
import json
import subprocess
import tempfile
import argparse
from pathlib import Path

FFMPEG_PATH = os.path.expandvars(
    r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"
)

# AudioSet class labels that map to each event type.
# These strings must match the labels in the PANNs label list exactly.
EVENT_CLASS_MAP = {
    "LAUGH": ["Laughter", "Chuckle, chortle", "Giggle"],
    "COUGH": ["Cough", "Throat clearing"],
    "SHOUT": ["Shout", "Yell", "Screaming", "Battle cry"],
}

SAMPLE_RATE = 16000  # AST model expects 16 kHz
CHUNK_SECONDS = 2.0
HOP_SECONDS = 1.0
MERGE_GAP_SECONDS = 2.0
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"


def extract_audio(video_path: str, wav_path: str) -> None:
    """Extract mono audio at SAMPLE_RATE from a video file using ffmpeg."""
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-vn",
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {video_path}:\n{result.stderr.decode(errors='replace')}"
        )


def load_model():
    """
    Load the Audio Spectrogram Transformer model from HuggingFace.
    Downloads weights on first run (~300 MB), cached locally afterwards.
    Returns (model, feature_extractor, target_indices).
    """
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    print(f"  Loading feature extractor from {MODEL_NAME}...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    print(f"  Loading model weights (downloading ~300 MB on first run)...")
    model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    model.eval()

    # Build label name -> index mapping from the model config
    id2label = model.config.id2label  # {0: "Speech", 1: "Male speech, man speaking", ...}
    label2id = {v.lower(): k for k, v in id2label.items()}

    target_indices = {}
    for event_type, class_names in EVENT_CLASS_MAP.items():
        indices = []
        for name in class_names:
            idx = label2id.get(name.lower())
            if idx is not None:
                indices.append(idx)
        target_indices[event_type] = indices
        if not indices:
            print(f"  WARNING: no label indices found for {event_type} — check EVENT_CLASS_MAP")

    return model, feature_extractor, target_indices


def detect_events(wav_path: str, model, feature_extractor, target_indices: dict, threshold: float) -> list:
    """
    Slide a window over the audio and return a list of (timestamp_sec, event_type) tuples.
    """
    import torch
    import numpy as np
    import librosa

    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

    chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
    hop_samples = int(HOP_SECONDS * SAMPLE_RATE)

    raw_detections = []  # (timestamp_sec, event_type, confidence)

    pos = 0
    while pos < len(audio):
        chunk = audio[pos: pos + chunk_samples]
        if len(chunk) < hop_samples:
            break

        inputs = feature_extractor(
            chunk.tolist(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits  # shape (1, 527)
        probs = torch.sigmoid(logits[0])     # multi-label probabilities

        timestamp = pos / SAMPLE_RATE

        for event_type, indices in target_indices.items():
            if not indices:
                continue
            confidence = float(max(probs[i].item() for i in indices))
            if confidence >= threshold:
                raw_detections.append((timestamp, event_type, confidence))

        pos += hop_samples

    return merge_detections(raw_detections)


def merge_detections(raw: list) -> list:
    """
    Merge detections of the same event type that occur within MERGE_GAP_SECONDS
    of each other. Returns sorted list of (timestamp_sec, event_type).
    """
    if not raw:
        return []

    # Group by event type
    by_type = {}
    for ts, ev, conf in raw:
        by_type.setdefault(ev, []).append(ts)

    merged = []
    for ev, timestamps in by_type.items():
        timestamps.sort()
        groups = [[timestamps[0]]]
        for ts in timestamps[1:]:
            if ts - groups[-1][-1] <= MERGE_GAP_SECONDS:
                groups[-1].append(ts)
            else:
                groups.append([ts])
        for group in groups:
            # Use the first timestamp of the group
            merged.append((group[0], ev))

    merged.sort(key=lambda x: x[0])
    return merged


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def find_video_files(input_path: str) -> list:
    p = Path(input_path)
    if p.is_file():
        return [p]
    extensions = {".mp4", ".mkv"}
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in extensions)


def load_progress(progress_path: Path) -> dict:
    """Load existing progress from a JSON file, or return empty state."""
    if progress_path.exists():
        try:
            return json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"completed": {}}


def save_progress(progress_path: Path, progress: dict) -> None:
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def write_output(output_path: Path, video_files: list, progress: dict) -> None:
    """Write the final report from completed progress data, preserving file order."""
    lines = []
    for video_path in video_files:
        key = str(video_path)
        if key not in progress["completed"]:
            continue
        lines.append(f"=== {video_path.name} ===")
        events = progress["completed"][key]
        if events:
            for ts, ev in events:
                lines.append(f"[{format_timestamp(ts)}] {ev}")
        else:
            lines.append("  (no events detected)")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Detect audio events in video files.")
    parser.add_argument("input", help="Video file or directory of video files")
    parser.add_argument("output", help="Output text file path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence threshold (0.0-1.0, default: 0.3)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    progress_path = output_path.with_suffix(".progress.json")

    video_files = find_video_files(args.input)
    if not video_files:
        print(f"No MP4/MKV files found at: {args.input}")
        sys.exit(1)

    progress = load_progress(progress_path)
    completed_keys = set(progress["completed"].keys())

    remaining = [f for f in video_files if str(f) not in completed_keys]
    already_done = len(video_files) - len(remaining)

    if already_done:
        print(f"Resuming: {already_done} file(s) already done, {len(remaining)} remaining.")
    else:
        print(f"Found {len(video_files)} file(s) to process.")

    if not remaining:
        print("All files already processed. Writing output.")
        write_output(output_path, video_files, progress)
        print(f"Done. Results written to: {args.output}")
        return

    print("Loading Audio Spectrogram Transformer model...")
    model, feature_extractor, target_indices = load_model()
    print("Model loaded.\n")

    try:
        for i, video_path in enumerate(remaining, 1):
            print(f"[{i}/{len(remaining)}] Processing: {video_path.name}")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_wav = tmp.name

            try:
                print("  Extracting audio...")
                extract_audio(str(video_path), tmp_wav)

                print("  Detecting events...")
                events = detect_events(tmp_wav, model, feature_extractor, target_indices, args.threshold)

                if events:
                    for ts, ev in events:
                        print(f"  [{format_timestamp(ts)}] {ev}")
                else:
                    print("  No events detected.")

                progress["completed"][str(video_path)] = events

            except Exception as e:
                print(f"  ERROR: {e}")
                progress["completed"][str(video_path)] = []

            finally:
                try:
                    os.unlink(tmp_wav)
                except OSError:
                    pass

            # Save progress after every file so interruption loses at most one file
            save_progress(progress_path, progress)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Progress saved — rerun the same command to continue.")
        save_progress(progress_path, progress)
        write_output(output_path, video_files, progress)
        print(f"Partial results written to: {args.output}")
        sys.exit(0)

    write_output(output_path, video_files, progress)
    print(f"\nDone. Results written to: {args.output}")

    # Clean up progress file on successful completion
    try:
        progress_path.unlink()
    except OSError:
        pass


if __name__ == "__main__":
    main()
