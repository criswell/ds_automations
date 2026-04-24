"""
Video audio event detector.
Scans MP4/MKV video files for coughing, laughing, and loud shouting,
then writes a combined timestamp report to a text file.

For multi-channel audio, the script automatically identifies and uses only
the channels most likely to contain human speech, ignoring noise/gameplay channels.

Usage:
    python video_event_detector.py <input_dir_or_file> <output.txt> [--threshold 0.3] [--speech-channels 2]
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
FFPROBE_PATH = os.path.expandvars(
    r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffprobe.exe"
)

# AudioSet class labels that map to each event type.
# These strings must match the labels in the PANNs label list exactly.
EVENT_CLASS_MAP = {
    "LAUGH": ["Laughter", "Chuckle, chortle", "Giggle"],
    "COUGH": ["Cough", "Throat clearing"],
    "SHOUT": ["Shout", "Yell", "Screaming", "Battle cry"],
    "BURP" : ["Burping, eructation", "Gargling", "Stomach rumble"],
}

SAMPLE_RATE = 16000  # AST model expects 16 kHz
CHUNK_SECONDS = 2.0
HOP_SECONDS = 1.0
MERGE_GAP_SECONDS = 2.0
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"


def probe_audio_streams(video_path: str) -> list:
    """
    Return a list of dicts, one per audio stream: {stream_index, channels}.
    Uses ffprobe to inspect the file without decoding any audio.
    """
    cmd = [
        FFPROBE_PATH,
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "a",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {video_path}:\n{result.stderr.decode(errors='replace')}"
        )
    data = json.loads(result.stdout)
    return [
        {"stream_index": s["index"], "channels": s.get("channels", 1)}
        for s in data.get("streams", [])
    ]


def extract_mono_audio(video_path: str, wav_path: str) -> None:
    """Extract all audio downmixed to mono at SAMPLE_RATE."""
    cmd = [
        FFMPEG_PATH, "-y", "-i", video_path,
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-vn",
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {video_path}:\n{result.stderr.decode(errors='replace')}"
        )


def extract_channel(video_path: str, wav_path: str, stream_index: int, channel_index: int) -> None:
    """Extract a single audio channel as a mono WAV using the pan filter."""
    cmd = [
        FFMPEG_PATH, "-y", "-i", video_path,
        "-map", f"0:{stream_index}",
        "-filter:a", f"pan=mono|c0=c{channel_index}",
        "-ar", str(SAMPLE_RATE),
        "-vn",
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed extracting stream {stream_index} channel {channel_index} "
            f"from {video_path}:\n{result.stderr.decode(errors='replace')}"
        )


def score_channel_for_speech(wav_path: str) -> float:
    """
    Score a mono audio channel for speech likelihood (higher = more speech-like).
    Combines speech-band energy ratio with spectral non-flatness.
    Speech concentrates energy in 300-4000 Hz and has lower spectral flatness than
    broadband noise or music-heavy gameplay audio.
    """
    import numpy as np
    import librosa

    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

    # Sample 4 windows of 60s spread evenly across the full audio so that long
    # silent intros don't cause a speech channel to be misclassified as noise.
    window_samples = int(60 * SAMPLE_RATE)
    n_windows = 4
    if len(audio) <= window_samples:
        sample = audio
    else:
        max_start = len(audio) - window_samples
        starts = [int(i * max_start / (n_windows - 1)) for i in range(n_windows)]
        sample = np.concatenate([audio[s : s + window_samples] for s in starts])

    stft = np.abs(librosa.stft(sample))
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE)

    speech_mask = (freqs >= 300) & (freqs <= 4000)
    total_energy = stft.sum() + 1e-10
    energy_ratio = float(stft[speech_mask].sum() / total_energy)

    # spectral_flatness ≈ 1 for white noise, ≈ 0 for pure tones; speech is mid-range
    # Gameplay audio (music + effects) tends to be broader and flatter than isolated speech
    flatness = float(librosa.feature.spectral_flatness(S=stft).mean())

    return energy_ratio * (1.0 - flatness)


def enumerate_channels(video_path: str) -> list:
    """
    Return a flat list of (label, stream_index, channel_index) for every
    audio channel in the video, across all audio streams.
    """
    streams = probe_audio_streams(video_path)
    channels = []
    for s in streams:
        si = s["stream_index"]
        for ci in range(s["channels"]):
            label = f"stream{si}_ch{ci}"
            channels.append((label, si, ci))
    return channels


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

    id2label = model.config.id2label
    label2id = {v.lower(): k for k, v in id2label.items()}

    target_indices = {}
    for event_type, class_names in EVENT_CLASS_MAP.items():
        indices = [label2id[n.lower()] for n in class_names if n.lower() in label2id]
        target_indices[event_type] = indices
        if not indices:
            print(f"  WARNING: no label indices found for {event_type} — check EVENT_CLASS_MAP")

    return model, feature_extractor, target_indices


def detect_events_raw(wav_path: str, model, feature_extractor, target_indices: dict, threshold: float) -> list:
    """
    Slide a window over the audio and return raw (timestamp_sec, event_type, confidence) tuples.
    Does not merge — callers that aggregate across channels should merge afterwards.
    """
    import torch
    import librosa

    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
    hop_samples = int(HOP_SECONDS * SAMPLE_RATE)

    raw_detections = []
    pos = 0
    while pos < len(audio):
        chunk = audio[pos: pos + chunk_samples]
        if len(chunk) < hop_samples:
            break
        inputs = feature_extractor(chunk.tolist(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.sigmoid(logits[0])
        timestamp = pos / SAMPLE_RATE
        for event_type, indices in target_indices.items():
            if not indices:
                continue
            confidence = float(max(probs[i].item() for i in indices))
            if confidence >= threshold:
                raw_detections.append((timestamp, event_type, confidence))
        pos += hop_samples

    return raw_detections


def merge_detections(raw: list) -> list:
    """
    Merge detections of the same event type that occur within MERGE_GAP_SECONDS
    of each other. Returns sorted list of (timestamp_sec, event_type).
    """
    if not raw:
        return []

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
            merged.append((group[0], ev))

    merged.sort(key=lambda x: x[0])
    return merged


def process_video(video_path, model, feature_extractor, target_indices, threshold, n_speech_channels, tmp_dir):
    """
    Extract audio channels, select the most speech-like ones, run detection across all
    selected channels, and return a merged list of (timestamp_sec, event_type).
    """
    channels = enumerate_channels(str(video_path))
    total_channels = len(channels)

    if total_channels <= 1:
        wav_path = os.path.join(tmp_dir, "mono.wav")
        extract_mono_audio(str(video_path), wav_path)
        raw = detect_events_raw(wav_path, model, feature_extractor, target_indices, threshold)
        return merge_detections(raw)

    print(f"  Found {total_channels} audio channel(s) — scoring for speech content...")

    # Extract each channel and score it
    scored = []
    for label, si, ci in channels:
        wav_path = os.path.join(tmp_dir, f"{label}.wav")
        extract_channel(str(video_path), wav_path, si, ci)
        score = score_channel_for_speech(wav_path)
        scored.append((label, wav_path, score))

    scored.sort(key=lambda x: x[2], reverse=True)

    n_select = min(n_speech_channels, total_channels)
    selected_labels = {s[0] for s in scored[:n_select]}

    for label, wav_path, score in scored:
        marker = "<-- selected" if label in selected_labels else ""
        print(f"    {label}: speech score {score:.3f}  {marker}")

    # Collect raw detections across all selected channels, then merge together
    all_raw = []
    for label, wav_path, score in scored[:n_select]:
        print(f"  Detecting events in {label}...")
        raw = detect_events_raw(wav_path, model, feature_extractor, target_indices, threshold)
        all_raw.extend(raw)

    return merge_detections(all_raw)


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
    if progress_path.exists():
        try:
            return json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"completed": {}}


def save_progress(progress_path: Path, progress: dict) -> None:
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def write_output(output_path: Path, video_files: list, progress: dict) -> None:
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
        "--threshold", type=float, default=0.3,
        help="Confidence threshold (0.0-1.0, default: 0.3)",
    )
    parser.add_argument(
        "--speech-channels", type=int, default=2,
        help=(
            "Number of audio channels to treat as speech (default: 2). "
            "The script scores every channel and selects the N with the highest "
            "speech-content score, ignoring the rest."
        ),
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

            with tempfile.TemporaryDirectory() as tmp_dir:
                try:
                    print("  Extracting audio...")
                    events = process_video(
                        video_path, model, feature_extractor, target_indices,
                        args.threshold, args.speech_channels, tmp_dir,
                    )

                    if events:
                        for ts, ev in events:
                            print(f"  [{format_timestamp(ts)}] {ev}")
                    else:
                        print("  No events detected.")

                    progress["completed"][str(video_path)] = events

                except Exception as e:
                    print(f"  ERROR: {e}")
                    progress["completed"][str(video_path)] = []

            save_progress(progress_path, progress)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Progress saved — rerun the same command to continue.")
        save_progress(progress_path, progress)
        write_output(output_path, video_files, progress)
        print(f"Partial results written to: {args.output}")
        sys.exit(0)

    write_output(output_path, video_files, progress)
    print(f"\nDone. Results written to: {args.output}")

    try:
        progress_path.unlink()
    except OSError:
        pass


if __name__ == "__main__":
    main()
