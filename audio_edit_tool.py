"""
Semi-automated audio editing tool.
Step 1: Transcribe audio with faster-whisper
Step 2: Align transcription to script using fuzzy matching
Step 3: Generate Edit Decision List (EDL)
Step 5: Assemble final audio from EDL
"""

import sys
import json
import os
import re
from pathlib import Path

# Point pydub at ffmpeg installed by winget
FFMPEG_PATH = os.path.expandvars(
    r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"
)
FFPROBE_PATH = os.path.expandvars(
    r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffprobe.exe"
)

def transcribe(audio_path: str, output_path: str, model_size: str = "medium"):
    """Step 1: Transcribe audio with word-level timestamps."""
    from faster_whisper import WhisperModel

    print(f"Loading Whisper model '{model_size}'...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print(f"Transcribing {audio_path}...")
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=True,
        language="en",
    )

    results = []
    for segment in segments:
        seg_data = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "avg_logprob": segment.avg_logprob,
            "no_speech_prob": segment.no_speech_prob,
            "words": [],
        }
        if segment.words:
            for w in segment.words:
                seg_data["words"].append({
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "probability": w.probability,
                })
        results.append(seg_data)
        # Print progress
        print(f"  [{segment.start:.1f}s - {segment.end:.1f}s] {segment.text.strip()}")

    output = {
        "audio_file": audio_path,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nTranscription saved to {output_path}")
    print(f"Total segments: {len(results)}")
    print(f"Audio duration: {info.duration:.1f}s")
    return output


def parse_script(script_path: str, start_line: str = None):
    """Parse the script file into individual lines for matching.

    Filters out:
    - Markdown headers (lines starting with #)
    - Stage directions (lines in parentheses)
    - URLs
    - Empty lines
    - Timestamp-only lines

    If start_line is provided, only includes lines from that point onward.
    """
    with open(script_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # If start_line specified, trim to that point
    if start_line:
        idx = raw_text.find(start_line)
        if idx == -1:
            print(f"WARNING: Could not find start line '{start_line[:50]}...'")
            print("Using full script.")
        else:
            raw_text = raw_text[idx:]

    lines = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Skip markdown headers
        if line.startswith("#"):
            continue
        # Skip stage directions (entire line in parens)
        if line.startswith("(") and line.endswith(")"):
            continue
        # Skip URLs
        if line.startswith("http://") or line.startswith("https://"):
            continue
        # Skip timestamp-only lines (e.g., "8:24")
        if re.match(r"^\d+:\d+$", line):
            continue
        # Skip bullet points that are just stage directions
        lines.append(line)

    return lines


def _group_segments_into_runs(segments, max_gap_s=3.0):
    """Group consecutive transcript segments into contiguous 'runs'.

    A run is a sequence of segments with gaps no larger than max_gap_s.
    This groups words spoken together into coherent phrases/sentences.
    """
    if not segments:
        return []

    runs = []
    current_run = {
        "segments": [0],
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "text": segments[0]["text"].strip(),
    }

    for i in range(1, len(segments)):
        gap = segments[i]["start"] - segments[i - 1]["end"]
        if gap <= max_gap_s:
            current_run["segments"].append(i)
            current_run["end"] = segments[i]["end"]
            current_run["text"] += " " + segments[i]["text"].strip()
        else:
            runs.append(current_run)
            current_run = {
                "segments": [i],
                "start": segments[i]["start"],
                "end": segments[i]["end"],
                "text": segments[i]["text"].strip(),
            }

    runs.append(current_run)
    return runs


def align_to_script(transcript_path: str, script_lines: list, output_path: str):
    """Step 2 & 3: Align transcription segments to script lines and generate EDL.

    Strategy: Sequential segment-to-line assignment.
    Since the recording follows the script order (with retakes and tangents),
    we assign each segment to the best-matching script line, then group
    consecutive same-line segments into takes.
    """
    from rapidfuzz import fuzz

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    segments = transcript["segments"]
    print(f"Processing {len(segments)} segments against {len(script_lines)} script lines")

    # Pre-normalize script lines and split into sentences for sub-matching
    script_norms = [line.lower().strip() for line in script_lines]
    # Also create sentence-level fragments for long script lines
    script_sentences = []
    for i, line in enumerate(script_lines):
        # Split on sentence boundaries
        sents = re.split(r'(?<=[.!?])\s+', line.strip())
        script_sentences.append([(s.lower().strip(), i) for s in sents if len(s.strip()) > 10])

    # Step 1: Assign each segment to best matching script line
    seg_assignments = []
    for j, seg in enumerate(segments):
        seg_text = seg["text"].lower().strip()
        if not seg_text or len(seg_text) < 3:
            seg_assignments.append({"line": -1, "score": 0})
            continue

        best_line = -1
        best_score = 0

        # Check against each script line's sentences
        for i, script_norm in enumerate(script_norms):
            # Direct partial match against full line
            score = fuzz.partial_ratio(seg_text, script_norm)

            # Also check individual sentences
            for sent_norm, sent_line in script_sentences[i]:
                sent_score = fuzz.ratio(seg_text, sent_norm)
                partial_sent = fuzz.partial_ratio(seg_text, sent_norm)
                score = max(score, sent_score, partial_sent)

            if score > best_score:
                best_score = score
                best_line = i

        # Only assign if confident enough
        if best_score >= 65:
            seg_assignments.append({"line": best_line, "score": best_score})
        else:
            seg_assignments.append({"line": -1, "score": best_score})

    # Step 2: Group consecutive segments assigned to the same line into takes
    line_takes = {i: [] for i in range(len(script_lines))}

    current_line = -1
    current_segs = []

    for j, assignment in enumerate(seg_assignments):
        line_idx = assignment["line"]

        if line_idx == current_line and line_idx >= 0:
            current_segs.append(j)
        else:
            # Flush current group
            if current_segs and current_line >= 0:
                line_takes[current_line].append(current_segs)
            current_line = line_idx
            current_segs = [j] if line_idx >= 0 else []

    # Flush last group
    if current_segs and current_line >= 0:
        line_takes[current_line].append(current_segs)

    # Step 3: Also try merging adjacent takes for the same line
    # (separated by brief non-script segments like "um", pauses)
    for i in range(len(script_lines)):
        merged = []
        for take_segs in line_takes[i]:
            if merged and take_segs:
                last_end = segments[merged[-1][-1]]["end"]
                next_start = segments[take_segs[0]]["start"]
                gap = next_start - last_end
                # If gap is small (< 5s), merge the takes
                if gap < 5.0:
                    merged[-1].extend(take_segs)
                    continue
            merged.append(take_segs)
        line_takes[i] = merged

    # Step 4: Build EDL
    edl = []
    for i, script_line in enumerate(script_lines):
        takes = []
        for take_seg_indices in line_takes[i]:
            if not take_seg_indices:
                continue
            combined_text = " ".join(segments[j]["text"].strip() for j in take_seg_indices)
            start = segments[take_seg_indices[0]]["start"]
            end = segments[take_seg_indices[-1]]["end"]

            # Compute quality metrics
            script_norm = script_norms[i]
            combined_norm = combined_text.lower().strip()
            overall_score = fuzz.ratio(combined_norm, script_norm)

            # Coverage: what fraction of script line words appear in take
            script_words = set(re.findall(r'\w+', script_norm))
            take_words = set(re.findall(r'\w+', combined_norm))
            coverage = len(script_words & take_words) / max(len(script_words), 1)

            takes.append({
                "start": start,
                "end": end,
                "text": combined_text,
                "match_score": overall_score,
                "coverage": round(coverage, 3),
                "duration": round(end - start, 3),
                "segment_indices": take_seg_indices,
            })

        # Sort: prefer high coverage, then high match score
        takes.sort(key=lambda t: (t["coverage"], t["match_score"]), reverse=True)
        default_idx = 0 if takes else None

        edl.append({
            "line_number": i + 1,
            "script_text": script_line,
            "takes": takes,
            "selected_take": default_idx,
        })

    # Identify non-script segments
    all_matched = set()
    for entry in edl:
        for take in entry["takes"]:
            all_matched.update(take["segment_indices"])

    non_script = []
    current_non = None
    for j, seg in enumerate(segments):
        if j not in all_matched:
            if current_non and seg["start"] - current_non["end"] < 3.0:
                current_non["end"] = seg["end"]
                current_non["text"] += " " + seg["text"].strip()
            else:
                if current_non:
                    non_script.append(current_non)
                current_non = {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                }
        else:
            if current_non:
                non_script.append(current_non)
                current_non = None
    if current_non:
        non_script.append(current_non)

    result = {
        "edl": edl,
        "non_script_segments": non_script,
        "total_script_lines": len(script_lines),
        "total_segments": len(segments),
        "matched_lines": sum(1 for e in edl if e["takes"]),
        "unmatched_lines": sum(1 for e in edl if not e["takes"]),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nEDL saved to {output_path}")
    print(f"Script lines: {result['total_script_lines']}")
    print(f"Matched: {result['matched_lines']}")
    print(f"Unmatched: {result['unmatched_lines']}")
    print(f"Non-script segments: {len(non_script)}")

    return result


def generate_human_edl(edl_path: str, output_path: str):
    """Generate a human-readable EDL for review."""
    with open(edl_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def fmt_time(seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{int(h)}:{int(m):02d}:{s:06.3f}"
        return f"{int(m)}:{s:06.3f}"

    lines = []
    lines.append("=" * 80)
    lines.append("EDIT DECISION LIST")
    lines.append(f"Total script lines: {data['total_script_lines']}")
    lines.append(f"Matched: {data['matched_lines']} | Unmatched: {data['unmatched_lines']}")
    lines.append("=" * 80)
    lines.append("")

    for entry in data["edl"]:
        ln = entry["line_number"]
        script = entry["script_text"]
        selected = entry["selected_take"]

        # Truncate long script lines for display
        display_script = script if len(script) <= 100 else script[:97] + "..."
        lines.append(f"Line {ln}: \"{display_script}\"")

        if not entry["takes"]:
            lines.append("  *** NO MATCH FOUND ***")
            lines.append("")
            continue

        for ti, take in enumerate(entry["takes"]):
            marker = " [SELECTED]" if ti == selected else ""
            coverage = take.get("coverage", 0)
            duration = take.get("duration", take["end"] - take["start"])
            lines.append(
                f"  Take {chr(65 + min(ti, 25))} ({fmt_time(take['start'])} - {fmt_time(take['end'])}) "
                f"match: {take['match_score']:.0f}  coverage: {coverage:.0%}  "
                f"dur: {duration:.1f}s{marker}"
            )
            take_text = take["text"] if len(take["text"]) <= 100 else take["text"][:97] + "..."
            lines.append(f"    \"{take_text}\"")

        lines.append("")

    if data["non_script_segments"]:
        lines.append("-" * 80)
        lines.append("NON-SCRIPT SEGMENTS (will be removed)")
        lines.append("-" * 80)
        for seg in data["non_script_segments"]:
            lines.append(
                f"  {fmt_time(seg['start'])} - {fmt_time(seg['end'])}: "
                f"\"{seg['text'][:80]}{'...' if len(seg['text']) > 80 else ''}\""
            )

    text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Human-readable EDL saved to {output_path}")
    return text


def assemble_audio(edl_path: str, audio_paths: list, source_info: list,
                    output_path: str, crossfade_ms: int = 50):
    """Step 5: Assemble final audio from EDL selections.

    Args:
        edl_path: Path to the EDL JSON file
        audio_paths: List of source audio file paths (in order)
        source_info: List of dicts with 'offset' and 'duration' for each file
                     (from merged_transcript.json's source_info)
        output_path: Path for the assembled output WAV
        crossfade_ms: Crossfade duration between segments in milliseconds
    """
    from pydub import AudioSegment

    AudioSegment.converter = FFMPEG_PATH
    AudioSegment.ffprobe = FFPROBE_PATH

    with open(edl_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load all source audio files
    audio_files = []
    for i, path in enumerate(audio_paths):
        print(f"Loading audio file {i + 1}/{len(audio_paths)}: {path}")
        audio_files.append(AudioSegment.from_wav(path))

    def get_clip(merged_start, merged_end):
        """Extract a clip using merged timeline coordinates."""
        for i, info in enumerate(source_info):
            file_offset = info["offset"]
            file_duration = info["duration"]
            file_end = file_offset + file_duration

            if merged_start >= file_offset and merged_start < file_end:
                # This take starts in this file
                local_start = merged_start - file_offset
                local_end = merged_end - file_offset

                # Clamp to file bounds
                local_end = min(local_end, file_duration)

                start_ms = int(local_start * 1000)
                end_ms = int(local_end * 1000)
                clip = audio_files[i][start_ms:end_ms]

                # If the take spans into the next file, append from next file
                if merged_end > file_end and i + 1 < len(source_info):
                    overflow = merged_end - file_end
                    next_clip = audio_files[i + 1][0:int(overflow * 1000)]
                    clip += next_clip

                return clip
        return None

    assembled = AudioSegment.empty()
    labels = []  # For Audacity labels on the assembled output
    segment_count = 0
    current_pos_ms = 0

    for entry in data["edl"]:
        selected_idx = entry["selected_take"]
        if selected_idx is None or not entry["takes"]:
            print(f"  WARNING: Line {entry['line_number']} has no selected take, skipping")
            continue

        take = entry["takes"][selected_idx]
        clip = get_clip(take["start"], take["end"])

        if clip is None:
            print(f"  ERROR: Could not extract clip for line {entry['line_number']}")
            continue

        # Track position for labels
        clip_start_ms = current_pos_ms

        if segment_count > 0 and crossfade_ms > 0 and len(assembled) > crossfade_ms:
            assembled = assembled.append(clip, crossfade=crossfade_ms)
            clip_start_ms = len(assembled) - len(clip)
        else:
            assembled += clip

        current_pos_ms = len(assembled)

        # Create label for this segment
        label_start = clip_start_ms / 1000.0
        label_end = current_pos_ms / 1000.0
        script_preview = entry["script_text"][:50]
        labels.append((label_start, label_end, f"L{entry['line_number']}: {script_preview}"))

        segment_count += 1

    assembled.export(output_path, format="wav")
    print(f"\nAssembled {segment_count} segments into {output_path}")
    print(f"Output duration: {len(assembled) / 1000:.1f}s")

    # Save Audacity labels for the assembled output
    labels_path = output_path.replace(".wav", "_labels.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        for start, end, label in labels:
            f.write(f"{start:.6f}\t{end:.6f}\t{label}\n")
    print(f"Audacity labels (assembled): {labels_path}")

    return labels


def generate_source_labels(edl_path: str, audio_paths: list, source_info: list,
                           output_dir: str):
    """Generate Audacity label files for each source audio file.

    These labels mark the selected takes in the original recordings,
    so you can open each source WAV in Audacity with labels showing
    what to keep and what to cut.
    """
    with open(edl_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build labels per source file
    file_labels = {i: [] for i in range(len(audio_paths))}

    for entry in data["edl"]:
        selected_idx = entry["selected_take"]
        if selected_idx is None or not entry["takes"]:
            continue

        take = entry["takes"][selected_idx]
        merged_start = take["start"]
        merged_end = take["end"]

        # Find which source file this belongs to
        for i, info in enumerate(source_info):
            file_offset = info["offset"]
            file_end = file_offset + info["duration"]

            if merged_start >= file_offset and merged_start < file_end:
                local_start = merged_start - file_offset
                local_end = min(merged_end - file_offset, info["duration"])
                script_preview = entry["script_text"][:50]
                file_labels[i].append(
                    (local_start, local_end,
                     f"L{entry['line_number']}: {script_preview}")
                )
                break

    # Write label files
    for i, path in enumerate(audio_paths):
        if not file_labels[i]:
            continue
        base = os.path.splitext(os.path.basename(path))[0]
        label_path = os.path.join(output_dir, f"{base}_labels.txt")
        with open(label_path, "w", encoding="utf-8") as f:
            for start, end, label in sorted(file_labels[i]):
                f.write(f"{start:.6f}\t{end:.6f}\t{label}\n")
        print(f"Audacity labels for {base}: {label_path}")
        print(f"  {len(file_labels[i])} labeled regions")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python audio_edit_tool.py transcribe <audio> <output.json> [model_size]")
        print("  python audio_edit_tool.py align <transcript.json> <script.md> <edl.json> [start_line]")
        print("  python audio_edit_tool.py human_edl <edl.json> <output.txt>")
        print("  python audio_edit_tool.py assemble <edl.json> <audio.wav> <output.wav> [crossfade_ms]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "transcribe":
        audio = sys.argv[2]
        output = sys.argv[3]
        model = sys.argv[4] if len(sys.argv) > 4 else "medium"
        transcribe(audio, output, model)

    elif cmd == "align":
        transcript = sys.argv[2]
        script = sys.argv[3]
        output = sys.argv[4]
        start = sys.argv[5] if len(sys.argv) > 5 else None
        script_lines = parse_script(script, start)
        print(f"Parsed {len(script_lines)} script lines")
        align_to_script(transcript, script_lines, output)

    elif cmd == "human_edl":
        edl = sys.argv[2]
        output = sys.argv[3]
        generate_human_edl(edl, output)

    elif cmd == "assemble":
        edl = sys.argv[2]
        audio = sys.argv[3]
        output = sys.argv[4]
        crossfade = int(sys.argv[5]) if len(sys.argv) > 5 else 50
        assemble_audio(edl, audio, output, crossfade)
