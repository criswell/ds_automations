#!/usr/bin/env python3
"""
caption_merge.py

Merges a pre-written voiceover script with unscripted dialog transcribed from
a video, producing an SRT file suitable for YouTube closed captions.

The script is expected to be the voiceover text (chapter intros, etc.).
Everything spoken in the video that isn't in the script is transcribed by
Whisper and interleaved with the script text at the correct timestamps.

Usage:
    python caption_merge.py VIDEO SCRIPT [options]

Dependencies:
    pip install faster-whisper rapidfuzz
"""

import sys
import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Section:
    name: str   # markdown header text, used for debug output
    text: str   # clean prose, stage directions removed


@dataclass
class Caption:
    start: float
    end: float
    text: str
    source: str  # 'script' | 'whisper'


# ─── Script parsing ───────────────────────────────────────────────────────────

_STAGE_DIRECTION = re.compile(r'\[.*?\]')
_WHITESPACE = re.compile(r'\s+')


def parse_script(path: Path) -> List[Section]:
    """Parse markdown into ordered sections, stripping headers and stage cues."""
    text = path.read_text(encoding='utf-8')
    chunks = re.split(r'^(#{1,6}\s+.*)$', text, flags=re.MULTILINE)

    sections: List[Section] = []
    current_name = 'preamble'

    for chunk in chunks:
        if re.match(r'^#{1,6}\s+', chunk):
            current_name = re.sub(r'^#{1,6}\s+', '', chunk).strip()
        else:
            clean = _WHITESPACE.sub(' ', _STAGE_DIRECTION.sub('', chunk)).strip()
            if clean:
                sections.append(Section(name=current_name, text=clean))

    return sections


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation for fuzzy comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", ' ', text)
    return _WHITESPACE.sub(' ', text).strip()


# ─── Transcription ────────────────────────────────────────────────────────────

def transcribe(video_path: Path, model_size: str, device: str, initial_prompt: Optional[str]) -> List[Word]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("faster-whisper not installed. Run: pip install faster-whisper")

    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Loading Whisper '{model_size}' on {device}...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"Transcribing {video_path.name}...")
    segments, info = model.transcribe(
        str(video_path),
        word_timestamps=True,
        language="en",
        vad_filter=True,
        initial_prompt=initial_prompt or None,
    )

    words: List[Word] = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                t = w.word.strip()
                if t:
                    words.append(Word(text=t, start=w.start, end=w.end))

    print(f"Transcribed {len(words)} words, {info.duration:.0f}s total.")
    return words


# ─── Alignment ────────────────────────────────────────────────────────────────

def align_sections(
    words: List[Word],
    sections: List[Section],
    threshold: int,
) -> List[Tuple[int, int, Section]]:
    """
    For each script section (in order), find the best-matching contiguous window
    of Whisper words via sliding-window fuzzy match. Returns (start_idx, end_idx,
    section) tuples. Searches left-to-right so sections can't overlap.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        sys.exit("rapidfuzz not installed. Run: pip install rapidfuzz")

    norm_words = [_normalize(w.text) for w in words]
    search_from = 0
    matches: List[Tuple[int, int, Section]] = []

    for sec in sections:
        norm_sec = _normalize(sec.text)
        sec_tokens = norm_sec.split()
        n = len(sec_tokens)

        if n == 0:
            continue

        best_score, best_start, best_end = -1, -1, -1

        # Try several window sizes to accommodate Whisper insertions/omissions
        for factor in [1.0, 0.85, 1.15, 0.70, 1.30]:
            win = max(int(n * factor), 3)
            limit = len(words) - win + 1

            for i in range(search_from, limit):
                window_str = ' '.join(norm_words[i:i + win])
                score = fuzz.ratio(window_str, norm_sec)
                if score > best_score:
                    best_score = score
                    best_start = i
                    best_end = i + win

        if best_score >= threshold:
            matches.append((best_start, best_end, sec))
            search_from = best_end
            print(f"  [score={best_score:3.0f}] '{sec.name}' → words[{best_start}:{best_end}]")
        else:
            print(f"  [WARN ] '{sec.name}' unmatched (best={best_score:.0f}); treating as unscripted.")

    return matches


# ─── Caption construction ─────────────────────────────────────────────────────

def _split_phrases(text: str, max_words: int) -> List[str]:
    """Split text into caption-sized phrases, preferring sentence/clause breaks."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    out: List[str] = []

    for sent in sentences:
        tokens = sent.split()
        buf: List[str] = []

        for tok in tokens:
            buf.append(tok)
            at_clause = tok.rstrip("\"'").endswith((',', ';', ':')) and len(buf) >= 4
            at_max = len(buf) >= max_words
            if at_clause or at_max:
                out.append(' '.join(buf))
                buf = []

        if buf:
            out.append(' '.join(buf))

    return [p for p in out if p]


def build_captions(
    words: List[Word],
    matches: List[Tuple[int, int, Section]],
    max_words: int,
    max_dur: float,
) -> List[Caption]:
    scripted_indices = {idx for s, e, _ in matches for idx in range(s, e)}
    queue = sorted(matches, key=lambda x: x[0])
    captions: List[Caption] = []
    i = 0

    while i < len(words):
        # ── Scripted section ──────────────────────────────────────────────────
        if queue and i == queue[0][0]:
            start_i, end_i, sec = queue.pop(0)
            end_i = min(end_i, len(words))
            t0 = words[start_i].start
            t1 = words[end_i - 1].end
            dur = t1 - t0

            phrases = _split_phrases(sec.text, max_words)
            if phrases:
                total_words = sum(len(p.split()) for p in phrases)
                t = t0
                for phrase in phrases:
                    frac = len(phrase.split()) / total_words
                    captions.append(Caption(
                        start=t, end=t + dur * frac,
                        text=phrase, source='script',
                    ))
                    t += dur * frac

            i = end_i
            continue

        # Skip words already consumed by a scripted range (shouldn't happen, but safe)
        if i in scripted_indices:
            i += 1
            continue

        # ── Unscripted gap ────────────────────────────────────────────────────
        next_start = queue[0][0] if queue else len(words)
        buf: List[str] = []
        seg_start = words[i].start

        while i < next_start and i < len(words):
            w = words[i]

            # Flush if duration or word count limit reached
            if buf and (w.start - seg_start >= max_dur or len(buf) >= max_words):
                captions.append(Caption(
                    start=seg_start, end=words[i - 1].end,
                    text=' '.join(buf), source='whisper',
                ))
                buf = []
                seg_start = w.start

            buf.append(w.text)
            i += 1

            # Also flush at sentence-ending punctuation
            if len(buf) >= 3 and w.text.rstrip("\"'").endswith(('.', '!', '?')):
                captions.append(Caption(
                    start=seg_start, end=w.end,
                    text=' '.join(buf), source='whisper',
                ))
                buf = []
                seg_start = words[i].start if i < len(words) else w.end

        if buf:
            end_t = words[i - 1].end if i > 0 else seg_start + 1.0
            captions.append(Caption(
                start=seg_start, end=end_t,
                text=' '.join(buf), source='whisper',
            ))

    captions.sort(key=lambda c: c.start)
    return [c for c in captions if c.text.strip()]


# ─── SRT output ───────────────────────────────────────────────────────────────

def _fmt(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    ms = int((s % 1) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def write_srt(captions: List[Caption], path: Path) -> None:
    lines: List[str] = []
    for n, c in enumerate(captions, 1):
        lines += [str(n), f"{_fmt(c.start)} --> {_fmt(c.end)}", c.text, '']
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Wrote {len(captions)} captions → {path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("video", type=Path, help="Input video file")
    ap.add_argument("script", type=Path, help="Markdown voiceover script")
    ap.add_argument("-o", "--output", type=Path,
                    help="Output SRT file (default: <video>.srt)")
    ap.add_argument("--model", default="large-v3",
                    metavar="SIZE",
                    help="Whisper model: tiny/base/small/medium/large-v2/large-v3 (default: large-v3)")
    ap.add_argument("--initial-prompt", default="", metavar="TEXT",
                    help="Prompt to condition Whisper's initial state — useful for noisy audio. "
                         "Example: 'The following is spoken commentary with background music.'")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="Compute device (default: cuda)")
    ap.add_argument("--threshold", type=int, default=60, metavar="0-100",
                    help="Fuzzy match threshold for script alignment (default: 60); "
                         "lower if sections are being missed")
    ap.add_argument("--max-words", type=int, default=8,
                    help="Max words per caption line (default: 8)")
    ap.add_argument("--max-duration", type=float, default=7.0,
                    help="Max seconds per unscripted caption (default: 7.0)")
    ap.add_argument("--debug", action="store_true",
                    help="Write word list and alignment results to debug.json")
    args = ap.parse_args()

    for p, label in [(args.video, "video"), (args.script, "script")]:
        if not p.exists():
            sys.exit(f"File not found: {p} ({label})")

    output = args.output or args.video.with_suffix('.srt')

    # Parse script
    print("Parsing script...")
    sections = parse_script(args.script)
    if not sections:
        sys.exit("No text found in script file.")
    print(f"Found {len(sections)} section(s):")
    for s in sections:
        preview = s.text[:70].replace('\n', ' ')
        print(f"  [{s.name}] {preview}{'...' if len(s.text) > 70 else ''}")

    # Transcribe
    words = transcribe(args.video, args.model, args.device, args.initial_prompt)
    if not words:
        sys.exit("No words transcribed from video.")

    if args.debug:
        debug_path = Path("debug.json")
        debug_data: dict = {
            "words": [{"text": w.text, "start": w.start, "end": w.end} for w in words],
            "matches": [],
        }

    # Align
    print("\nAligning script sections to transcript...")
    matches = align_sections(words, sections, args.threshold)
    print(f"Aligned {len(matches)}/{len(sections)} section(s).\n")

    if args.debug:
        debug_data["matches"] = [
            {"section": sec.name, "start_word": a, "end_word": b,
             "start_time": words[a].start, "end_time": words[min(b, len(words)) - 1].end}
            for a, b, sec in matches
        ]
        debug_path.write_text(json.dumps(debug_data, indent=2), encoding='utf-8')
        print(f"Debug info written to {debug_path}")

    # Build and write captions
    captions = build_captions(words, matches, args.max_words, args.max_duration)
    write_srt(captions, output)

    scripted_n = sum(1 for c in captions if c.source == 'script')
    whisper_n = sum(1 for c in captions if c.source == 'whisper')
    print(f"Done: {scripted_n} scripted + {whisper_n} unscripted = {len(captions)} total captions.")


if __name__ == "__main__":
    main()
