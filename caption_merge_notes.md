# caption_merge.py — Usage Notes

Merges a pre-written voiceover script with unscripted dialog from a video into a YouTube-ready SRT caption file.

## Basic Usage

```bash
python caption_merge.py VIDEO SCRIPT [options]
```

**Example:**
```bash
python caption_merge.py my_video.mp4 voiceover_script.md
```

Output defaults to `<video_name>.srt` in the same directory as the video.

## Options

| Flag | Default | Description |
|---|---|---|
| `-o`, `--output` | `<video>.srt` | Output SRT file path |
| `--model` | `large-v3` | Whisper model size (see below) |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--initial-prompt` | _(none)_ | Text to condition Whisper's initial state (see below) |
| `--threshold` | `60` | Fuzzy match threshold 0–100 (see below) |
| `--max-words` | `8` | Max words per caption line |
| `--max-duration` | `7.0` | Max seconds per unscripted caption chunk |
| `--debug` | off | Write word list + alignment results to `debug.json` |

## Script Format

The script file is expected to be Markdown. The tool handles:

- **Headers** (`#`, `##`, etc.) — stripped from caption text, used as section names in debug output
- **Stage directions** like `[pause]`, `[cut]` — stripped automatically
- **Plain prose** — used verbatim as caption text for scripted sections

The script sections are assumed to be **non-overlapping chapter intros** — they should not be interleaved with each other mid-sentence. Unscripted dialog between sections is handled automatically.

## How It Works

1. Parses the script into sections (one per markdown header block)
2. Transcribes the video with `faster-whisper` using word-level timestamps
3. Aligns each script section to the transcript via **left-to-right sliding-window fuzzy matching** (`rapidfuzz`)
4. For **scripted ranges**: uses the script's own text, split into phrase-sized chunks with timestamps drawn proportionally from the matched Whisper word span
5. For **unscripted gaps**: uses Whisper's transcript directly, grouped into phrase-level caption chunks
6. Outputs SRT

## Whisper Model Sizes

From fastest/smallest to slowest/most accurate:

| Model | Notes |
|---|---|
| `tiny`, `base` | Fast, lower accuracy — not recommended for caption use |
| `small`, `medium` | Reasonable accuracy, much faster than large |
| `large-v2` | Stable; slightly lower noise robustness than v3 |
| `large-v3` | **Default.** Better accuracy on noisy/music-backed audio; trained on more diverse data |

## Handling Background Noise and Music

Whisper large-v3 (the default) handles noisy and music-backed audio well, but you can improve accuracy further with `--initial-prompt`. This text conditions Whisper's initial hidden state — it doesn't appear in the output, it just tells the model what kind of audio to expect.

For a video with background music:
```bash
python caption_merge.py my_video.mp4 script.md \
  --initial-prompt "The following is a spoken video commentary with background music."
```

If Whisper is picking up song lyrics or gibberish from the music instead of speech, a more directive prompt helps:
```bash
--initial-prompt "Spoken narration only. Ignore background music and sound effects."
```

`--initial-prompt` works best when combined with `vad_filter=True` (always on), which strips silence and low-energy frames before transcription.

## Tuning the Threshold

`--threshold` controls how similar a window of transcript words must be to a script section to count as a match (0–100, higher = stricter).

- Default `60` works well for clear audio and closely-read scripts
- **Lower it (e.g. `45`)** if sections are being reported as unmatched — the speaker may be paraphrasing or Whisper is mishearing words
- **Raise it (e.g. `75`)** if you're getting false matches (wrong part of the video matched to a script section)
- Always run with `--debug` first on a new video to inspect alignment quality before trusting the output

## Diagnosing Problems with `--debug`

`--debug` writes `debug.json` to the current directory containing:

- `words`: every Whisper word with its start/end timestamp
- `matches`: each aligned script section with its word index range and timestamps

Use this to check:
- Whether each section matched the right part of the video
- Whether the matched word range is too short or too long (suggests threshold needs tuning)
- Whether Whisper is producing garbled output for specific segments (audio quality issue)

## Items of Note

**Short script sections are harder to match.** Sections under ~10 words have less text to match against and are more likely to produce false positives or misses. If you have very short intro lines, consider lowering `--threshold` or combining them into a single section.

**The script text is used verbatim for scripted captions.** Timestamps come from Whisper, but the words in the SRT file for scripted sections are taken directly from your script — not from what Whisper heard. This means any deviation between what was actually said and the script will still be captioned as the script text. If the speaker went significantly off-script, lower the threshold but also consider manually reviewing those sections.

**Proportional timestamp distribution for scripted sections.** Within a scripted section, phrase timestamps are distributed proportionally by word count across the total duration Whisper detected. This means pauses within the scripted delivery won't be reflected in the caption timing. The captions will still be readable, but may not land perfectly on the spoken words. If precise sync matters, use `--debug` and adjust manually.

**VAD filter is on by default.** `faster-whisper` is run with `vad_filter=True`, which strips silence before transcription. This improves accuracy but means very quiet or low-energy speech may be dropped. If you notice missing unscripted sections, try re-running with a modified version that sets `vad_filter=False`.

**GPU highly recommended.** `large-v2` on CPU is very slow (10–30 minutes for an hour of video). On a modern GPU it typically takes 2–5 minutes.

**SRT is the recommended upload format for YouTube.** YouTube also accepts VTT and SBV, but SRT is the most universally compatible and is what this tool produces.
