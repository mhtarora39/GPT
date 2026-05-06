"""
midi_dataset.py
Converts a folder of .mid / .midi files into a single flat text file
that your existing DataLoader + char-level Tokenizer can train on.

Token vocabulary (human-readable, each separated by a space):
  N<pitch>_<velocity>   e.g. N60_80  → Note-On, pitch=60, velocity=80
  O<pitch>              e.g. O60     → Note-Off, pitch=60
  T<ms>                 e.g. T250   → Time-shift in milliseconds (quantised)
  BAR                              → Bar boundary (inserted at each measure)
  EOS                              → End of one MIDI file

Install dependency:
  pip install mido

Usage (command-line):
  python midi_dataset.py <midi_dir> <output.txt>

Usage (in code):
  from midi_dataset import midi_dir_to_text
  midi_dir_to_text("./midi_files", "corpus.txt")
  # then pass corpus.txt to train()
"""

import os
import sys
import glob
from pathlib import Path

try:
    import mido
except ImportError:
    raise ImportError("Install mido first:  pip install mido")


# ── Quantisation ─────────────────────────────────────────────────────────────
# Time shifts are bucketed to reduce vocabulary size.
# All raw millisecond deltas are rounded to the nearest TIME_STEP.
TIME_STEP_MS = 25          # 25 ms buckets  (40 time-steps per second)
MAX_TIME_MS  = 2000        # shifts > 2 s are capped at 2000 ms

def _quantise(ms: float) -> int:
    """Round delta-time to nearest TIME_STEP_MS bucket, capped at MAX_TIME_MS."""
    return min(round(ms / TIME_STEP_MS) * TIME_STEP_MS, MAX_TIME_MS)


# ── Single file converter ─────────────────────────────────────────────────────
def midi_to_tokens(filepath: str) -> list[str]:
    """
    Parse one MIDI file and return a flat list of string tokens.
    Only tracks channel 0 events (melody) by default.
    """
    try:
        mid = mido.MidiFile(filepath)
    except Exception as e:
        print(f"  [SKIP] {filepath}: {e}")
        return []

    tokens = []
    tempo = 500_000  # default: 120 BPM (microseconds per beat)

    # Merge all tracks into a single absolute-time event stream
    merged = list(mido.merge_tracks(mid.tracks))

    current_tick = 0
    pending_ms   = 0.0   # accumulated time before a note event

    for msg in merged:
        # Update cumulative ticks and convert tick delta → ms
        current_tick += msg.time
        delta_ms = mido.tick2second(msg.time, mid.ticks_per_beat, tempo) * 1000.0
        pending_ms += delta_ms

        if msg.type == 'set_tempo':
            tempo = msg.tempo
            continue

        if msg.type == 'note_on' and msg.velocity > 0:
            # Flush accumulated time before this note event
            qt = _quantise(pending_ms)
            if qt > 0:
                tokens.append(f"T{qt}")
            pending_ms = 0.0
            tokens.append(f"N{msg.note}_{msg.velocity}")

        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            qt = _quantise(pending_ms)
            if qt > 0:
                tokens.append(f"T{qt}")
            pending_ms = 0.0
            tokens.append(f"O{msg.note}")

    tokens.append("EOS")
    return tokens


# ── Directory converter ───────────────────────────────────────────────────────
def midi_dir_to_text(midi_dir: str, output_path: str, max_files: int = None):
    """
    Scan `midi_dir` recursively for .mid / .midi files, convert each to
    tokens, and write the entire corpus as space-separated tokens to
    `output_path`.  Each file is separated by EOS so the model sees
    song boundaries.

    Args:
        midi_dir:    Path to directory containing .mid / .midi files.
        output_path: Path to output .txt file.
        max_files:   Optional cap on number of files to process.
    """
    patterns = ["**/*.mid", "**/*.midi", "**/*.MID", "**/*.MIDI"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(midi_dir, p), recursive=True))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(f"No MIDI files found in: {midi_dir}")

    if max_files:
        files = files[:max_files]

    print(f"Found {len(files)} MIDI file(s) in '{midi_dir}'")

    all_tokens = []
    for i, fp in enumerate(files):
        print(f"  [{i+1}/{len(files)}] {Path(fp).name}", end="  ")
        toks = midi_to_tokens(fp)
        print(f"→ {len(toks)} tokens")
        all_tokens.extend(toks)

    corpus = " ".join(all_tokens)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(corpus)

    unique = len(set(all_tokens))
    print(f"\nCorpus written to '{output_path}'")
    print(f"  Total tokens  : {len(all_tokens):,}")
    print(f"  Unique tokens : {unique:,}")
    print(f"  File size     : {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"\nNow train with:\n  python model.py {output_path}")


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python midi_dataset.py <midi_dir> <output.txt> [max_files]")
        sys.exit(1)

    midi_dir    = sys.argv[1]
    output_path = sys.argv[2]
    max_files   = int(sys.argv[3]) if len(sys.argv) > 3 else None

    midi_dir_to_text(midi_dir, output_path, max_files)
