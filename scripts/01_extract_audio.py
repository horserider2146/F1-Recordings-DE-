"""
01_extract_audio.py — Extract 16 kHz mono WAV audio from video files or YouTube URLs.

Usage:
    # From a folder of local video files:
    python scripts/01_extract_audio.py --input data/raw_video/

    # From a single video file:
    python scripts/01_extract_audio.py --input "my_video.mp4"

    # From a text file of YouTube URLs (one URL per line):
    python scripts/01_extract_audio.py --urls urls.txt

    # Download a single YouTube URL:
    python scripts/01_extract_audio.py --urls "https://www.youtube.com/watch?v=..."
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import soundfile as sf

# Make sure project root is on the path so config is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_AUDIO_DIR, SAMPLE_RATE, CHANNELS


def _ffmpeg_path() -> str:
    """Return the ffmpeg binary path, preferring the imageio-managed one."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"  # fall back to system ffmpeg


def download_youtube(url: str, out_dir: Path) -> Path:
    """Download a YouTube video using yt-dlp and return the saved file path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Download best audio quality, prefer webm/mp4
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--output", str(out_dir / "%(title)s.%(ext)s"),
        "--format", "bestaudio/best",
        url,
    ]
    print(f"  Downloading: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] yt-dlp failed:\n{result.stderr}")
        return None

    # Find the newest file in out_dir
    files = sorted(out_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print("  [ERROR] No file found after download.")
        return None
    return files[0]


def extract_audio_ffmpeg(input_path: Path, output_path: Path) -> bool:
    """Use ffmpeg to extract audio as 16 kHz mono WAV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg_path()
    cmd = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-vn",                   # drop video stream
        "-acodec", "pcm_s16le",  # 16-bit PCM
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] ffmpeg failed on {input_path.name}:\n{result.stderr[-500:]}")
        return False
    return True


def process_video_file(video_path: Path) -> bool:
    """Extract audio from a single video file into RAW_AUDIO_DIR."""
    stem = video_path.stem
    # Sanitise filename — replace spaces and special characters
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    output_wav = RAW_AUDIO_DIR / f"{safe_stem}.wav"

    if output_wav.exists():
        print(f"  [SKIP] Already extracted: {output_wav.name}")
        return True

    print(f"  Extracting: {video_path.name} → {output_wav.name}")
    success = extract_audio_ffmpeg(video_path, output_wav)
    if success:
        # Quick validation: ensure file is readable and has content
        try:
            info = sf.info(output_wav)
            duration = info.frames / info.samplerate
            print(f"    OK — {duration:.1f}s, {info.samplerate} Hz, {info.channels}ch")
        except Exception as e:
            print(f"    [WARN] Could not validate output: {e}")
    return success


def main():
    parser = argparse.ArgumentParser(description="Extract audio from video files or YouTube URLs.")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to a video file or folder of video files")
    parser.add_argument("--urls", type=str, default=None,
                        help="A YouTube URL or path to a text file containing one URL per line")
    args = parser.parse_args()

    if not args.input and not args.urls:
        parser.print_help()
        sys.exit(1)

    RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    video_download_dir = RAW_AUDIO_DIR.parent / "raw_video"

    # ── Handle YouTube URLs ────────────────────────────────────────────────────
    if args.urls:
        urls = []
        if args.urls.startswith("http"):
            urls = [args.urls.strip()]
        else:
            url_file = Path(args.urls)
            if not url_file.exists():
                print(f"[ERROR] URL file not found: {url_file}")
                sys.exit(1)
            urls = [line.strip() for line in url_file.read_text().splitlines()
                    if line.strip() and not line.startswith("#")]

        print(f"\nDownloading {len(urls)} YouTube URL(s)...")
        for url in urls:
            video_path = download_youtube(url, video_download_dir)
            if video_path:
                process_video_file(video_path)

    # ── Handle local video files ───────────────────────────────────────────────
    if args.input:
        input_path = Path(args.input)
        VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v", ".mp3", ".m4a", ".opus", ".ogg"}

        if input_path.is_file():
            video_files = [input_path]
        elif input_path.is_dir():
            video_files = [p for p in sorted(input_path.iterdir())
                           if p.suffix.lower() in VIDEO_EXTS]
            if not video_files:
                print(f"[ERROR] No video files found in: {input_path}")
                sys.exit(1)
        else:
            print(f"[ERROR] Path does not exist: {input_path}")
            sys.exit(1)

        print(f"\nProcessing {len(video_files)} video file(s)...")
        ok = sum(1 for vf in video_files if process_video_file(vf))
        print(f"\nDone — {ok}/{len(video_files)} files extracted successfully.")
        print(f"Output directory: {RAW_AUDIO_DIR}")


if __name__ == "__main__":
    main()
