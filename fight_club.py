"""Render the complete fight_club animation to submission/fight_club.mp4."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time

from imageio_ffmpeg import get_ffmpeg_exe

from fight_club_scene import build_scene
from renderer.render import PROJECT_DIR, SceneRenderer, frame_pixels
from renderer.animation import load_shot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--scale", type=int, default=4,
                        help="Ray samples per pixel along each axis (default: %(default)s)")
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    if min(args.width, args.height, args.scale, args.fps) < 1:
        parser.error("dimensions, scale, and frame rate must be positive")
    shot = load_shot(PROJECT_DIR / "shot.json")
    ffmpeg = get_ffmpeg_exe()
    output = PROJECT_DIR / "submission" / "fight_club.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    scene_path, counts = build_scene()
    renderer = SceneRenderer(scene_path, args.width * args.scale, args.height * args.scale, shot=shot)
    frame_count = math.ceil(shot["duration"] * args.fps)
    # Preserve odd output dimensions, which libx264 cannot encode in 4:2:0.
    pixel_format = "yuv444p" if args.width % 2 or args.height % 2 else "yuv420p"
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        pending_output = Path(temporary) / output.name
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{args.width}x{args.height}",
            "-framerate", str(args.fps), "-i", "pipe:0", "-an",
            "-vf", "scale=in_range=full:out_range=tv:out_color_matrix=bt709",
            "-c:v", "libx264", "-preset", "slow", "-crf", "15", "-pix_fmt", pixel_format,
            "-colorspace", "bt709", "-color_range", "tv", "-movflags", "+faststart", str(pending_output),
        ]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            try:
                with encoder.stdin as frame_stream:
                    for index in range(frame_count):
                        frame = renderer.render_frame(index / args.fps)
                        pixels = frame_pixels(frame, args.width, args.height)
                        frame_stream.write(pixels[:, :, :3].tobytes())
                        if (index + 1) % args.fps == 0 or index == frame_count - 1:
                            print(f"Rendered {index + 1}/{frame_count} frames", flush=True)
                exit_code = encoder.wait()
            finally:
                if encoder.returncode is None:
                    encoder.kill()
                    encoder.wait()
        if exit_code:
            raise RuntimeError(f"FFmpeg could not encode the MP4 (exit code {exit_code}).")
        pending_output.replace(output)
    print(json.dumps({"output": str(output), "mode": "raytrace", "codec": "h264",
                      "frames": frame_count, "duration": frame_count / args.fps, "fps": args.fps,
                      "resolution": [args.width, args.height], "supersampling": args.scale, **counts,
                      "elapsed_seconds": round(time.perf_counter() - started, 2)}, indent=2))


if __name__ == "__main__":
    main()
