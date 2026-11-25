# fight_club

A standalone ray-traced animation inspired by the final window shot in *Fight Club*. The 15-second shot combines an office interior, a moving camera, three collapsing towers, burning window floors, and volumetric smoke. It provides an environment plate for live-action compositing.

## Run

Requires Python 3.13, uv, and a GPU supported by SlangPy. Run from the project directory.

```powershell
uv sync --locked
uv run --locked python fight_club.py
```

The command streams rendered frames to FFmpeg and writes `submission/fight_club.mp4`. The default is a 15-second H.264 video at 960 × 400 and 12 fps (180 frames). The locked dependencies provide FFmpeg on supported platforms. Re-running the command replaces the video after encoding succeeds.

The default `--scale 4` traces a 4 × 4 grid of primary rays per output pixel. At 960 × 400, the renderer traces each frame at 3840 × 1600 and downsamples it with a Lanczos filter in linear light before bloom and tone mapping. Use `--scale 1` for a faster single-sample render. The other output options are `--width`, `--height`, and `--fps`. Paths resolve relative to the project directory.


## Files

```text
fight_club/
  fight_club.py             Complete animation export
  fight_club_scene.py       Scene generation
  shot.json                 Camera, collapse, and effects
  renderer/                 GPU rendering, animation, and BVHs
  shaders/                  Ray tracing, lighting, and volumes
  scene/                    Generated geometry
  submission/               Source ZIP and MP4 for submission
  tests/                    Animation and volume checks
  assignment/               Local copies of course references
  pyproject.toml            Dependency configuration
  uv.lock                   Locked dependencies
  THIRD_PARTY_NOTICES.md    Source credits and license
```