# Running Zone Watch in Docker

The image runs inference on CPU. The model, videos, logs and outputs are not in the image: Compose mounts them from these project folders.

| Folder | Purpose |
|--------|---------|
| `models/` | `best.pt` (required) |
| `videos/` | input videos |
| `outputs/` | annotated videos |
| `logs/` | `events.csv` |

## Build

```bash
docker compose build
```

## Video file (Windows, macOS and Linux)

A container has no window here, so give the zone as `x1,y1,x2,y2` and save the result to a file:

```bash
docker compose run --rm zone-watch --source videos/input.mp4 --roi 100,100,800,600 --no-display --save-video outputs/annotated.mp4
```

Then open `outputs/annotated.mp4` on your computer. To find coordinates for the zone, run the app outside Docker or read them from any image viewer.

## Webcam (Linux only)

Linux can share a camera device directly with a container:

```bash
docker compose run --rm --device /dev/video0 zone-watch --source 0 --roi 100,100,500,400 --no-display --save-video outputs/webcam.mp4
```

Stop it with `Ctrl+C`.

To also get the window and draw the zone with the mouse, share the X11 display:

```bash
xhost +local:docker
docker compose run --rm --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix zone-watch --source 0
```

## Webcam on Windows and macOS

Docker runs inside a Linux virtual machine on these systems, which cannot easily reach the host's camera. Run the app directly on the computer instead, as described in the README, and use Docker for video files.

## Testing status

The video-file command has been tested with Docker Desktop on Windows 11. The Linux webcam commands follow standard Docker usage but have not been tested.
