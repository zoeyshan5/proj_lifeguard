# Head Direction Video Analyzer V0.1

`head_direction.py` processes a recorded video, detects facial landmarks, estimates head yaw, and labels each frame as:

- `left`
- `center`
- `right`
- `no face`

The annotated result is saved as an MP4 video. Directions are described from the camera's point of view.

## How it works

The script uses MediaPipe Face Landmarker to obtain 478 facial landmarks and a facial transformation matrix. It derives yaw from that matrix and classifies the head direction using a configurable center threshold.

If full-frame landmark detection fails during a sharper head turn, MediaPipe Face Detector locates the face. The script crops that region and retries landmark detection on the crop.

To reduce frame-to-frame jitter, the displayed yaw is averaged over the five most recent detections.

## Requirements

- Python 3.10 or newer


Python packages:

- `mediapipe`
- `numpy`
- `opencv-python`

## Usage

Process the default video, `Input_data/left.MOV`:

```bash
python head_direction.py
```

Process a specific video:

```bash
python head_direction.py Input_data/ex1.MOV
```

Choose the output path:

```bash
python head_direction.py Input_data/ex1.MOV \
  --output Output_data/ex1_annotated.mp4
```

Change the center threshold:

```bash
python head_direction.py Input_data/ex1.MOV --threshold 12
```

The threshold is measured in degrees. With `--threshold 12`, yaw between `-12°` and `+12°` is classified as `center`.

View all command-line options:

```bash
python head_direction.py --help
```

## Output

Unless `--output` is supplied, the result is written to:

```text
Output_data/<input-name>_head_direction.mp4
```

Each output frame contains:

- facial contour landmarks
- a marker on the nose
- the classified direction
- the smoothed yaw angle

After processing, the terminal displays the total number of frames, counts for each direction, and the output path.

## Camera-angle calibration

The script compensates for a camera that is not directly in front of the subject:

```python
LEFT_CAMERA = True
YAW_OFFSET_DEGREES = get_yaw_offset(LEFT_CAMERA)
```

The current offsets are:

- left-side camera: `-25°`
- other camera position: `+30°`

Adjust these values using a clip in which the subject looks straight ahead. The corrected yaw should be close to `0°` while the subject faces forward.

The default center range is configured by:

```python
CENTER_YAW_DEGREES = 10.0
```

## Models

On the first run, the script downloads:

- `models/face_landmarker.task`
- `models/blaze_face_full_range.tflite`

Later runs reuse the downloaded files.

