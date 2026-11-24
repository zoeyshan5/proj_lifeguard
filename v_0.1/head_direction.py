#!/usr/bin/env python3
"""Read a recorded video, find face landmarks, and label head direction.

MediaPipe Face Landmarker returns 478 landmarks per face. Yaw (left/right
turn) comes from the face's 4x4 pose matrix. Directions are from the
camera's point of view:

  left    nose points toward the left side of the frame
  center  head faces the camera, within the yaw threshold
  right   nose points toward the right side of the frame

On this low-angle clip the full-frame landmarker drops the face during
sharper turns. When that happens, a MediaPipe face detector crops the
face and the landmarker runs again on that crop.

Example:
  python head_direction.py Input_data/jw_ex1.MOV
"""

from __future__ import annotations

import argparse
import ssl
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = ROOT / "Input_data" / "left.MOV"
LANDMARKER_PATH = ROOT / "models" / "face_landmarker.task"
DETECTOR_PATH = ROOT / "models" / "blaze_face_full_range.tflite"
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_full_range/float16/latest/blaze_face_full_range.tflite"
)

# Horizontal turn, in degrees, that still counts as facing the camera.
CENTER_YAW_DEGREES = 10.0
LEFT_CAMERA = True

def get_yaw_offset(camera_angle):
    if (camera_angle):
        return -25
    else:
        return 30
    
YAW_OFFSET_DEGREES = get_yaw_offset(LEFT_CAMERA)   # to account for angle of camera
CROP_PADDING = 0.45

DIRECTION_BGR = {
    "left": (255, 120, 40),
    "center": (40, 180, 60),
    "right": (40, 140, 255),
    "no face": (160, 160, 160),
}



def ensure_model(path: Path, url: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {path.name} ...")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    cafile = "/etc/ssl/cert.pem"
    context = ssl.create_default_context(cafile=cafile) if Path(cafile).exists() else ssl.create_default_context()
    with urllib.request.urlopen(request, context=context) as response, path.open("wb") as handle:
        handle.write(response.read())


def yaw_degrees_from_matrix(matrix: np.ndarray) -> float:
    """Yaw of a MediaPipe facial transformation matrix, in degrees.

    Negative yaw means the nose points toward the left side of the frame.
    That sign was checked on jw_ex1.MOV: frames where the nose sits left of
    the eye midpoint produce a negative angle, and the angle grows as the
    turn gets stronger.
    """
    rotation = np.array(matrix, dtype=np.float64)[:3, :3].copy()
    column_norms = np.linalg.norm(rotation, axis=0)
    column_norms[column_norms < 1e-8] = 1.0
    rotation /= column_norms
    left, _, right = np.linalg.svd(rotation)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    horizontal = np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    return float(np.degrees(np.arctan2(-rotation[2, 0], horizontal)))


def classify_direction(yaw_degrees: float, center_degrees: float = CENTER_YAW_DEGREES) -> str:
    if yaw_degrees <= -center_degrees:
        return "left"
    if yaw_degrees >= center_degrees:
        return "right"
    return "center"


def _image(frame_bgr: np.ndarray) -> mp.Image:
    rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _padded_crop(frame: np.ndarray, detection) -> tuple[np.ndarray, int, int] | None:
    box = detection.bounding_box
    height, width = frame.shape[:2]
    x0 = max(0, int(box.origin_x - CROP_PADDING * box.width))
    y0 = max(0, int(box.origin_y - CROP_PADDING * box.height))
    x1 = min(width, int(box.origin_x + (1.0 + CROP_PADDING) * box.width))
    y1 = min(height, int(box.origin_y + (1.0 + CROP_PADDING) * box.height))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    return frame[y0:y1, x0:x1], x0, y0


def _to_full_frame(landmarks, origin_x: int, origin_y: int, crop_shape, frame_shape):
    crop_height, crop_width = crop_shape[:2]
    frame_height, frame_width = frame_shape[:2]
    mapped = []
    for point in landmarks:
        mapped.append(
            NormalizedLandmark(
                x=(origin_x + point.x * crop_width) / frame_width,
                y=(origin_y + point.y * crop_height) / frame_height,
                z=point.z,
                visibility=point.visibility,
                presence=point.presence,
            )
        )
    return mapped


def detect_head(frame, timestamp_ms, video_landmarker, image_landmarker, detector):
    """Return (landmarks in full-frame coordinates, yaw degrees) or (None, None)."""
    result = video_landmarker.detect_for_video(_image(frame), timestamp_ms)
    landmarks = result.face_landmarks[0] if result.face_landmarks else None
    matrix = result.facial_transformation_matrixes[0] if landmarks else None

    if landmarks is None:
        detected = detector.detect(_image(frame))
        if detected.detections:
            cropped = _padded_crop(frame, detected.detections[0])
            if cropped is not None:
                crop, origin_x, origin_y = cropped
                crop_result = image_landmarker.detect(_image(crop))
                if crop_result.face_landmarks:
                    landmarks = _to_full_frame(
                        crop_result.face_landmarks[0],
                        origin_x,
                        origin_y,
                        crop.shape,
                        frame.shape,
                    )
                    matrix = crop_result.facial_transformation_matrixes[0]

    if landmarks is None or matrix is None:
        return None, None
    return landmarks, yaw_degrees_from_matrix(matrix)


def draw_overlay(frame, landmarks, direction: str, yaw_degrees: float | None) -> None:
    if landmarks is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        drawing_utils.draw_landmarks(
            rgb,
            landmarks,
            FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=drawing_utils.DrawingSpec(
                color=(80, 220, 255), thickness=1, circle_radius=1
            ),
            connection_drawing_spec=drawing_utils.DrawingSpec(
                color=(80, 220, 255), thickness=1, circle_radius=1
            ),
        )
        frame[:] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        nose = landmarks[1]
        center = (int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0]))
        cv2.circle(frame, center, 5, DIRECTION_BGR[direction], -1)

    label = direction if yaw_degrees is None else f"{direction}   yaw {yaw_degrees:+.0f} deg"
    color = DIRECTION_BGR[direction]
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 64), (20, 20, 20), -1)
    cv2.putText(frame, label, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)


def process_video(video_path: Path, output_path: Path, center_degrees: float) -> None:
    ensure_model(LANDMARKER_PATH, LANDMARKER_URL)
    ensure_model(DETECTOR_PATH, DETECTOR_URL)

    base = dict(
        num_faces=1,
        min_face_detection_confidence=0.15,
        min_face_presence_confidence=0.15,
        min_tracking_confidence=0.15,
        output_facial_transformation_matrixes=True,
    )
    video_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            **base,
        )
    )
    image_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            **base,
        )
    )
    detector = mp.tasks.vision.FaceDetector.create_from_options(
        mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(DETECTOR_PATH)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=0.3,
        )
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    counts = {"left": 0, "center": 0, "right": 0, "no face": 0}
    recent_yaw: deque[float] = deque(maxlen=5)
    frame_index = 0
    last_timestamp = -1

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp_ms = int(round(frame_index * 1000.0 / fps))
        if timestamp_ms <= last_timestamp:
            timestamp_ms = last_timestamp + 1
        last_timestamp = timestamp_ms

        landmarks, yaw = detect_head(
            frame, timestamp_ms, video_landmarker, image_landmarker, detector
        )
        if yaw is None:
            recent_yaw.clear()
            direction = "no face"
            shown_yaw = None
        else:
            corrected_yaw = yaw - YAW_OFFSET_DEGREES
            recent_yaw.append(corrected_yaw)
            shown_yaw = sum(recent_yaw) / len(recent_yaw)
            direction = classify_direction(shown_yaw, center_degrees)
        counts[direction] += 1
        draw_overlay(frame, landmarks, direction, shown_yaw)
        writer.write(frame)
        frame_index += 1

    capture.release()
    writer.release()
    video_landmarker.close()
    image_landmarker.close()
    detector.close()

    total = sum(counts.values())
    print(f"Frames: {total}")
    for name in ("left", "center", "right", "no face"):
        print(f"  {name:8} {counts[name]:4d}")
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "video",
        nargs="?",
        default=str(DEFAULT_VIDEO),
        help=f"recorded video (default: {DEFAULT_VIDEO.name})",
    )
    parser.add_argument("-o", "--output", help="annotated video path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=CENTER_YAW_DEGREES,
        help="yaw within +/- this many degrees is classified as center (default: 15)",
    )
    args = parser.parse_args()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = ROOT / "Output_data" / f"{video_path.stem}_head_direction.mp4"
    process_video(video_path, output_path, args.threshold)


if __name__ == "__main__":
    main()
