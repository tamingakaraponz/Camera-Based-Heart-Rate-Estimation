"""Readable, standalone version of the active iPPG v2 workflow.

Pipeline
--------
1. Track 478 face landmarks with MediaPipe in VIDEO mode.
2. Smooth the landmarks and create polygon ROIs for the forehead and cheeks.
3. Calibrate an adaptive YCrCb skin model from the first seconds of the clip.
4. Smooth the skin masks and store mean [R, G, B] for each region.
5. Combine the forehead and both cheeks using their skin-pixel counts.
6. Resample RGB to the video FPS and extract a pulse with POS, G-R, or POS+ICA.
7. Denoise POS/ICA with DWT -> soft threshold -> IDWT, then band-pass filter.
8. Find the strongest FFT frequency in the configured heart-rate range.

Background extraction is disabled by default. The legacy and normalized-power
background experiments run only when explicitly requested.

If MediaPipe or its model is unavailable, the program automatically falls back
to Haar face rectangles. The adaptive skin model is used with either tracker.

POS+ICA uses quality-weighted time and spectral fusion. Wavelet denoising is an
explicit stage; use --denoise none to bypass it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import warnings
from importlib.metadata import version as distribution_version, PackageNotFoundError

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

try:
    import pywt
except (ImportError, OSError):
    pywt = None

try:
    import mediapipe as mp
except (ImportError, OSError):
    mp = None

try:
    from sklearn.decomposition import FastICA
    from sklearn.exceptions import ConvergenceWarning
except (ImportError, OSError):
    FastICA = None
    ConvergenceWarning = Warning


@dataclass(frozen=True)
class IPPGConfig:
    """Configuration for video processing and signal analysis."""

    max_duration_seconds: float | None = 30
    face_scale_factor: float = 1.3
    face_min_neighbors: int = 9
    face_cascade_path: Path | None = None

    prefer_mediapipe: bool = True
    face_landmarker_model_path: Path | None = None
    landmark_ema_alpha: float = 0.35
    face_detection_confidence: float = 0.5
    face_presence_confidence: float = 0.5
    face_tracking_confidence: float = 0.5

    # These fixed values are used only when adaptive calibration is disabled.
    skin_cr_min: int = 133
    skin_cr_max: int = 173
    skin_cb_min: int = 77
    skin_cb_max: int = 127
    skin_hue_max: int = 25
    skin_saturation_min: int = 20
    skin_value_min: int = 40
    minimum_skin_pixels_per_region: int = 20

    roi_method: str = "legacy"
    roi_trim_fraction: float = 0.10
    roi_min_patch_pixels: int = 10
    roi_min_quality: float = 0.05
    roi_exposure_low: float = 30.0
    roi_exposure_high: float = 225.0
    roi_homogeneity_scale: float = 0.08
    roi_forehead_columns: int = 3
    roi_cheek_rows: int = 2
    roi_keep_forehead_patches: int = 2
    roi_keep_cheek_patches: int = 1

    adaptive_skin_detection: bool = True
    skin_calibration_seconds: float = 2.0
    skin_calibration_samples_per_frame: int = 2500
    skin_mad_multiplier: float = 3.5
    skin_min_chroma_half_width: float = 8.0
    skin_mask_ema_alpha: float = 0.40

    forehead_low_hz: float = 0.6
    forehead_high_hz: float = 2.0
    forehead_filter_order: int = 6

    background_low_hz: float = 0.7
    background_high_hz: float = 4.0
    background_filter_order: int = 8
    background_spectrum_scale: float = 1.5
    background_method: str | None = None
    background_power_weight: float = 1.0
    background_patch_size: int = 32
    background_calibration_frames: int = 30
    background_exclusion_margin_pixels: int = 20
    background_min_luminance: float = 20.0
    background_max_luminance: float = 235.0
    background_epsilon: float = 1e-8

    pulse_method: str = "pos"
    pos_window_seconds: float = 1.6
    pos_epsilon: float = 1e-8

    fusion_domain: str = "spectral"
    fusion_min_correlation: float = 0.2
    fusion_bpm_tolerance: float = 6.0
    fusion_peak_half_width_hz: float = 0.1
    fusion_epsilon: float = 1e-8
    ica_random_state: int = 0
    ica_max_iter: int = 1000
    ica_tolerance: float = 1e-4

    denoise: str = "dwt"
    wavelet: str = "db4"
    wavelet_level: int | str = "auto"
    wavelet_strength: float = 1.0
    # Keep only coefficients whose approximate frequency band cannot contain
    # the configured pulse range. This avoids thresholding the pulse itself.
    wavelet_strategy: str = "protect-pulse-band"

    show_preview: bool = True


@dataclass(frozen=True)
class ExtractedSignals:
    """Uniformly sampled RGB values and their extracted pulse signal."""

    time_seconds: np.ndarray
    forehead_rgb: np.ndarray
    cheeks_rgb: np.ndarray
    combined_rgb: np.ndarray
    pulse_signal: np.ndarray | None
    background: np.ndarray | None
    fps: float
    pulse_method: str
    background_patch_box: tuple[int, int, int, int] | None
    roi_quality: np.ndarray | None = None
    roi_weights: np.ndarray | None = None
    roi_method: str = "legacy"
    face_tracker_name: str | None = None


@dataclass(frozen=True)
class IPPGResult:
    """All useful outputs from one video analysis."""

    signals: ExtractedSignals
    filtered_pulse_signal: np.ndarray
    filtered_background: np.ndarray | None
    frequencies_hz: np.ndarray
    compensated_spectrum: np.ndarray
    heart_rate_bpm: float
    background_method: str
    reconstructed_pulse_signal: np.ndarray
    denoise_metadata: dict
    heart_rate_source: str
    fusion_result: FusionResult | None


@dataclass(frozen=True)
class ProcessedPulse:
    """Preserve the raw, reconstructed and filtered pulse separately."""

    raw: np.ndarray
    reconstructed: np.ndarray
    filtered: np.ndarray
    denoise_metadata: dict


@dataclass(frozen=True)
class PulseCandidateResult:
    """A processed POS or ICA candidate and its in-band quality score."""

    method: str
    processed: ProcessedPulse
    frequencies_hz: np.ndarray
    power_spectrum: np.ndarray
    heart_rate_bpm: float
    quality: float
    component_index: int | None = None


@dataclass(frozen=True)
class FusionResult:
    """All inspectable outputs from POS+ICA time and spectral fusion."""

    pos_candidate: PulseCandidateResult | None
    ica_candidates: tuple[PulseCandidateResult, ...]
    selected_ica: PulseCandidateResult | None
    time_fused_signal: np.ndarray
    time_spectrum: np.ndarray
    time_bpm: float
    spectral_score: np.ndarray
    spectral_bpm: float
    weights: dict[str, float]
    correlation: float | None
    ica_polarity: int
    status: str
    metadata: dict


@dataclass(frozen=True)
class FaceRegions:
    """Cropped ROIs, geometry masks, and their frame boxes."""

    forehead: np.ndarray
    left_cheek: np.ndarray
    right_cheek: np.ndarray
    face_box: tuple[int, int, int, int]
    forehead_box: tuple[int, int, int, int]
    left_cheek_box: tuple[int, int, int, int]
    right_cheek_box: tuple[int, int, int, int]
    forehead_geometry: np.ndarray
    left_cheek_geometry: np.ndarray
    right_cheek_geometry: np.ndarray


@dataclass(frozen=True)
class ROISelection:
    """Robust RGB and quality information for one anatomical ROI."""

    rgb: np.ndarray
    quality: float
    pixel_count: int
    selected_patches: tuple[int, ...]


# MediaPipe Face Mesh indices. The polygons deliberately avoid eyes, eyebrows,
# nose, and mouth because those areas add motion and non-skin pixels to iPPG.
FOREHEAD_LANDMARKS = np.array(
    [
        21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251,
        300, 293, 334, 296, 336, 107, 66, 105, 63, 70,
    ],
    dtype=np.int32,
)
LEFT_CHEEK_LANDMARKS = np.array(
    [50, 101, 118, 117, 123, 147, 187, 207, 206, 205],
    dtype=np.int32,
)
RIGHT_CHEEK_LANDMARKS = np.array(
    [280, 330, 347, 346, 352, 376, 411, 427, 426, 425],
    dtype=np.int32,
)


def _crop_from_geometry(
    frame: np.ndarray,
    full_frame_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Crop one polygon and its mask to the polygon's bounding rectangle."""

    nonzero_points = cv2.findNonZero(full_frame_mask)
    if nonzero_points is None:
        raise ValueError("A landmark ROI produced an empty polygon.")

    left, top, width, height = cv2.boundingRect(nonzero_points)
    right = left + width
    bottom = top + height
    return (
        frame[top:bottom, left:right],
        full_frame_mask[top:bottom, left:right],
        (left, top, right, bottom),
    )


class MediaPipeRegionTracker:
    """Track face landmarks and turn them into stable polygon ROIs."""

    name = "MediaPipe Face Landmarker (VIDEO)"

    def __init__(self, config: IPPGConfig) -> None:
        if mp is None:
            raise RuntimeError("The mediapipe package is not installed.")

        project_root = Path(__file__).resolve().parent.parent
        model_path = config.face_landmarker_model_path
        if model_path is None:
            model_path = project_root / "models" / "face_landmarker.task"
        model_path = model_path.expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Face Landmarker model not found: {model_path}")

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=config.face_detection_confidence,
            min_face_presence_confidence=config.face_presence_confidence,
            min_tracking_confidence=config.face_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        )
        self._ema_alpha = config.landmark_ema_alpha
        self._smoothed_points: np.ndarray | None = None
        self._last_timestamp_ms = -1
        print(f"Face tracker: {self.name}")
        print(f"Face Landmarker model: {model_path}")

    def close(self) -> None:
        """Release native MediaPipe resources."""

        self._landmarker.close()

    def detect(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
    ) -> FaceRegions | None:
        """Return polygon regions for one video frame."""

        # VIDEO mode requires monotonically increasing timestamps.
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        media_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )
        result = self._landmarker.detect_for_video(media_image, timestamp_ms)
        if not result.face_landmarks:
            self._smoothed_points = None
            return None

        frame_height, frame_width = frame.shape[:2]
        points = np.array(
            [
                [
                    np.clip(landmark.x * frame_width, 0, frame_width - 1),
                    np.clip(landmark.y * frame_height, 0, frame_height - 1),
                ]
                for landmark in result.face_landmarks[0]
            ],
            dtype=np.float32,
        )

        if self._smoothed_points is None:
            self._smoothed_points = points
        else:
            alpha = self._ema_alpha
            self._smoothed_points = (
                alpha * points + (1.0 - alpha) * self._smoothed_points
            )
        integer_points = np.rint(self._smoothed_points).astype(np.int32)

        full_masks = []
        for landmark_indices in (
            FOREHEAD_LANDMARKS,
            LEFT_CHEEK_LANDMARKS,
            RIGHT_CHEEK_LANDMARKS,
        ):
            mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            cv2.fillPoly(mask, [integer_points[landmark_indices]], 255)
            full_masks.append(mask)

        forehead, forehead_geometry, forehead_box = _crop_from_geometry(
            frame, full_masks[0]
        )
        left_cheek, left_geometry, left_cheek_box = _crop_from_geometry(
            frame, full_masks[1]
        )
        right_cheek, right_geometry, right_cheek_box = _crop_from_geometry(
            frame, full_masks[2]
        )

        face_box = (
            int(np.min(integer_points[:, 0])),
            int(np.min(integer_points[:, 1])),
            int(np.max(integer_points[:, 0])) + 1,
            int(np.max(integer_points[:, 1])) + 1,
        )

        return FaceRegions(
            forehead=forehead,
            left_cheek=left_cheek,
            right_cheek=right_cheek,
            face_box=face_box,
            forehead_box=forehead_box,
            left_cheek_box=left_cheek_box,
            right_cheek_box=right_cheek_box,
            forehead_geometry=forehead_geometry,
            left_cheek_geometry=left_geometry,
            right_cheek_geometry=right_geometry,
        )


class HaarRegionTracker:
    """Fallback face tracker using the original proportional rectangles."""

    name = "Haar cascade rectangles (fallback)"

    def __init__(self, config: IPPGConfig) -> None:
        self._config = config
        self._detector = _load_face_detector(config)
        print(f"Face tracker: {self.name}")

    def close(self) -> None:
        """Match the MediaPipe tracker interface; Haar owns no resources."""

    def detect(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
    ) -> FaceRegions | None:
        """Detect the largest face and return proportional rectangle ROIs."""

        del timestamp_ms
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._detector.detectMultiScale(
            gray_frame,
            scaleFactor=self._config.face_scale_factor,
            minNeighbors=self._config.face_min_neighbors,
        )
        face = _largest_face(faces)
        if face is None:
            return None
        return _extract_rois(frame, face)


def _create_region_tracker(config: IPPGConfig):
    """Prefer MediaPipe and fall back to Haar if it cannot be initialized."""

    if config.prefer_mediapipe:
        try:
            return MediaPipeRegionTracker(config)
        except Exception as error:
            print(f"MediaPipe unavailable ({error}); using Haar fallback.")
    return HaarRegionTracker(config)


def _largest_face(faces: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the largest detected face, or ``None`` when no face is found."""

    if len(faces) == 0:
        return None

    x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
    return int(x), int(y), int(width), int(height)


def _load_face_detector(config: IPPGConfig) -> cv2.CascadeClassifier:
    """Load a Haar cascade from OpenCV data or the project's local fallback."""

    module_directory = Path(__file__).resolve().parent
    project_root = module_directory.parent
    bundled_default = (
        module_directory / "assets" / "haarcascade_frontalface_default.xml"
    )
    third_party_fallback = (
        project_root
        / "third_party"
        / "deepgaze-master"
        / "etc"
        / "xml"
        / "haarcascade_frontalface_alt.xml"
    )
    opencv_default = (
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    )

    candidates = []
    if config.face_cascade_path is not None:
        candidates.append(config.face_cascade_path)
    candidates.extend([bundled_default, opencv_default, third_party_fallback])

    for cascade_path in candidates:
        resolved_path = Path(cascade_path).expanduser().resolve()
        if not resolved_path.is_file():
            continue

        detector = cv2.CascadeClassifier(str(resolved_path))
        if not detector.empty():
            print(f"Face cascade: {resolved_path}")
            return detector

    checked_paths = "\n".join(f"  - {Path(path)}" for path in candidates)
    raise FileNotFoundError(
        "Could not load a frontal-face Haar cascade. Checked:\n"
        f"{checked_paths}"
    )


def _relative_face_box(
    face: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
) -> tuple[int, int, int, int]:
    """Convert a box expressed as face-relative ratios to frame coordinates."""

    face_x, face_y, face_width, face_height = face
    frame_height, frame_width = frame_shape[:2]

    left = int(np.clip(face_x + face_width * left_ratio, 0, frame_width))
    top = int(np.clip(face_y + face_height * top_ratio, 0, frame_height))
    right = int(np.clip(face_x + face_width * right_ratio, 0, frame_width))
    bottom = int(np.clip(face_y + face_height * bottom_ratio, 0, frame_height))

    if right <= left or bottom <= top:
        raise ValueError("A face-relative ROI produced an empty box.")

    return left, top, right, bottom


def _crop_box(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    """Crop a frame using a (left, top, right, bottom) box."""

    left, top, right, bottom = box
    return frame[top:bottom, left:right]


def _extract_rois(
    frame: np.ndarray,
    face: tuple[int, int, int, int],
) -> FaceRegions:
    """Extract proportional forehead and cheek regions for the Haar fallback."""

    x, y, face_width, face_height = face

    # The ratios follow the reference image: a wide upper-forehead region and
    # two lower regions under the eyes. Keeping the cheeks separate avoids the
    # nose; their pixels are combined after skin detection.
    forehead_box = _relative_face_box(face, frame.shape, 0.18, 0.08, 0.82, 0.28)
    left_cheek_box = _relative_face_box(face, frame.shape, 0.10, 0.45, 0.46, 0.70)
    right_cheek_box = _relative_face_box(face, frame.shape, 0.54, 0.45, 0.90, 0.70)

    forehead = _crop_box(frame, forehead_box)
    left_cheek = _crop_box(frame, left_cheek_box)
    right_cheek = _crop_box(frame, right_cheek_box)

    if any(
        region.size == 0
        for region in (forehead, left_cheek, right_cheek)
    ):
        raise ValueError("One or more face regions are empty.")

    return FaceRegions(
        forehead=forehead,
        left_cheek=left_cheek,
        right_cheek=right_cheek,
        face_box=(x, y, x + face_width, y + face_height),
        forehead_box=forehead_box,
        left_cheek_box=left_cheek_box,
        right_cheek_box=right_cheek_box,
        forehead_geometry=np.full(forehead.shape[:2], 255, dtype=np.uint8),
        left_cheek_geometry=np.full(left_cheek.shape[:2], 255, dtype=np.uint8),
        right_cheek_geometry=np.full(right_cheek.shape[:2], 255, dtype=np.uint8),
    )


class AdaptiveSkinDetector:
    """Learn clip-specific YCrCb limits and temporally smooth skin masks."""

    def __init__(self, config: IPPGConfig) -> None:
        self._config = config
        self._calibration_start: float | None = None
        self._calibration_samples: list[np.ndarray] = []
        self._mask_probabilities: dict[str, np.ndarray] = {}
        self._bounds = (
            config.skin_cr_min,
            config.skin_cr_max,
            config.skin_cb_min,
            config.skin_cb_max,
        )
        self.ready = not config.adaptive_skin_detection
        if self.ready:
            print("Skin detector: fixed YCrCb + HSV thresholds")
        else:
            print(
                "Skin detector: adaptive YCrCb calibration "
                f"({config.skin_calibration_seconds:.1f} s)"
            )

    def _collect_region_samples(
        self,
        region: np.ndarray,
        geometry_mask: np.ndarray,
        sample_limit: int,
    ) -> None:
        """Collect deterministic, illumination-safe Cr/Cb calibration pixels."""

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        reliable_color = (
            (hsv[:, :, 1] >= 5)
            & (hsv[:, :, 2] >= self._config.skin_value_min)
        )
        sample_mask = (geometry_mask > 0) & reliable_color

        ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
        chroma = ycrcb[:, :, 1:3][sample_mask]
        if chroma.size == 0:
            return

        if chroma.shape[0] > sample_limit:
            step = int(np.ceil(chroma.shape[0] / sample_limit))
            chroma = chroma[::step][:sample_limit]
        self._calibration_samples.append(chroma.astype(np.uint8, copy=False))

    def calibrate(self, regions: FaceRegions, timestamp_seconds: float) -> None:
        """Observe ROI colors and freeze robust adaptive limits when ready."""

        if self.ready:
            return
        if self._calibration_start is None:
            self._calibration_start = timestamp_seconds

        per_region_limit = max(
            1,
            self._config.skin_calibration_samples_per_frame // 3,
        )
        for region, geometry in (
            (regions.forehead, regions.forehead_geometry),
            (regions.left_cheek, regions.left_cheek_geometry),
            (regions.right_cheek, regions.right_cheek_geometry),
        ):
            self._collect_region_samples(region, geometry, per_region_limit)

        elapsed = timestamp_seconds - self._calibration_start
        if elapsed >= self._config.skin_calibration_seconds:
            self._finish_calibration()

    def _finish_calibration(self) -> None:
        """Calculate median/MAD limits, with fixed thresholds as a safe backup."""

        if not self._calibration_samples:
            print("Skin calibration found no usable pixels; using fixed limits.")
            self.ready = True
            return

        samples = np.concatenate(self._calibration_samples, axis=0).astype(float)
        median = np.median(samples, axis=0)
        mad = np.median(np.abs(samples - median), axis=0)
        robust_sigma = 1.4826 * mad
        half_width = np.maximum(
            self._config.skin_min_chroma_half_width,
            self._config.skin_mad_multiplier * robust_sigma,
        )

        cr_min = int(np.clip(np.floor(median[0] - half_width[0]), 0, 255))
        cr_max = int(np.clip(np.ceil(median[0] + half_width[0]), 0, 255))
        cb_min = int(np.clip(np.floor(median[1] - half_width[1]), 0, 255))
        cb_max = int(np.clip(np.ceil(median[1] + half_width[1]), 0, 255))
        self._bounds = cr_min, cr_max, cb_min, cb_max
        self._calibration_samples.clear()
        self._mask_probabilities.clear()
        self.ready = True
        print(
            "Adaptive skin limits: "
            f"Cr={cr_min}..{cr_max}, Cb={cb_min}..{cb_max}"
        )

    def calibration_progress(self, timestamp_seconds: float) -> float:
        """Return calibration progress from zero to one for preview text."""

        if self.ready:
            return 1.0
        if self._calibration_start is None:
            return 0.0
        duration = max(self._config.skin_calibration_seconds, 1e-6)
        return float(
            np.clip((timestamp_seconds - self._calibration_start) / duration, 0, 1)
        )

    def create_mask(
        self,
        region: np.ndarray,
        geometry_mask: np.ndarray,
        region_name: str,
    ) -> np.ndarray:
        """Create one geometry-limited and temporally smoothed skin mask."""

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        if self.ready:
            cr_min, cr_max, cb_min, cb_max = self._bounds
            ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
            chroma_mask = cv2.inRange(
                ycrcb,
                np.array([0, cr_min, cb_min], dtype=np.uint8),
                np.array([255, cr_max, cb_max], dtype=np.uint8),
            )

            if self._config.adaptive_skin_detection:
                # Hue is intentionally unrestricted after adaptive calibration:
                # camera white balance can move otherwise valid skin across it.
                color_quality_mask = cv2.inRange(
                    hsv,
                    np.array(
                        [0, self._config.skin_saturation_min, self._config.skin_value_min],
                        dtype=np.uint8,
                    ),
                    np.array([179, 255, 255], dtype=np.uint8),
                )
            else:
                color_quality_mask = cv2.inRange(
                    hsv,
                    np.array(
                        [0, self._config.skin_saturation_min, self._config.skin_value_min],
                        dtype=np.uint8,
                    ),
                    np.array([self._config.skin_hue_max, 255, 255], dtype=np.uint8),
                )
            candidate = cv2.bitwise_and(chroma_mask, color_quality_mask)
        else:
            # During warm-up, show plausible pixels but do not feed them to iPPG.
            candidate = cv2.inRange(
                hsv,
                np.array([0, 5, self._config.skin_value_min], dtype=np.uint8),
                np.array([179, 255, 255], dtype=np.uint8),
            )

        candidate = cv2.bitwise_and(candidate, geometry_mask)
        current_probability = candidate.astype(np.float32) / 255.0
        previous_probability = self._mask_probabilities.get(region_name)
        if previous_probability is not None:
            if previous_probability.shape != current_probability.shape:
                previous_probability = cv2.resize(
                    previous_probability,
                    (current_probability.shape[1], current_probability.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            alpha = self._config.skin_mask_ema_alpha
            current_probability = (
                alpha * current_probability
                + (1.0 - alpha) * previous_probability
            )
        current_probability[geometry_mask == 0] = 0.0
        self._mask_probabilities[region_name] = current_probability

        skin_mask = np.where(current_probability >= 0.5, 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        return cv2.bitwise_and(skin_mask, geometry_mask)


def mean_skin_rgb(
    region: np.ndarray,
    geometry_mask: np.ndarray,
    region_name: str,
    skin_detector: AdaptiveSkinDetector,
    config: IPPGConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return mean [R, G, B], the skin mask, and valid-pixel count."""

    skin_mask = skin_detector.create_mask(region, geometry_mask, region_name)
    skin_pixels_bgr = region[skin_mask > 0]
    pixel_count = int(skin_pixels_bgr.shape[0])

    if pixel_count < config.minimum_skin_pixels_per_region:
        raise ValueError(
            "Not enough skin pixels were detected in one of the face regions."
        )

    mean_bgr = np.mean(skin_pixels_bgr, axis=0)
    mean_rgb = mean_bgr[::-1].astype(float)
    return mean_rgb, skin_mask, pixel_count


def validate_roi_config(config: IPPGConfig) -> None:
    """Validate multi-patch ROI selection settings."""

    if config.roi_method not in {"quality", "legacy"}:
        raise ValueError("roi_method must be 'quality' or 'legacy'.")
    if not 0.0 <= config.roi_trim_fraction < 0.5:
        raise ValueError("roi_trim_fraction must be in [0, 0.5).")
    if config.roi_min_patch_pixels < 1:
        raise ValueError("roi_min_patch_pixels must be positive.")
    if not 0.0 <= config.roi_min_quality <= 1.0:
        raise ValueError("roi_min_quality must be between 0 and 1.")
    if not 0 <= config.roi_exposure_low < config.roi_exposure_high <= 255:
        raise ValueError("ROI exposure limits must satisfy 0 <= low < high <= 255.")
    if config.roi_homogeneity_scale <= 0 or not np.isfinite(config.roi_homogeneity_scale):
        raise ValueError("roi_homogeneity_scale must be positive and finite.")
    integer_settings = (
        config.roi_forehead_columns,
        config.roi_cheek_rows,
        config.roi_keep_forehead_patches,
        config.roi_keep_cheek_patches,
    )
    if any(value < 1 for value in integer_settings):
        raise ValueError("ROI grid and retained-patch counts must be positive.")
    if config.roi_keep_forehead_patches > config.roi_forehead_columns:
        raise ValueError("Cannot retain more forehead patches than the grid contains.")
    if config.roi_keep_cheek_patches > config.roi_cheek_rows:
        raise ValueError("Cannot retain more cheek patches than the grid contains.")


def robust_trimmed_mean_rgb(pixels_bgr: np.ndarray, trim_fraction: float) -> np.ndarray:
    """Return channel-wise trimmed mean RGB to suppress highlights/outliers."""

    pixels = np.asarray(pixels_bgr, dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 3 or pixels.shape[0] == 0:
        raise ValueError("Trimmed RGB mean needs a non-empty (N, 3) BGR array.")
    if not np.all(np.isfinite(pixels)) or not 0.0 <= trim_fraction < 0.5:
        raise ValueError("Invalid pixels or trim fraction for robust RGB mean.")
    trim_count = int(np.floor(pixels.shape[0] * trim_fraction))
    sorted_pixels = np.sort(pixels, axis=0)
    if trim_count > 0 and 2 * trim_count < pixels.shape[0]:
        sorted_pixels = sorted_pixels[trim_count:-trim_count]
    return np.mean(sorted_pixels, axis=0)[::-1].astype(float)


def select_quality_roi(
    region: np.ndarray,
    geometry_mask: np.ndarray,
    region_name: str,
    rows: int,
    columns: int,
    keep_count: int,
    skin_detector: AdaptiveSkinDetector,
    config: IPPGConfig,
) -> tuple[ROISelection, np.ndarray]:
    """Split one anatomical ROI, score patches and retain the cleanest ones."""

    validate_roi_config(config)
    if region.ndim != 3 or region.shape[2] != 3 or geometry_mask.shape != region.shape[:2]:
        raise ValueError("ROI image and geometry mask shapes do not match.")
    row_edges = np.linspace(0, region.shape[0], rows + 1, dtype=int)
    column_edges = np.linspace(0, region.shape[1], columns + 1, dtype=int)
    candidates: list[dict] = []
    patch_index = 0
    for row in range(rows):
        for column in range(columns):
            top, bottom = row_edges[row], row_edges[row + 1]
            left, right = column_edges[column], column_edges[column + 1]
            patch_region = region[top:bottom, left:right]
            patch_geometry = geometry_mask[top:bottom, left:right]
            geometry_pixels = int(np.count_nonzero(patch_geometry))
            if patch_region.size == 0 or geometry_pixels == 0:
                patch_index += 1
                continue
            patch_mask = skin_detector.create_mask(
                patch_region, patch_geometry, f"{region_name}_patch_{patch_index}"
            )
            pixels_bgr = patch_region[patch_mask > 0]
            pixel_count = int(pixels_bgr.shape[0])
            if pixel_count < config.roi_min_patch_pixels:
                patch_index += 1
                continue
            pixels_rgb = pixels_bgr[:, ::-1].astype(float)
            luminance = (
                0.2126 * pixels_rgb[:, 0]
                + 0.7152 * pixels_rgb[:, 1]
                + 0.0722 * pixels_rgb[:, 2]
            )
            exposure_quality = float(np.mean(
                (luminance >= config.roi_exposure_low)
                & (luminance <= config.roi_exposure_high)
            ))
            coverage = pixel_count / geometry_pixels
            channel_means = np.mean(pixels_rgb, axis=0)
            relative_dispersion = float(np.mean(
                np.std(pixels_rgb, axis=0) / np.maximum(np.abs(channel_means), 1e-8)
            ))
            homogeneity = float(np.exp(
                -relative_dispersion / config.roi_homogeneity_scale
            ))
            quality = float(np.clip(coverage * exposure_quality * homogeneity, 0.0, 1.0))
            if quality >= config.roi_min_quality:
                candidates.append({
                    "index": patch_index,
                    "rgb": robust_trimmed_mean_rgb(pixels_bgr, config.roi_trim_fraction),
                    "quality": quality,
                    "pixel_count": pixel_count,
                    "mask": patch_mask,
                    "bounds": (top, bottom, left, right),
                })
            patch_index += 1
    if not candidates:
        raise ValueError(f"No usable quality patches were found in {region_name}.")
    selected = sorted(candidates, key=lambda item: (-item["quality"], item["index"]))[
        :keep_count
    ]
    qualities = np.asarray([item["quality"] for item in selected], dtype=float)
    rgb = np.average(
        np.stack([item["rgb"] for item in selected]), axis=0, weights=qualities
    )
    selected_mask = np.zeros_like(geometry_mask, dtype=np.uint8)
    for item in selected:
        top, bottom, left, right = item["bounds"]
        selected_mask[top:bottom, left:right] = cv2.bitwise_or(
            selected_mask[top:bottom, left:right], item["mask"]
        )
    return (
        ROISelection(
            rgb=np.asarray(rgb, dtype=float),
            quality=float(np.mean(qualities)),
            pixel_count=int(sum(item["pixel_count"] for item in selected)),
            selected_patches=tuple(int(item["index"]) for item in selected),
        ),
        selected_mask,
    )


def quality_weighted_face_rgb(
    regions: FaceRegions,
    skin_detector: AdaptiveSkinDetector,
    config: IPPGConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    """Select robust patches and combine valid anatomical regions by quality."""

    specifications = (
        (
            "forehead",
            regions.forehead,
            regions.forehead_geometry,
            1,
            config.roi_forehead_columns,
            config.roi_keep_forehead_patches,
        ),
        (
            "left_cheek",
            regions.left_cheek,
            regions.left_cheek_geometry,
            config.roi_cheek_rows,
            1,
            config.roi_keep_cheek_patches,
        ),
        (
            "right_cheek",
            regions.right_cheek,
            regions.right_cheek_geometry,
            config.roi_cheek_rows,
            1,
            config.roi_keep_cheek_patches,
        ),
    )
    selections: list[ROISelection | None] = []
    masks: list[np.ndarray] = []
    for name, region, geometry, rows, columns, keep_count in specifications:
        try:
            selection, mask = select_quality_roi(
                region,
                geometry,
                name,
                rows,
                columns,
                keep_count,
                skin_detector,
                config,
            )
        except ValueError:
            selection = None
            mask = np.zeros_like(geometry, dtype=np.uint8)
        selections.append(selection)
        masks.append(mask)
    valid = [(index, item) for index, item in enumerate(selections) if item is not None]
    if not valid:
        raise ValueError("No face region contained a usable quality patch.")
    region_qualities = np.asarray(
        [0.0 if item is None else item.quality for item in selections], dtype=float
    )
    region_weights = region_qualities / np.sum(region_qualities)
    combined_rgb = np.sum(
        [region_weights[index] * item.rgb for index, item in valid], axis=0
    )
    forehead_rgb = combined_rgb if selections[0] is None else selections[0].rgb
    cheek_valid = [(index, selections[index]) for index in (1, 2) if selections[index] is not None]
    if cheek_valid:
        cheek_weights = np.asarray([item.quality for _, item in cheek_valid], dtype=float)
        cheeks_rgb = np.average(
            np.stack([item.rgb for _, item in cheek_valid]), axis=0, weights=cheek_weights
        )
    else:
        cheeks_rgb = combined_rgb.copy()
    return (
        np.asarray(forehead_rgb, dtype=float),
        np.asarray(cheeks_rgb, dtype=float),
        np.asarray(combined_rgb, dtype=float),
        (masks[0], masks[1], masks[2]),
        region_qualities,
        region_weights,
    )


def _weighted_rgb(
    rgb_and_counts: list[tuple[np.ndarray, int]],
) -> np.ndarray:
    """Combine region means while preserving each region's pixel weight."""

    rgb_values = np.stack([rgb for rgb, _ in rgb_and_counts])
    pixel_counts = np.asarray([count for _, count in rgb_and_counts], dtype=float)
    return np.average(rgb_values, axis=0, weights=pixel_counts)


def _overlay_skin_pixels(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    skin_mask: np.ndarray,
    geometry_mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    alpha: float = 0.45,
) -> None:
    """Tint detected skin pixels and draw the polygon ROI boundary."""

    left, top, right, bottom = box
    region = frame[top:bottom, left:right]
    color_layer = np.full_like(region, color_bgr)
    blended = cv2.addWeighted(region, 1.0 - alpha, color_layer, alpha, 0)
    region[skin_mask > 0] = blended[skin_mask > 0]
    contours, _ = cv2.findContours(
        geometry_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(region, contours, -1, color_bgr, 2)


def _boxes_intersect(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    """Return whether two (left, top, right, bottom) boxes overlap."""

    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def _integral_box_mean(
    integral_image: np.ndarray,
    box: tuple[int, int, int, int],
) -> float:
    """Calculate a rectangular mean from an OpenCV integral image."""

    left, top, right, bottom = box
    total = (
        integral_image[bottom, right]
        - integral_image[top, right]
        - integral_image[bottom, left]
        + integral_image[top, left]
    )
    return float(total / ((right - left) * (bottom - top)))


class StableBackgroundPatch:
    """Select one fixed, low-texture, temporally stable background patch."""

    def __init__(self, config: IPPGConfig) -> None:
        if config.background_patch_size < 8:
            raise ValueError("Background patch size must be at least 8 pixels.")
        if config.background_calibration_frames < 2:
            raise ValueError("Background calibration needs at least two frames.")

        self._config = config
        self._candidate_boxes: list[tuple[int, int, int, int]] = []
        self._valid = np.empty(0, dtype=bool)
        self._count = np.empty(0, dtype=int)
        self._sum_y = np.empty(0, dtype=float)
        self._sum_y_squared = np.empty(0, dtype=float)
        self._sum_gradient = np.empty(0, dtype=float)
        self._frames_seen = 0
        self.box: tuple[int, int, int, int] | None = None

    def _expanded_face_box(
        self,
        face_box: tuple[int, int, int, int],
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        margin = self._config.background_exclusion_margin_pixels
        return (
            max(0, face_box[0] - margin),
            max(0, face_box[1] - margin),
            min(width, face_box[2] + margin),
            min(height, face_box[3] + margin),
        )

    def _initialize_candidates(
        self,
        frame_shape: tuple[int, ...],
        face_box: tuple[int, int, int, int],
    ) -> None:
        height, width = frame_shape[:2]
        size = self._config.background_patch_size
        if size > height or size > width:
            raise ValueError(
                f"Background patch {size}x{size} does not fit the video frame."
            )

        excluded_face = self._expanded_face_box(face_box, frame_shape)
        stride = size
        for top in range(0, height - size + 1, stride):
            for left in range(0, width - size + 1, stride):
                box = (left, top, left + size, top + size)
                if not _boxes_intersect(box, excluded_face):
                    self._candidate_boxes.append(box)

        if not self._candidate_boxes:
            raise RuntimeError("No background patch candidates remain outside the face.")

        candidate_count = len(self._candidate_boxes)
        self._valid = np.ones(candidate_count, dtype=bool)
        self._count = np.zeros(candidate_count, dtype=int)
        self._sum_y = np.zeros(candidate_count, dtype=float)
        self._sum_y_squared = np.zeros(candidate_count, dtype=float)
        self._sum_gradient = np.zeros(candidate_count, dtype=float)

    def _choose_best_candidate(self) -> None:
        enough_samples = self._count >= max(2, int(0.8 * self._frames_seen))
        eligible = self._valid & enough_samples
        if not np.any(eligible):
            raise RuntimeError(
                "Could not find a stable, unsaturated background patch."
            )

        indices = np.flatnonzero(eligible)
        counts = self._count[indices].astype(float)
        mean_y = self._sum_y[indices] / counts
        variance_y = np.maximum(
            self._sum_y_squared[indices] / counts - mean_y**2,
            0.0,
        )
        temporal_std = np.sqrt(variance_y)
        spatial_gradient = self._sum_gradient[indices] / counts

        def normalize(values: np.ndarray) -> np.ndarray:
            spread = float(np.ptp(values))
            if spread <= self._config.background_epsilon:
                return np.zeros_like(values)
            return (values - np.min(values)) / spread

        score = 0.7 * normalize(temporal_std) + 0.3 * normalize(
            spatial_gradient
        )
        selected_index = int(indices[int(np.argmin(score))])
        self.box = self._candidate_boxes[selected_index]
        print(f"Stable background patch: {self.box}")

    def update(
        self,
        frame: np.ndarray,
        face_box: tuple[int, int, int, int],
    ) -> float | None:
        """Observe calibration frames, then return mean Y from the fixed patch."""

        y_channel = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        if self.box is not None:
            left, top, right, bottom = self.box
            return float(np.mean(y_channel[top:bottom, left:right]))

        if not self._candidate_boxes:
            self._initialize_candidates(frame.shape, face_box)

        excluded_face = self._expanded_face_box(face_box, frame.shape)
        gradient_x = cv2.Sobel(y_channel, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(y_channel, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(gradient_x, gradient_y)
        y_integral = cv2.integral(y_channel)
        gradient_integral = cv2.integral(gradient)

        for index, box in enumerate(self._candidate_boxes):
            if not self._valid[index]:
                continue
            if _boxes_intersect(box, excluded_face):
                self._valid[index] = False
                continue

            mean_y = _integral_box_mean(y_integral, box)
            if not (
                self._config.background_min_luminance
                <= mean_y
                <= self._config.background_max_luminance
            ):
                self._valid[index] = False
                continue

            mean_gradient = _integral_box_mean(gradient_integral, box)
            self._count[index] += 1
            self._sum_y[index] += mean_y
            self._sum_y_squared[index] += mean_y * mean_y
            self._sum_gradient[index] += mean_gradient

        self._frames_seen += 1
        if self._frames_seen >= self._config.background_calibration_frames:
            self._choose_best_candidate()
            return self.update(frame, face_box)
        return None


def resample_to_fps(
    time_seconds: np.ndarray,
    values: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate 1-D or multi-channel values onto a uniform video-FPS grid."""

    time_seconds = np.asarray(time_seconds, dtype=float)
    values = np.asarray(values, dtype=float)
    if time_seconds.ndim != 1:
        raise ValueError("time_seconds must be a one-dimensional array.")
    if values.ndim not in (1, 2):
        raise ValueError("values must be a one- or two-dimensional array.")
    if values.shape[0] != time_seconds.size:
        raise ValueError("time_seconds and values must contain the same samples.")
    if time_seconds.size < 2:
        raise ValueError("At least two samples are required for resampling.")
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("FPS must be a positive finite number.")
    if not np.all(np.isfinite(time_seconds)) or not np.all(np.isfinite(values)):
        raise ValueError("Cannot resample arrays containing NaN or infinity.")
    if np.any(np.diff(time_seconds) <= 0):
        raise ValueError("time_seconds must be strictly increasing.")

    sample_count = int(
        np.floor((time_seconds[-1] - time_seconds[0]) * fps)
    ) + 1
    if sample_count < 2:
        raise ValueError("The signal duration is too short for resampling.")
    uniform_time = time_seconds[0] + np.arange(sample_count) / fps

    if values.ndim == 1:
        uniform_values = np.interp(uniform_time, time_seconds, values)
    else:
        uniform_values = np.column_stack(
            [
                np.interp(uniform_time, time_seconds, values[:, channel])
                for channel in range(values.shape[1])
            ]
        )
    return uniform_time, uniform_values


def relative_luminance(
    luminance: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Convert absolute Y luminance to fractional change around its mean."""

    luminance = np.asarray(luminance, dtype=float)
    if luminance.ndim != 1:
        raise ValueError("Luminance must be a one-dimensional array.")
    if luminance.size == 0 or not np.all(np.isfinite(luminance)):
        raise ValueError("Luminance must contain finite samples.")

    mean_luminance = float(np.mean(luminance))
    if abs(mean_luminance) <= epsilon:
        raise ValueError("Mean luminance is zero; relative Y is undefined.")
    return luminance / mean_luminance - 1.0


def calculate_pos(
    rgb: np.ndarray,
    fps: float,
    window_seconds: float = 1.6,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Extract a pulse using Plane-Orthogonal-to-Skin overlap-add.

    ``rgb`` must have shape ``(samples, 3)`` in [R, G, B] order. Each window
    is normalized by its channel means, projected onto the two POS axes, and
    combined using their standard-deviation ratio. The centered window pulse
    is overlap-added to the output as described by the original POS method.
    """

    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("POS input must have shape (samples, 3) in RGB order.")
    if not np.all(np.isfinite(rgb)):
        raise ValueError("POS input cannot contain NaN or infinity.")
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("FPS must be a positive finite number.")
    if window_seconds <= 0 or not np.isfinite(window_seconds):
        raise ValueError("POS window duration must be positive and finite.")
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("POS epsilon must be positive and finite.")

    window_samples = max(2, int(round(window_seconds * fps)))
    if rgb.shape[0] < window_samples:
        raise ValueError(
            "POS needs at least "
            f"{window_samples} samples ({window_seconds:.3f} s at {fps:.3f} FPS), "
            f"but received {rgb.shape[0]}."
        )

    pulse = np.zeros(rgb.shape[0], dtype=float)
    for start in range(rgb.shape[0] - window_samples + 1):
        end = start + window_samples
        window_rgb = rgb[start:end].T
        channel_means = np.mean(window_rgb, axis=1, keepdims=True)
        if np.any(np.abs(channel_means) <= epsilon):
            raise ValueError("POS encountered a zero RGB channel mean.")

        normalized = window_rgb / channel_means
        first_projection = normalized[1] - normalized[2]
        second_projection = (
            normalized[1] + normalized[2] - 2.0 * normalized[0]
        )

        second_std = float(np.std(second_projection))
        if second_std <= epsilon:
            alpha = 0.0
        else:
            alpha = float(np.std(first_projection)) / second_std

        window_pulse = first_projection + alpha * second_projection
        window_pulse -= np.mean(window_pulse)
        pulse[start:end] += window_pulse

    if not np.all(np.isfinite(pulse)):
        raise RuntimeError("POS produced NaN or infinity.")
    return pulse


def extract_pulse_signal(
    combined_rgb: np.ndarray,
    fps: float,
    config: IPPGConfig,
) -> tuple[np.ndarray | None, str]:
    """Create the configured pulse signal while retaining G-R as a baseline."""

    method = config.pulse_method.strip().lower()
    if method == "pos":
        pulse = calculate_pos(
            combined_rgb,
            fps,
            config.pos_window_seconds,
            config.pos_epsilon,
        )
        return pulse, "pos"
    if method == "g-r":
        return combined_rgb[:, 1] - combined_rgb[:, 0], "g-r"
    if method == "pos-ica":
        # Both branches are built together in analyze_signals. Delaying POS
        # here also lets ICA remain available if POS itself cannot be formed.
        return None, "pos-ica"
    raise ValueError(
        f"Unknown pulse method {config.pulse_method!r}; "
        "choose 'pos', 'g-r', or 'pos-ica'."
    )


def extract_video_signals(
    video_path: Path,
    config: IPPGConfig,
) -> ExtractedSignals:
    """Extract skin-only RGB arrays and synchronized analysis signals."""

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    region_tracker = _create_region_tracker(config)
    skin_detector = AdaptiveSkinDetector(config)
    background_method = resolve_background_method(config, config.pulse_method)
    use_background = background_method != "none"
    background_tracker = StableBackgroundPatch(config) if use_background else None

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open the video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise ValueError("The video reports an invalid FPS value.")

    times: list[float] = []
    forehead_rgb_values: list[np.ndarray] = []
    cheeks_rgb_values: list[np.ndarray] = []
    combined_rgb_values: list[np.ndarray] = []
    background_values: list[float] = []
    roi_quality_values: list[np.ndarray] = []
    roi_weight_values: list[np.ndarray] = []
    validate_roi_config(config)

    try:
        while True:
            frame_was_read, frame = capture.read()
            if not frame_was_read:
                break

            frame_number = capture.get(cv2.CAP_PROP_POS_FRAMES)
            current_time = frame_number / fps
            if (
                config.max_duration_seconds is not None
                and current_time > config.max_duration_seconds
            ):
                break

            timestamp_ms = int(round((frame_number - 1) * 1000.0 / fps))
            regions = region_tracker.detect(frame, timestamp_ms)

            if regions is not None:
                background_y = (
                    background_tracker.update(frame, regions.face_box)
                    if background_tracker is not None
                    else None
                )
                skin_detector.calibrate(regions, current_time)
                try:
                    if config.roi_method == "quality":
                        (
                            forehead_rgb,
                            cheeks_rgb,
                            combined_rgb,
                            quality_masks,
                            roi_qualities,
                            roi_weights,
                        ) = quality_weighted_face_rgb(regions, skin_detector, config)
                        forehead_mask, left_cheek_mask, right_cheek_mask = quality_masks
                    else:
                        forehead_rgb, forehead_mask, forehead_count = mean_skin_rgb(
                            regions.forehead,
                            regions.forehead_geometry,
                            "forehead",
                            skin_detector,
                            config,
                        )
                        left_cheek_rgb, left_cheek_mask, left_cheek_count = mean_skin_rgb(
                            regions.left_cheek,
                            regions.left_cheek_geometry,
                            "left_cheek",
                            skin_detector,
                            config,
                        )
                        right_cheek_rgb, right_cheek_mask, right_cheek_count = mean_skin_rgb(
                            regions.right_cheek,
                            regions.right_cheek_geometry,
                            "right_cheek",
                            skin_detector,
                            config,
                        )
                        cheeks_rgb = _weighted_rgb(
                            [
                                (left_cheek_rgb, left_cheek_count),
                                (right_cheek_rgb, right_cheek_count),
                            ]
                        )
                        combined_rgb = _weighted_rgb(
                            [
                                (forehead_rgb, forehead_count),
                                (left_cheek_rgb, left_cheek_count),
                                (right_cheek_rgb, right_cheek_count),
                            ]
                        )
                        counts = np.asarray(
                            [forehead_count, left_cheek_count, right_cheek_count],
                            dtype=float,
                        )
                        roi_qualities = np.ones(3, dtype=float)
                        roi_weights = counts / np.sum(counts)
                except ValueError:
                    pass
                else:
                    # Warm-up frames teach the color model but are not mixed
                    # into the final iPPG time series.
                    if skin_detector.ready and (
                        not use_background or background_y is not None
                    ):
                        times.append(current_time)
                        forehead_rgb_values.append(forehead_rgb)
                        cheeks_rgb_values.append(cheeks_rgb)
                        combined_rgb_values.append(combined_rgb)
                        if background_y is not None:
                            background_values.append(background_y)
                        roi_quality_values.append(roi_qualities)
                        roi_weight_values.append(roi_weights)

                    if config.show_preview:
                        # Yellow = forehead; cyan = left and right cheeks.
                        _overlay_skin_pixels(
                            frame,
                            regions.forehead_box,
                            forehead_mask,
                            regions.forehead_geometry,
                            (0, 255, 255),
                        )
                        _overlay_skin_pixels(
                            frame,
                            regions.left_cheek_box,
                            left_cheek_mask,
                            regions.left_cheek_geometry,
                            (255, 255, 0),
                        )
                        _overlay_skin_pixels(
                            frame,
                            regions.right_cheek_box,
                            right_cheek_mask,
                            regions.right_cheek_geometry,
                            (255, 255, 0),
                        )
                        if (
                            background_tracker is not None
                            and background_tracker.box is not None
                        ):
                            left, top, right, bottom = background_tracker.box
                            cv2.rectangle(
                                frame,
                                (left, top),
                                (right, bottom),
                                (255, 0, 255),
                                2,
                            )

                if config.show_preview and not skin_detector.ready:
                    progress = 100.0 * skin_detector.calibration_progress(
                        current_time
                    )
                    cv2.putText(
                        frame,
                        f"Calibrating skin color: {progress:.0f}%",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            if config.show_preview:
                cv2.imshow("iPPG v2", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        region_tracker.close()
        if config.show_preview:
            cv2.destroyAllWindows()

    if len(times) < 2:
        raise RuntimeError(
            "Not enough valid face frames were found to build an iPPG signal."
        )

    raw_time = np.asarray(times, dtype=float)
    uniform_time, forehead_rgb = resample_to_fps(
        raw_time,
        np.asarray(forehead_rgb_values, dtype=float),
        fps,
    )
    _, cheeks_rgb = resample_to_fps(
        raw_time,
        np.asarray(cheeks_rgb_values, dtype=float),
        fps,
    )
    _, combined_rgb = resample_to_fps(
        raw_time,
        np.asarray(combined_rgb_values, dtype=float),
        fps,
    )
    background = None
    background_patch_box = None
    if use_background:
        _, background = resample_to_fps(
            raw_time,
            np.asarray(background_values, dtype=float),
            fps,
        )
        background = relative_luminance(background, config.background_epsilon)
        if background_tracker is None or background_tracker.box is None:
            raise RuntimeError("A stable background patch was never selected.")
        background_patch_box = background_tracker.box
    _, roi_quality = resample_to_fps(
        raw_time,
        np.asarray(roi_quality_values, dtype=float),
        fps,
    )
    _, roi_weights = resample_to_fps(
        raw_time,
        np.asarray(roi_weight_values, dtype=float),
        fps,
    )
    pulse_signal, pulse_method = extract_pulse_signal(combined_rgb, fps, config)

    return ExtractedSignals(
        time_seconds=uniform_time,
        forehead_rgb=forehead_rgb,
        cheeks_rgb=cheeks_rgb,
        combined_rgb=combined_rgb,
        pulse_signal=pulse_signal,
        background=background,
        fps=fps,
        pulse_method=pulse_method,
        background_patch_box=background_patch_box,
        roi_quality=roi_quality,
        roi_weights=roi_weights,
        roi_method=config.roi_method,
        face_tracker_name=getattr(region_tracker, "name", None),
    )


def parse_wavelet_level(value: str) -> int | str:
    if value == "auto":
        return value
    try:
        level = int(value)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("Wavelet level must be 'auto' or a positive integer.") from error
    if level < 1:
        raise argparse.ArgumentTypeError("Wavelet level must be positive.")
    return level


def add_wavelet_arguments(parser: argparse.ArgumentParser, include_denoise: bool = True) -> None:
    """Keep CLI defaults identical across all consumers of the POS pipeline."""

    if include_denoise:
        parser.add_argument("--denoise", choices=("dwt", "none"), default="dwt",
                            help="POS denoising (default: dwt); G-R is always unchanged")
    parser.add_argument("--wavelet", default="db4", help="Discrete wavelet name (default: db4)")
    parser.add_argument("--wavelet-level", type=parse_wavelet_level, default="auto",
                        help="auto=min(4, maximum supported), or a positive integer")
    parser.add_argument("--wavelet-strength", type=float, default=1.0,
                        help="Non-negative soft-threshold multiplier (default: 1.0)")
    parser.add_argument("--wavelet-strategy", choices=("protect-pulse-band", "all-details"),
                        default="protect-pulse-band",
                        help=("DWT detail handling: protect pulse-overlapping bands (default), "
                              "or reproduce legacy all-detail thresholding"))


def wavelet_config_from_arguments(arguments: argparse.Namespace) -> dict:
    return {"denoise": getattr(arguments, "denoise", "dwt"),
            "wavelet": arguments.wavelet, "wavelet_level": arguments.wavelet_level,
            "wavelet_strength": arguments.wavelet_strength,
            "wavelet_strategy": arguments.wavelet_strategy}


def validate_denoise_config(config: IPPGConfig) -> None:
    if config.denoise not in {"none", "dwt"}:
        raise ValueError("Denoise must be 'none' or 'dwt'.")
    if not np.isfinite(config.wavelet_strength) or config.wavelet_strength < 0:
        raise ValueError("Wavelet strength must be finite and non-negative.")
    if config.wavelet_strategy not in {"protect-pulse-band", "all-details"}:
        raise ValueError("Wavelet strategy must be 'protect-pulse-band' or 'all-details'.")
    level = config.wavelet_level
    if level != "auto" and (isinstance(level, bool) or not isinstance(level, (int, np.integer)) or level < 1):
        raise ValueError("Wavelet level must be 'auto' or a positive integer.")


def wavelet_denoise(
    pulse: np.ndarray, config: IPPGConfig, fps: float,
) -> tuple[np.ndarray, dict]:
    """Full-clip db4 DWT denoising; approximation coefficients are preserved.

    Noise sigma comes only from cD1, never from contact PPG. Symmetric extension
    is used for both transforms, and IDWT padding is trimmed to input length.
    """

    validate_denoise_config(config)
    raw = np.asarray(pulse, dtype=float)
    if raw.ndim != 1 or raw.size == 0 or not np.all(np.isfinite(raw)):
        raise ValueError("Wavelet input must be a nonempty, finite 1-D pulse.")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Wavelet FPS must be finite and positive.")
    if pywt is None:
        raise RuntimeError("DWT requires PyWavelets. Install it in this Python environment with: "
                           "python -m pip install PyWavelets==1.9.0; or use --denoise none.")
    wavelet = pywt.Wavelet(config.wavelet)
    maximum = pywt.dwt_max_level(raw.size, wavelet.dec_len)
    automatic = config.wavelet_level == "auto"
    level = min(4, maximum) if automatic else int(config.wavelet_level)
    if level > maximum:
        raise ValueError(f"Wavelet level {level} exceeds maximum {maximum} for {raw.size} samples.")
    metadata = {
        "method": "dwt", "wavelet": wavelet.name, "requested_level": config.wavelet_level,
        "actual_level": level, "maximum_level": maximum, "extension_mode": "symmetric",
        "threshold_mode": "soft", "strategy": config.wavelet_strategy,
        "pulse_band_hz": [float(config.forehead_low_hz), float(config.forehead_high_hz)],
        "sampling_fps": float(fps),
        "strength": float(config.wavelet_strength),
        "sigma": 0.0, "threshold": 0.0, "status": "applied", "applied": False,
        "input_samples": int(raw.size), "pywavelets_version": pywt.__version__,
    }
    # Distribution metadata and module __version__ can differ in packaged wheels.
    try:
        metadata["pywavelets_distribution_version"] = distribution_version("PyWavelets")
    except PackageNotFoundError:
        metadata["pywavelets_distribution_version"] = None
    if level == 0:
        warnings.warn("Pulse too short for DWT; returning raw POS unchanged.", RuntimeWarning, stacklevel=2)
        metadata["status"] = "skipped_short_signal"
        return raw.copy(), metadata
    if config.wavelet_strength == 0:
        metadata["status"] = "skipped_zero_strength"
        return raw.copy(), metadata
    coefficients = pywt.wavedec(raw, wavelet, mode="symmetric", level=level)
    finest_detail = coefficients[-1]
    sigma = float(np.median(np.abs(finest_detail - np.median(finest_detail))) / 0.6745)
    threshold = float(config.wavelet_strength * sigma * np.sqrt(2 * np.log(raw.size)))
    if not np.isfinite(threshold):
        raise ValueError("Wavelet threshold is not finite.")
    metadata.update(sigma=sigma, threshold=threshold)
    if threshold == 0:
        metadata["status"] = "skipped_zero_threshold"
        return raw.copy(), metadata
    # Coefficients are [cA_level, cD_level, ..., cD1].  The bands below are
    # the standard dyadic approximation; boundaries are not ideal brick walls.
    def detail_band(detail_level: int) -> tuple[float, float]:
        return (fps / (2.0 ** (detail_level + 1)), fps / (2.0 ** detail_level))

    def overlaps_pulse_band(lower_hz: float, upper_hz: float) -> bool:
        return lower_hz < config.forehead_high_hz and upper_hz > config.forehead_low_hz

    modified = [coefficients[0].copy()]
    coefficient_report = [{
        "coefficient": f"cA{level}", "band_hz_approx": [0.0, fps / (2.0 ** (level + 1))],
        "action": "preserved_approximation", "energy_before": float(np.sum(coefficients[0] ** 2)),
        "energy_after": float(np.sum(coefficients[0] ** 2)), "zero_fraction_after": float(np.mean(coefficients[0] == 0)),
    }]
    for index, detail in enumerate(coefficients[1:], start=1):
        detail_level = level - index + 1
        lower_hz, upper_hz = detail_band(detail_level)
        protected = (config.wavelet_strategy == "protect-pulse-band" and
                     overlaps_pulse_band(lower_hz, upper_hz))
        output = detail.copy() if protected else pywt.threshold(detail, threshold, mode="soft")
        modified.append(output)
        coefficient_report.append({
            "coefficient": f"cD{detail_level}", "band_hz_approx": [lower_hz, upper_hz],
            "action": "preserved_pulse_overlap" if protected else "soft_threshold",
            "energy_before": float(np.sum(detail ** 2)), "energy_after": float(np.sum(output ** 2)),
            "zero_fraction_after": float(np.mean(output == 0)),
        })
    metadata["coefficients"] = coefficient_report
    reconstructed = pywt.waverec(modified, wavelet, mode="symmetric")[:raw.size].copy()
    if reconstructed.shape != raw.shape or not np.all(np.isfinite(reconstructed)):
        raise RuntimeError("Wavelet reconstruction produced an invalid pulse.")
    metadata["applied"] = True
    return reconstructed, metadata


def process_pulse(pulse: np.ndarray, fps: float, pulse_method: str, config: IPPGConfig) -> ProcessedPulse:
    """One shared raw -> optional DWT -> band-pass path; never denoise G-R."""

    validate_denoise_config(config)
    if pulse_method not in {"pos", "g-r", "ica"}:
        raise ValueError("Unknown pulse extraction method.")
    raw = np.asarray(pulse, dtype=float).copy()
    if raw.ndim != 1 or raw.size < 2 or not np.all(np.isfinite(raw)):
        raise ValueError("Pulse must contain at least two finite 1-D samples.")
    if pulse_method in {"pos", "ica"} and config.denoise == "dwt":
        reconstructed, metadata = wavelet_denoise(raw, config, fps)
    else:
        reconstructed = raw.copy()
        metadata = {"method": "none", "status": "disabled" if pulse_method in {"pos", "ica"} else "not_applicable",
                    "applied": False, "wavelet": None, "actual_level": None, "threshold": 0.0}
    if np.ptp(reconstructed) <= 1e-12:
        raise ValueError("Pulse is flat; insufficient signal to estimate BPM.")
    filtered = bandpass_filter(reconstructed, fps, config.forehead_low_hz,
                               config.forehead_high_hz, config.forehead_filter_order)
    if not np.all(np.isfinite(filtered)) or np.ptp(filtered) <= 1e-12:
        raise ValueError("Filtered pulse is invalid or flat; cannot estimate BPM.")
    return ProcessedPulse(raw, reconstructed, filtered, metadata)


def bandpass_filter(
    signal: np.ndarray,
    fps: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter."""

    nyquist_hz = fps / 2.0
    if not 0 < low_hz < high_hz < nyquist_hz:
        raise ValueError(
            "Band-pass limits must satisfy 0 < low < high < FPS/2; "
            f"received low={low_hz}, high={high_hz}, FPS={fps}."
        )

    normalized_band = [low_hz / nyquist_hz, high_hz / nyquist_hz]
    numerator, denominator = butter(order, normalized_band, btype="band")

    required_samples = 3 * max(len(numerator), len(denominator)) + 1
    if signal.size < required_samples:
        raise ValueError(
            f"The filter needs at least {required_samples} valid samples, "
            f"but only {signal.size} were extracted."
        )

    return filtfilt(numerator, denominator, signal)


def calculate_spectrum(
    time_seconds: np.ndarray,
    signal: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample, mean-center, and transform a signal to positive frequencies."""

    uniform_time = np.arange(time_seconds[0], time_seconds[-1], 1.0 / fps)
    if uniform_time.size < 2:
        raise ValueError("The extracted signal is too short for FFT analysis.")

    uniform_signal = np.interp(uniform_time, time_seconds, signal)
    centered_signal = uniform_signal - np.mean(uniform_signal)

    spectrum = np.fft.rfft(centered_signal)
    frequencies_hz = np.fft.rfftfreq(centered_signal.size, d=1.0 / fps)
    return frequencies_hz, spectrum


def estimate_heart_rate(
    frequencies_hz: np.ndarray,
    spectrum: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    """Return BPM from the strongest spectral component in a valid range."""

    if (frequencies_hz.shape != spectrum.shape or not np.all(np.isfinite(spectrum))
            or not np.all(np.isfinite(frequencies_hz))):
        raise ValueError("Invalid frequency/spectrum arrays for HR estimation.")
    valid_frequency = (frequencies_hz >= low_hz) & (frequencies_hz <= high_hz)
    if not np.any(valid_frequency):
        raise ValueError("The FFT contains no bins inside the heart-rate range.")
    if np.max(np.abs(spectrum[valid_frequency])) <= 1e-12:
        raise ValueError("No usable spectral peak in the heart-rate range; BPM is unavailable.")

    valid_indices = np.flatnonzero(valid_frequency)
    strongest_local_index = int(np.argmax(np.abs(spectrum[valid_frequency])))
    strongest_index = int(valid_indices[strongest_local_index])
    return float(frequencies_hz[strongest_index] * 60.0)


def validate_fusion_config(config: IPPGConfig) -> None:
    """Reject ambiguous or numerically unsafe ICA/fusion settings."""

    if config.fusion_domain not in {"spectral", "time"}:
        raise ValueError("fusion_domain must be 'spectral' or 'time'.")
    if not 0.0 <= config.fusion_min_correlation <= 1.0:
        raise ValueError("fusion_min_correlation must be between 0 and 1.")
    if config.fusion_bpm_tolerance < 0 or not np.isfinite(config.fusion_bpm_tolerance):
        raise ValueError("fusion_bpm_tolerance must be finite and non-negative.")
    if config.fusion_peak_half_width_hz <= 0 or not np.isfinite(config.fusion_peak_half_width_hz):
        raise ValueError("fusion_peak_half_width_hz must be positive and finite.")
    if config.fusion_epsilon <= 0 or not np.isfinite(config.fusion_epsilon):
        raise ValueError("fusion_epsilon must be positive and finite.")
    if config.ica_max_iter < 1:
        raise ValueError("ica_max_iter must be at least 1.")
    if config.ica_tolerance <= 0 or not np.isfinite(config.ica_tolerance):
        raise ValueError("ica_tolerance must be positive and finite.")


def calculate_ica_components(
    rgb: np.ndarray,
    config: IPPGConfig,
) -> tuple[np.ndarray, dict]:
    """Normalize RGB and return deterministic FastICA source candidates."""

    validate_fusion_config(config)
    if FastICA is None:
        raise RuntimeError(
            "POS+ICA requires scikit-learn. Install project dependencies with "
            "'pip install -r requirements-ippg.txt'."
        )
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] != 3 or rgb.shape[0] < 3:
        raise ValueError("ICA input must have shape (N, 3) with at least 3 samples.")
    if not np.all(np.isfinite(rgb)):
        raise ValueError("ICA input cannot contain NaN or infinity.")
    channel_means = np.mean(rgb, axis=0)
    if np.any(np.abs(channel_means) <= config.fusion_epsilon):
        raise ValueError("ICA encountered a zero RGB channel mean.")

    normalized = rgb / channel_means - 1.0
    rank = int(np.linalg.matrix_rank(normalized))
    component_count = min(3, rank)
    if component_count < 2:
        raise ValueError("Normalized RGB rank is below 2; ICA is unavailable.")

    model = FastICA(
        n_components=component_count,
        whiten="unit-variance",
        whiten_solver="svd",
        algorithm="parallel",
        fun="logcosh",
        random_state=config.ica_random_state,
        max_iter=config.ica_max_iter,
        tol=config.ica_tolerance,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        components = model.fit_transform(normalized)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("FastICA did not converge within the configured iterations.")
    if components.shape != (rgb.shape[0], component_count) or not np.all(np.isfinite(components)):
        raise RuntimeError("FastICA returned invalid source components.")
    metadata = {
        "engine": "sklearn.FastICA",
        "rgb_rank": rank,
        "component_count": component_count,
        "random_state": config.ica_random_state,
        "max_iter": config.ica_max_iter,
        "tolerance": config.ica_tolerance,
        "iterations": int(getattr(model, "n_iter_", 0)),
        "converged": True,
    }
    return components, metadata


def _zscore_pulse(signal: np.ndarray, epsilon: float) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    standard_deviation = float(np.std(signal))
    if signal.ndim != 1 or not np.all(np.isfinite(signal)) or standard_deviation <= epsilon:
        raise ValueError("Pulse cannot be Z-scored because it is invalid or flat.")
    return (signal - np.mean(signal)) / standard_deviation


def _normalized_inband_power(
    frequencies_hz: np.ndarray,
    power_spectrum: np.ndarray,
    config: IPPGConfig,
) -> np.ndarray:
    heart_band = (
        (frequencies_hz >= config.forehead_low_hz)
        & (frequencies_hz <= config.forehead_high_hz)
    )
    total = float(np.sum(power_spectrum[heart_band]))
    if not np.any(heart_band) or total <= config.fusion_epsilon:
        raise ValueError("Candidate has no usable power in the heart-rate band.")
    normalized = np.zeros_like(power_spectrum, dtype=float)
    normalized[heart_band] = power_spectrum[heart_band] / total
    return normalized


def build_pulse_candidate(
    method: str,
    processed: ProcessedPulse,
    time_seconds: np.ndarray,
    fps: float,
    config: IPPGConfig,
    component_index: int | None = None,
) -> PulseCandidateResult:
    """Calculate BPM and spectral concentration for one processed branch."""

    frequencies_hz, spectrum = calculate_spectrum(time_seconds, processed.filtered, fps)
    power = np.abs(spectrum) ** 2
    heart_band = (
        (frequencies_hz >= config.forehead_low_hz)
        & (frequencies_hz <= config.forehead_high_hz)
    )
    if not np.any(heart_band):
        raise ValueError("Candidate FFT has no heart-rate bins.")
    valid_indices = np.flatnonzero(heart_band)
    peak_index = int(valid_indices[int(np.argmax(power[heart_band]))])
    peak_frequency = float(frequencies_hz[peak_index])
    peak_band = heart_band & (
        np.abs(frequencies_hz - peak_frequency) <= config.fusion_peak_half_width_hz
    )
    total_power = float(np.sum(power[heart_band]))
    if total_power <= config.fusion_epsilon:
        raise ValueError("Candidate has no usable heart-rate power.")
    quality = float(np.sum(power[peak_band]) / total_power)
    bpm = estimate_heart_rate(
        frequencies_hz, power, config.forehead_low_hz, config.forehead_high_hz
    )
    return PulseCandidateResult(
        method=method,
        processed=processed,
        frequencies_hz=frequencies_hz,
        power_spectrum=power,
        heart_rate_bpm=bpm,
        quality=quality,
        component_index=component_index,
    )


def calculate_pos_ica_fusion(
    signals: ExtractedSignals,
    config: IPPGConfig,
) -> FusionResult:
    """Build POS and ICA independently, then form both requested fusions."""

    validate_fusion_config(config)
    if FastICA is None:
        raise RuntimeError(
            "POS+ICA requires scikit-learn. Install project dependencies with "
            "'pip install -r requirements-ippg.txt'."
        )

    errors: dict[str, str] = {}
    pos_candidate = None
    try:
        pos_raw = calculate_pos(
            signals.combined_rgb,
            signals.fps,
            config.pos_window_seconds,
            config.pos_epsilon,
        )
        pos_candidate = build_pulse_candidate(
            "pos",
            process_pulse(pos_raw, signals.fps, "pos", config),
            signals.time_seconds,
            signals.fps,
            config,
        )
    except (ValueError, RuntimeError) as error:
        errors["pos"] = str(error)

    ica_metadata: dict = {}
    ica_candidates: list[PulseCandidateResult] = []
    try:
        components, ica_metadata = calculate_ica_components(signals.combined_rgb, config)
        component_errors: dict[str, str] = {}
        for component_index in range(components.shape[1]):
            try:
                processed = process_pulse(
                    components[:, component_index], signals.fps, "ica", config
                )
                ica_candidates.append(
                    build_pulse_candidate(
                        "ica",
                        processed,
                        signals.time_seconds,
                        signals.fps,
                        config,
                        component_index,
                    )
                )
            except (ValueError, RuntimeError) as error:
                component_errors[str(component_index)] = str(error)
        if component_errors:
            ica_metadata["component_errors"] = component_errors
        if not ica_candidates:
            raise RuntimeError("No processed ICA component was usable.")
    except (ValueError, RuntimeError) as error:
        errors["ica"] = str(error)

    selected_ica = max(
        ica_candidates,
        key=lambda candidate: (candidate.quality, -int(candidate.component_index or 0)),
        default=None,
    )
    if pos_candidate is None and selected_ica is None:
        raise RuntimeError(f"Neither POS nor ICA produced a usable pulse: {errors}")

    correlation: float | None = None
    polarity = 1
    if pos_candidate is not None and selected_ica is not None:
        pos_z = _zscore_pulse(pos_candidate.processed.filtered, config.fusion_epsilon)
        ica_z = _zscore_pulse(selected_ica.processed.filtered, config.fusion_epsilon)
        correlation = float(np.corrcoef(pos_z, ica_z)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0
        polarity = -1 if correlation < 0 else 1
        quality_sum = pos_candidate.quality + selected_ica.quality
        if quality_sum <= config.fusion_epsilon:
            pos_weight = 0.5
            ica_weight = 0.5
        else:
            pos_weight = pos_candidate.quality / quality_sum
            ica_weight = selected_ica.quality / quality_sum
        low_agreement = (
            abs(correlation) < config.fusion_min_correlation
            and abs(pos_candidate.heart_rate_bpm - selected_ica.heart_rate_bpm)
            > config.fusion_bpm_tolerance
        )
        if low_agreement:
            if pos_candidate.quality >= selected_ica.quality:
                pos_weight, ica_weight, status = 1.0, 0.0, "fallback-pos-low-agreement"
            else:
                pos_weight, ica_weight, status = 0.0, 1.0, "fallback-ica-low-agreement"
        else:
            status = "fused"
    elif pos_candidate is not None:
        pos_z = _zscore_pulse(pos_candidate.processed.filtered, config.fusion_epsilon)
        ica_z = np.zeros_like(pos_z)
        pos_weight, ica_weight, status = 1.0, 0.0, "fallback-pos-ica-unavailable"
    else:
        assert selected_ica is not None
        ica_z = _zscore_pulse(selected_ica.processed.filtered, config.fusion_epsilon)
        pos_z = np.zeros_like(ica_z)
        pos_weight, ica_weight, status = 0.0, 1.0, "fallback-ica-pos-unavailable"

    combined_time = pos_weight * pos_z + ica_weight * polarity * ica_z
    time_fused = bandpass_filter(
        combined_time,
        signals.fps,
        config.forehead_low_hz,
        config.forehead_high_hz,
        config.forehead_filter_order,
    )
    time_fused = _zscore_pulse(time_fused, config.fusion_epsilon)
    frequencies_hz, time_spectrum = calculate_spectrum(
        signals.time_seconds, time_fused, signals.fps
    )
    time_bpm = estimate_heart_rate(
        frequencies_hz,
        time_spectrum,
        config.forehead_low_hz,
        config.forehead_high_hz,
    )

    spectral_score = np.zeros_like(frequencies_hz, dtype=float)
    if pos_candidate is not None:
        if not np.array_equal(frequencies_hz, pos_candidate.frequencies_hz):
            raise RuntimeError("POS and fusion FFT grids do not match.")
        spectral_score += pos_weight * _normalized_inband_power(
            frequencies_hz, pos_candidate.power_spectrum, config
        )
    if selected_ica is not None:
        if not np.array_equal(frequencies_hz, selected_ica.frequencies_hz):
            raise RuntimeError("ICA and fusion FFT grids do not match.")
        spectral_score += ica_weight * _normalized_inband_power(
            frequencies_hz, selected_ica.power_spectrum, config
        )
    spectral_bpm = estimate_heart_rate(
        frequencies_hz,
        spectral_score,
        config.forehead_low_hz,
        config.forehead_high_hz,
    )
    weights = {"pos": float(pos_weight), "ica": float(ica_weight)}
    metadata = {
        "status": status,
        "selected_domain": config.fusion_domain,
        "weights": weights,
        "correlation_before_polarity": correlation,
        "ica_polarity": polarity,
        "quality": {
            "pos": None if pos_candidate is None else pos_candidate.quality,
            "ica": None if selected_ica is None else selected_ica.quality,
        },
        "candidate_bpm": {
            "pos": None if pos_candidate is None else pos_candidate.heart_rate_bpm,
            "ica": None if selected_ica is None else selected_ica.heart_rate_bpm,
        },
        "selected_ica_component": None if selected_ica is None else selected_ica.component_index,
        "peak_half_width_hz": config.fusion_peak_half_width_hz,
        "minimum_correlation": config.fusion_min_correlation,
        "bpm_tolerance": config.fusion_bpm_tolerance,
        "ica": ica_metadata,
        "errors": errors,
        "uses_contact_ppg": False,
    }
    return FusionResult(
        pos_candidate=pos_candidate,
        ica_candidates=tuple(ica_candidates),
        selected_ica=selected_ica,
        time_fused_signal=time_fused,
        time_spectrum=time_spectrum,
        time_bpm=time_bpm,
        spectral_score=spectral_score,
        spectral_bpm=spectral_bpm,
        weights=weights,
        correlation=correlation,
        ica_polarity=polarity,
        status=status,
        metadata=metadata,
    )


def resolve_background_method(config: IPPGConfig, pulse_method: str) -> str:
    """Disable background by default; experiments require explicit opt-in."""

    if config.background_method is None:
        return "none"
    method = config.background_method.strip().lower()
    if method not in {"none", "legacy", "power"}:
        raise ValueError(
            f"Unknown background method {config.background_method!r}; "
            "choose 'none', 'legacy', or 'power'."
        )
    return method


def calculate_background_score(
    frequencies_hz: np.ndarray,
    pulse_spectrum: np.ndarray,
    background_spectrum: np.ndarray,
    config: IPPGConfig,
    method: str,
) -> np.ndarray:
    """Return a non-negative spectrum used to select the heart-rate peak."""

    if pulse_spectrum.shape != background_spectrum.shape:
        raise ValueError("Pulse and background spectra must have equal shapes.")
    if frequencies_hz.shape != pulse_spectrum.shape:
        raise ValueError("Frequency and spectrum arrays must have equal shapes.")

    pulse_power = np.abs(pulse_spectrum) ** 2
    if method == "none":
        return pulse_power
    if method == "legacy":
        legacy_spectrum = (
            pulse_spectrum
            - config.background_spectrum_scale * background_spectrum
        )
        return np.abs(legacy_spectrum) ** 2
    if method != "power":
        raise ValueError(f"Unsupported background method: {method}")
    if config.background_power_weight < 0:
        raise ValueError("Background power weight cannot be negative.")

    heart_band = (
        (frequencies_hz >= config.forehead_low_hz)
        & (frequencies_hz <= config.forehead_high_hz)
    )
    if not np.any(heart_band):
        raise ValueError("No FFT bins fall inside the configured heart-rate band.")

    background_power = np.abs(background_spectrum) ** 2
    pulse_energy = float(np.sum(pulse_power[heart_band]))
    background_energy = float(np.sum(background_power[heart_band]))
    epsilon = config.background_epsilon
    normalized_pulse_power = pulse_power / (pulse_energy + epsilon)
    normalized_background_power = background_power / (
        background_energy + epsilon
    )
    return np.maximum(
        normalized_pulse_power
        - config.background_power_weight * normalized_background_power,
        0.0,
    )


def analyze_signals(
    signals: ExtractedSignals,
    config: IPPGConfig,
    background_method: str | None = None,
) -> IPPGResult:
    """Filter extracted signals and estimate HR with one background method."""

    if background_method is None:
        method = resolve_background_method(config, signals.pulse_method)
    else:
        method = background_method.strip().lower()
        if method not in {"none", "legacy", "power"}:
            raise ValueError(
                "background_method must be 'none', 'legacy', or 'power'."
            )

    filtered_background = None
    if method != "none":
        if signals.background is None:
            raise ValueError(
                f"background_method={method!r} requires background extraction."
            )
        filtered_background = bandpass_filter(
            signals.background,
            signals.fps,
            config.background_low_hz,
            config.background_high_hz,
            config.background_filter_order,
        )

    if signals.pulse_method == "pos-ica":
        if method != "none":
            raise ValueError("POS+ICA fusion supports background_method='none' only.")
        fusion = calculate_pos_ica_fusion(signals, config)
        frequencies_hz = (
            fusion.pos_candidate.frequencies_hz
            if fusion.pos_candidate is not None
            else fusion.selected_ica.frequencies_hz
        )
        if config.fusion_domain == "spectral":
            compensated_spectrum = fusion.spectral_score
            heart_rate_bpm = fusion.spectral_bpm
            heart_rate_source = "spectral_fusion"
        else:
            compensated_spectrum = np.abs(fusion.time_spectrum) ** 2
            heart_rate_bpm = fusion.time_bpm
            heart_rate_source = "time_fusion"
        denoise_metadata = {
            "pos": None if fusion.pos_candidate is None else fusion.pos_candidate.processed.denoise_metadata,
            "ica": None if fusion.selected_ica is None else fusion.selected_ica.processed.denoise_metadata,
            "fusion": fusion.metadata,
        }
        return IPPGResult(
            signals=signals,
            filtered_pulse_signal=fusion.time_fused_signal,
            filtered_background=filtered_background,
            frequencies_hz=frequencies_hz,
            compensated_spectrum=compensated_spectrum,
            heart_rate_bpm=heart_rate_bpm,
            background_method=method,
            reconstructed_pulse_signal=fusion.time_fused_signal.copy(),
            denoise_metadata=denoise_metadata,
            heart_rate_source=heart_rate_source,
            fusion_result=fusion,
        )

    if signals.pulse_signal is None:
        raise ValueError("The selected pulse method did not produce a raw signal.")
    processed = process_pulse(signals.pulse_signal, signals.fps, signals.pulse_method, config)
    filtered_pulse_signal = processed.filtered

    frequencies_hz, pulse_spectrum = calculate_spectrum(
        signals.time_seconds,
        filtered_pulse_signal,
        signals.fps,
    )
    if method == "none":
        compensated_spectrum = np.abs(pulse_spectrum) ** 2
    else:
        assert filtered_background is not None
        background_frequencies_hz, background_spectrum = calculate_spectrum(
            signals.time_seconds,
            filtered_background,
            signals.fps,
        )
        if not np.array_equal(frequencies_hz, background_frequencies_hz):
            raise RuntimeError("Pulse and background FFT frequencies do not match.")
        compensated_spectrum = calculate_background_score(
            frequencies_hz,
            pulse_spectrum,
            background_spectrum,
            config,
            method,
        )
    heart_rate_bpm = estimate_heart_rate(
        frequencies_hz,
        compensated_spectrum,
        config.forehead_low_hz,
        config.forehead_high_hz,
    )

    return IPPGResult(
        signals=signals,
        filtered_pulse_signal=filtered_pulse_signal,
        filtered_background=filtered_background,
        frequencies_hz=frequencies_hz,
        compensated_spectrum=compensated_spectrum,
        heart_rate_bpm=heart_rate_bpm,
        background_method=method,
        reconstructed_pulse_signal=processed.reconstructed,
        denoise_metadata=processed.denoise_metadata,
        heart_rate_source="fft",
        fusion_result=None,
    )


def analyze_video(video_path: Path, config: IPPGConfig) -> IPPGResult:
    """Run the complete active iPPG v2 analysis pipeline."""

    signals = extract_video_signals(video_path, config)
    return analyze_signals(signals, config)


def plot_result(result: IPPGResult, *, show: bool = True):
    """Plot raw signals, filtered signals, and the compensated spectrum."""

    figure, all_axes = plt.subplots(4, 1, figsize=(12, 12))
    axes = [all_axes[0], all_axes[2], all_axes[3]]
    stage_axis = all_axes[1]
    if result.fusion_result is None:
        stage_axis.plot(result.signals.time_seconds, result.signals.pulse_signal, label="Raw pulse", alpha=0.7)
        stage_axis.plot(result.signals.time_seconds, result.reconstructed_pulse_signal, label="Reconstructed pulse", alpha=0.7)
        stage_title = f"Before band-pass: {result.denoise_metadata['status']}"
    else:
        fusion = result.fusion_result
        if fusion.pos_candidate is not None:
            stage_axis.plot(
                result.signals.time_seconds,
                fusion.pos_candidate.processed.reconstructed,
                label="POS after DWT",
                alpha=0.75,
            )
        if fusion.selected_ica is not None:
            stage_axis.plot(
                result.signals.time_seconds,
                fusion.ica_polarity * fusion.selected_ica.processed.reconstructed,
                label=f"ICA {fusion.selected_ica.component_index} after DWT",
                alpha=0.75,
            )
        stage_title = f"POS + ICA branches: {fusion.status}"
    stage_axis.set(title=stage_title, xlabel="Time (s)", ylabel="Amplitude")
    stage_axis.legend()

    rgb_colors = ("red", "green", "blue")
    for channel_index, (channel_name, channel_color) in enumerate(
        zip("RGB", rgb_colors)
    ):
        axes[0].plot(
            result.signals.time_seconds,
            result.signals.combined_rgb[:, channel_index],
            color=channel_color,
            label=f"Combined {channel_name}",
        )
    axes[0].set_title("Skin-only RGB from forehead and both cheeks")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Mean color value")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        result.signals.time_seconds,
        result.filtered_pulse_signal,
        color="blue",
        label=(
            "Time-fused POS + ICA"
            if result.fusion_result is not None
            else f"Filtered combined skin signal ({result.signals.pulse_method.upper()})"
        ),
    )
    if result.filtered_background is not None:
        axes[1].plot(
            result.signals.time_seconds,
            result.filtered_background,
            color="orange",
            alpha=0.7,
            label="Filtered background",
        )
    axes[1].set_title(
        "Band-pass filtered pulse"
        if result.filtered_background is None
        else "Band-pass filtered signals"
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(
        result.frequencies_hz,
        np.abs(result.compensated_spectrum),
        color="red",
        label=f"Heart-rate score ({result.heart_rate_source})",
    )
    axes[2].set_xlim(0.0, 4.0)
    axes[2].set_title(f"Estimated heart rate: {result.heart_rate_bpm:.1f} BPM")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Score")
    axes[2].grid(True)
    axes[2].legend()

    figure.tight_layout()
    if show:
        plt.show()
    return figure


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description="Estimate heart rate from a recorded face video."
    )
    parser.add_argument(
        "video",
        type=Path,
        nargs="?",
        help="Path to the input video (you will be prompted when omitted)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=34.333,
        help="Maximum seconds to analyze (default: 34.333)",
    )
    parser.add_argument(
        "--background-scale",
        type=float,
        default=1.5,
        help="Legacy complex-FFT subtraction scale (default: 1.5)",
    )
    parser.add_argument(
        "--background-method",
        choices=("none", "legacy", "power"),
        default=None,
        help=(
            "Optional background experiment; disabled by default for every pulse method"
        ),
    )
    parser.add_argument(
        "--background-power-weight",
        type=float,
        default=1.0,
        help="Penalty weight for normalized background power (default: 1.0)",
    )
    parser.add_argument(
        "--background-patch-size",
        type=int,
        default=32,
        help="Fixed stable background patch width/height (default: 32)",
    )
    parser.add_argument(
        "--background-calibration-frames",
        type=int,
        default=30,
        help="Frames used to select the stable background patch (default: 30)",
    )
    parser.add_argument(
        "--pulse-method",
        choices=("pos", "g-r", "pos-ica"),
        default="pos",
        help="RGB-to-pulse method: POS (default), POS+ICA fusion, or legacy G-R",
    )
    parser.add_argument(
        "--pos-window",
        type=float,
        default=1.6,
        help="POS sliding-window duration in seconds (default: 1.6)",
    )
    parser.add_argument(
        "--roi-method",
        choices=("quality", "legacy"),
        default="legacy",
        help="ROI aggregation: legacy pixel weighting (default) or experimental quality-selected patches",
    )
    parser.add_argument(
        "--roi-trim-fraction",
        type=float,
        default=0.10,
        help="Fraction trimmed from each RGB tail inside a patch (default: 0.10)",
    )
    parser.add_argument(
        "--roi-min-quality",
        type=float,
        default=0.05,
        help="Minimum accepted patch quality from 0 to 1 (default: 0.05)",
    )
    parser.add_argument(
        "--fusion-domain",
        choices=("spectral", "time"),
        default="spectral",
        help="POS+ICA BPM source: spectral (default) or time fusion",
    )
    parser.add_argument(
        "--fusion-min-correlation",
        type=float,
        default=0.2,
        help="Minimum absolute POS/ICA correlation before disagreement fallback (default: 0.2)",
    )
    parser.add_argument(
        "--fusion-bpm-tolerance",
        type=float,
        default=6.0,
        help="Maximum POS/ICA BPM disagreement before fallback (default: 6.0)",
    )
    parser.add_argument(
        "--ica-random-state",
        type=int,
        default=0,
        help="FastICA random seed for reproducible components (default: 0)",
    )
    parser.add_argument(
        "--cascade",
        type=Path,
        help="Optional path to a frontal-face Haar cascade XML file",
    )
    parser.add_argument(
        "--landmark-model",
        type=Path,
        help="Optional path to MediaPipe face_landmarker.task",
    )
    parser.add_argument(
        "--no-mediapipe",
        action="store_true",
        help="Use Haar rectangle ROIs instead of MediaPipe polygon ROIs",
    )
    parser.add_argument(
        "--no-adaptive-skin",
        action="store_true",
        help="Use the fixed YCrCb + HSV skin thresholds",
    )
    parser.add_argument(
        "--calibration-seconds",
        type=float,
        default=2.0,
        help="Seconds used to learn clip-specific skin color (default: 2.0)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Do not display OpenCV preview windows",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not display Matplotlib result plots",
    )
    add_wavelet_arguments(parser)
    return parser.parse_args()


def ask_for_video_path() -> Path:
    """Ask for a video path until the user enters an existing file."""

    while True:
        entered_path = input("Enter video filepath: ").strip()

        # Dragging a Windows file into a terminal can surround the path with
        # single or double quotes. Remove only those outer quote characters.
        entered_path = entered_path.strip('"\'')
        video_path = Path(entered_path).expanduser()

        if video_path.is_file():
            return video_path

        print(f"File not found: {video_path}")


def main() -> None:
    """Command-line entry point."""

    arguments = parse_arguments()
    video_path = arguments.video or ask_for_video_path()

    config = IPPGConfig(
        max_duration_seconds=arguments.duration,
        background_spectrum_scale=arguments.background_scale,
        background_method=arguments.background_method,
        background_power_weight=arguments.background_power_weight,
        background_patch_size=arguments.background_patch_size,
        background_calibration_frames=arguments.background_calibration_frames,
        face_cascade_path=arguments.cascade,
        prefer_mediapipe=not arguments.no_mediapipe,
        face_landmarker_model_path=arguments.landmark_model,
        adaptive_skin_detection=not arguments.no_adaptive_skin,
        skin_calibration_seconds=max(0.0, arguments.calibration_seconds),
        pulse_method=arguments.pulse_method,
        pos_window_seconds=arguments.pos_window,
        roi_method=arguments.roi_method,
        roi_trim_fraction=arguments.roi_trim_fraction,
        roi_min_quality=arguments.roi_min_quality,
        fusion_domain=arguments.fusion_domain,
        fusion_min_correlation=arguments.fusion_min_correlation,
        fusion_bpm_tolerance=arguments.fusion_bpm_tolerance,
        ica_random_state=arguments.ica_random_state,
        show_preview=not arguments.no_preview,
        **wavelet_config_from_arguments(arguments),
    )

    print(f"Analyzing: {video_path.resolve()}")
    result = analyze_video(video_path, config)
    print(f"Valid samples: {result.signals.time_seconds.size}")
    print(f"Video FPS: {result.signals.fps:.3f}")
    print(f"Pulse extraction method: {result.signals.pulse_method.upper()}")
    print(f"ROI aggregation: {result.signals.roi_method}")
    print(f"Background method: {result.background_method}")
    print(f"Denoising: {result.denoise_metadata}")
    print(f"Heart-rate source: {result.heart_rate_source}")
    if result.fusion_result is not None:
        print(f"Fusion: {result.fusion_result.metadata}")
    if result.signals.background_patch_box is not None:
        print(f"Background patch: {result.signals.background_patch_box}")
    else:
        print("Background calculation: disabled")
    print(f"Forehead RGB array shape: {result.signals.forehead_rgb.shape}")
    print(f"Cheeks RGB array shape: {result.signals.cheeks_rgb.shape}")
    print(f"Combined RGB array shape: {result.signals.combined_rgb.shape}")
    print(f"Estimated heart rate: {result.heart_rate_bpm:.1f} BPM")

    if not arguments.no_plot:
        plot_result(result)


if __name__ == "__main__":
    main()
