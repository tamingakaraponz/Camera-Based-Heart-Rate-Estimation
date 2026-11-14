"""Compare background compensation methods against synchronized contact PPG.

Example
-------
python benchmark_background_methods.py \
  --pair ../DATA/temp/11.avi ../DATAcppg/11.xlsx \
  --pair ../DATA/temp/12.avi ../DATAcppg/12.xlsx
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from ippg_v2_refactored import (
    IPPGConfig,
    add_wavelet_arguments, wavelet_config_from_arguments,
    analyze_signals,
    bandpass_filter,
    calculate_spectrum,
    estimate_heart_rate,
    extract_video_signals,
    resample_to_fps,
)


XML_NAMESPACE = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NAMESPACE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NAMESPACE = "http://schemas.openxmlformats.org/package/2006/relationships"


@dataclass(frozen=True)
class ContactPPGAnalysis:
    """Full-workbook contact PPG after resampling, band-pass and FFT."""

    time_seconds: np.ndarray
    raw_amplitude: np.ndarray
    filtered_amplitude: np.ndarray
    fps: float
    frequencies_hz: np.ndarray
    spectrum: np.ndarray
    heart_rate_bpm: float


def _column_index(cell_reference: str) -> int:
    """Convert the letters in an Excel cell reference to a zero-based index."""

    match = re.match(r"([A-Z]+)", cell_reference.upper())
    if match is None:
        raise ValueError(f"Invalid Excel cell reference: {cell_reference}")
    index = 0
    for letter in match.group(1):
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def read_first_sheet_xlsx(path: Path) -> list[list[object | None]]:
    """Read values from the first XLSX worksheet using the standard library."""

    with zipfile.ZipFile(path) as archive:
        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        first_sheet = workbook_root.find(f".//{{{XML_NAMESPACE}}}sheet")
        if first_sheet is None:
            raise ValueError(f"Workbook has no worksheets: {path}")
        relationship_id = first_sheet.attrib[f"{{{REL_NAMESPACE}}}id"]

        relationships = ET.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        target = None
        for relationship in relationships.findall(
            f"{{{PACKAGE_REL_NAMESPACE}}}Relationship"
        ):
            if relationship.attrib.get("Id") == relationship_id:
                target = relationship.attrib["Target"]
                break
        if target is None:
            raise ValueError(f"Cannot resolve the first worksheet in {path}")

        target_path = PurePosixPath(target)
        if target_path.is_absolute():
            sheet_path = str(target_path).lstrip("/")
        else:
            sheet_path = str(PurePosixPath("xl") / target_path)

        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall(f"{{{XML_NAMESPACE}}}si"):
                text_parts = [
                    node.text or ""
                    for node in item.iter(f"{{{XML_NAMESPACE}}}t")
                ]
                shared_strings.append("".join(text_parts))

        sheet_root = ET.fromstring(archive.read(sheet_path))
        output: list[list[object | None]] = []
        for row in sheet_root.findall(f".//{{{XML_NAMESPACE}}}row"):
            row_values: dict[int, object | None] = {}
            for cell in row.findall(f"{{{XML_NAMESPACE}}}c"):
                index = _column_index(cell.attrib["r"])
                value_node = cell.find(f"{{{XML_NAMESPACE}}}v")
                if value_node is None or value_node.text is None:
                    value: object | None = None
                elif cell.attrib.get("t") == "s":
                    value = shared_strings[int(value_node.text)]
                else:
                    try:
                        value = float(value_node.text)
                    except ValueError:
                        value = value_node.text
                row_values[index] = value

            if row_values:
                width = max(row_values) + 1
                output.append([row_values.get(index) for index in range(width)])
        return output


def load_contact_ppg(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load amplitude/timestamp columns and infer contact-PPG sampling rate."""

    if path.suffix.lower() != ".xlsx":
        raise ValueError("Contact PPG input must currently be an .xlsx workbook.")
    rows = read_first_sheet_xlsx(path)
    numeric_rows = [
        row
        for row in rows
        if len(row) >= 2
        and isinstance(row[0], (int, float))
        and isinstance(row[1], (int, float))
    ]
    if len(numeric_rows) < 2:
        raise ValueError(f"No amplitude/timestamp columns found in {path}")

    amplitude = np.asarray([row[0] for row in numeric_rows], dtype=float)
    timestamp = np.asarray([row[1] for row in numeric_rows], dtype=float)
    valid = np.isfinite(amplitude) & np.isfinite(timestamp)
    amplitude = amplitude[valid]
    timestamp = timestamp[valid]
    order = np.argsort(timestamp)
    amplitude = amplitude[order]
    timestamp = timestamp[order]

    unique_timestamp, unique_indices = np.unique(timestamp, return_index=True)
    amplitude = amplitude[unique_indices]
    timestamp = unique_timestamp
    differences = np.diff(timestamp)
    if differences.size == 0 or np.any(differences <= 0):
        raise ValueError(f"Invalid contact-PPG timestamps in {path}")
    fps = float(1.0 / np.median(differences))
    return timestamp, amplitude, fps


def analyze_contact_ppg(
    path: Path,
    low_hz: float,
    high_hz: float,
    filter_order: int = 3,
) -> ContactPPGAnalysis:
    """Calculate cPPG BPM from the entire workbook, independent of video ROI."""

    timestamp, amplitude, fps = load_contact_ppg(path)
    return analyze_contact_samples(
        timestamp, amplitude, fps, low_hz, high_hz, filter_order
    )


def analyze_contact_samples(
    timestamp: np.ndarray,
    amplitude: np.ndarray,
    fps: float,
    low_hz: float,
    high_hz: float,
    filter_order: int = 3,
) -> ContactPPGAnalysis:
    """Apply the canonical full-series cPPG resample, band-pass and FFT path."""

    timestamp = np.asarray(timestamp, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    if timestamp.ndim != 1 or amplitude.shape != timestamp.shape or timestamp.size < 2:
        raise ValueError("cPPG timestamps and amplitude must be matching 1-D arrays.")
    if not np.all(np.isfinite(timestamp)) or not np.all(np.isfinite(amplitude)):
        raise ValueError("cPPG timestamps and amplitude must be finite.")
    if np.any(np.diff(timestamp) <= 0):
        raise ValueError("cPPG timestamps must be strictly increasing.")
    timestamp = timestamp - timestamp[0]
    uniform_time, uniform_amplitude = resample_to_fps(timestamp, amplitude, fps)
    filtered = bandpass_filter(
        uniform_amplitude, fps, low_hz, high_hz, order=filter_order
    )
    frequencies_hz, spectrum = calculate_spectrum(uniform_time, filtered, fps)
    bpm = estimate_heart_rate(frequencies_hz, spectrum, low_hz, high_hz)
    return ContactPPGAnalysis(
        time_seconds=uniform_time,
        raw_amplitude=uniform_amplitude,
        filtered_amplitude=filtered,
        fps=fps,
        frequencies_hz=frequencies_hz,
        spectrum=spectrum,
        heart_rate_bpm=bpm,
    )


def estimate_contact_hr(
    path: Path,
    low_hz: float,
    high_hz: float,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> float:
    """Estimate reference HR over the same relative time window as iPPG."""

    if start_seconds is None and end_seconds is None:
        return analyze_contact_ppg(path, low_hz, high_hz).heart_rate_bpm

    timestamp, amplitude, fps = load_contact_ppg(path)
    timestamp = timestamp - timestamp[0]
    selected = np.ones(timestamp.size, dtype=bool)
    if start_seconds is not None:
        selected &= timestamp >= start_seconds
    if end_seconds is not None:
        selected &= timestamp <= end_seconds
    timestamp = timestamp[selected]
    amplitude = amplitude[selected]
    if timestamp.size < 2:
        raise ValueError(f"Contact-PPG window is empty for {path}")

    uniform_time, uniform_amplitude = resample_to_fps(timestamp, amplitude, fps)
    filtered = bandpass_filter(
        uniform_amplitude,
        fps,
        low_hz,
        high_hz,
        order=3,
    )
    frequencies, spectrum = calculate_spectrum(uniform_time, filtered, fps)
    return estimate_heart_rate(frequencies, spectrum, low_hz, high_hz)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark none/legacy/power background methods against contact PPG."
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        required=True,
        metavar=("VIDEO", "CONTACT_XLSX"),
        help="Video and corresponding contact-PPG workbook; repeat for more clips",
    )
    parser.add_argument("--duration", type=float, default=None)
    add_wavelet_arguments(parser)
    parser.add_argument("--pulse-method", choices=("pos", "g-r"), default="pos")
    parser.add_argument("--no-mediapipe", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional CSV output path")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = IPPGConfig(
        max_duration_seconds=arguments.duration,
        pulse_method=arguments.pulse_method,
        prefer_mediapipe=not arguments.no_mediapipe,
        show_preview=False,
        **wavelet_config_from_arguments(arguments),
    )
    methods = ("none", "legacy", "power")
    rows: list[dict[str, object]] = []

    for video_text, contact_text in arguments.pair:
        video_path = Path(video_text)
        contact_path = Path(contact_text)
        print(f"\nVideo: {video_path}")
        signals = extract_video_signals(video_path, config)
        contact_bpm = analyze_contact_ppg(
            contact_path,
            config.forehead_low_hz,
            config.forehead_high_hz,
        ).heart_rate_bpm
        print(f"Contact PPG: {contact_bpm:.2f} BPM ({contact_path})")

        for method in methods:
            result = analyze_signals(signals, config, method)
            error = result.heart_rate_bpm - contact_bpm
            rows.append(
                {
                    "video": str(video_path),
                    "contact_ppg": str(contact_path),
                    "pulse_method": signals.pulse_method,
                    "denoise": result.denoise_metadata["method"],
                    "wavelet_strategy": result.denoise_metadata.get("strategy", "none"),
                    "wavelet_threshold": result.denoise_metadata["threshold"],
                    "background_method": method,
                    "contact_bpm": contact_bpm,
                    "estimated_bpm": result.heart_rate_bpm,
                    "error_bpm": error,
                    "absolute_error_bpm": abs(error),
                }
            )
            print(
                f"  {method:>6}: {result.heart_rate_bpm:7.2f} BPM  "
                f"error={error:+7.2f}"
            )

    print("\nAggregate")
    for method in methods:
        errors = np.asarray(
            [row["error_bpm"] for row in rows if row["background_method"] == method],
            dtype=float,
        )
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))
        print(f"  {method:>6}: MAE={mae:.2f} BPM  RMSE={rmse:.2f} BPM")

    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("w", newline="", encoding="utf-8-sig") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
