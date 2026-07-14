import base64
import random
import shutil
import subprocess
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import os

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
FEEDER_AUTO_TC_TEMPLATE = "template_feeder_auto_tc.png"
FUJI_FONT_CANDIDATES = (
    os.path.join(TEMPLATE_DIR, "DejaVuSans.ttf"),
    "tahoma.ttf",
    "arial.ttf",
    "micross.ttf",
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "tahoma.ttf"),
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arial.ttf"),
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "micross.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
TABLE_FIRST_ROW_Y = 99
TABLE_ROW_STRIDE = 30
TABLE_VISIBLE_ROWS = 3
TABLE_NO_COL_SLICE = slice(18, 96)
TABLE_PARTS_NAME_COL_SLICE = slice(96, 205)
TABLE_FEEDER_TYPE_COL_SLICE = slice(360, 440)
KNOWN_FEEDER_TYPE_LABELS = (
    "Auto TC",
    "8mmTape",
)
KNOWN_CYLINDER_PART_NAME_SUFFIXES = (
    "FWD200X-170",
    "FWD0200XA-120",
    "FWD200XA-120",
    "KG3B_35_5F4Z",
    "MBDT200X-170",
    "MBT250XA-120",
    "MBDT250XA-120",
)
CIRCLE_PROFILE_BY_NAME = {
    "t0.2*11.0": ("t0.2*11.0", [(200, 320)], 240, "large"),
    "t0.3*2.0": ("t0.3*2.0", [(45, 60)], 54, "small"),
    "t0.15*10": ("t0.15*10", [(20, 35)], 28, "small"),
    "t0.15*10.0": ("t0.15*10.0", [(20, 35)], 28, "small"),
    "t0.2*10": ("t0.2*10", [(20, 35)], 28, "small"),
    "t0.2*10.0": ("t0.2*10.0", [(20, 35)], 28, "small"),
    "t0.15*9.0": ("t0.15*9.0", [(65, 90)], 75, "medium"),
    "t0.15*9": ("t0.15*9.0", [(65, 90)], 75, "medium"),
    "t0.2*9.0": ("t0.2*9.0", [(65, 90)], 75, "medium"),
    "t0.2*9": ("t0.2*9.0", [(65, 90)], 75, "medium"),
}
ALL_KNOWN_PART_NAMES = KNOWN_CYLINDER_PART_NAME_SUFFIXES + tuple(dict.fromkeys(
    list(CIRCLE_PROFILE_BY_NAME.keys()) +
    [key[:-2] for key in CIRCLE_PROFILE_BY_NAME if key.endswith(".0")]
))
BASE_FRAME_WIDTH = 800
BASE_FRAME_HEIGHT = 600
TARGET_FRAME_SIZE = (BASE_FRAME_WIDTH, BASE_FRAME_HEIGHT)
TEACH_BUTTON_REGION = (728, 571, 802, 647)
TEACH_ICON_REGION = (735, 570, 805, 615)


def normalize_fuji_frame_size(arr):
    if arr.shape[1] == BASE_FRAME_WIDTH and arr.shape[0] == BASE_FRAME_HEIGHT:
        return arr
    return cv2.resize(arr, TARGET_FRAME_SIZE, interpolation=cv2.INTER_AREA)


def get_frame_scale_x(arr):
    return arr.shape[1] / float(BASE_FRAME_WIDTH)


def get_frame_scale_y(arr):
    return arr.shape[0] / float(BASE_FRAME_HEIGHT)


def get_frame_uniform_scale(arr):
    return (get_frame_scale_x(arr) + get_frame_scale_y(arr)) / 2.0


def scale_frame_x(arr, value):
    return int(round(value * get_frame_scale_x(arr)))


def scale_frame_y(arr, value):
    return int(round(value * get_frame_scale_y(arr)))


def scale_frame_len(arr, value):
    return max(1, int(round(value * get_frame_uniform_scale(arr))))


def scale_frame_x_slice(arr, col_slice):
    return slice(scale_frame_x(arr, col_slice.start), scale_frame_x(arr, col_slice.stop))


def get_table_layout(arr):
    no_col_slice = scale_frame_x_slice(arr, TABLE_NO_COL_SLICE)
    if get_frame_uniform_scale(arr) >= 1.15:
        no_col_slice = slice(no_col_slice.start + 5, no_col_slice.stop + 5)
    return {
        "row_y_start": scale_frame_y(arr, TABLE_FIRST_ROW_Y),
        "row_height": scale_frame_len(arr, 29),
        "row_stride": scale_frame_len(arr, TABLE_ROW_STRIDE),
        "no_col_slice": no_col_slice,
        "no_col_width": TABLE_NO_COL_SLICE.stop - TABLE_NO_COL_SLICE.start,
        "parts_name_slice": scale_frame_x_slice(arr, TABLE_PARTS_NAME_COL_SLICE),
        "parts_name_width": TABLE_PARTS_NAME_COL_SLICE.stop - TABLE_PARTS_NAME_COL_SLICE.start,
        "feeder_slice": scale_frame_x_slice(arr, TABLE_FEEDER_TYPE_COL_SLICE),
        "feeder_width": TABLE_FEEDER_TYPE_COL_SLICE.stop - TABLE_FEEDER_TYPE_COL_SLICE.start,
    }


def normalize_text_crop(crop, target_width, target_height):
    if crop.size == 0:
        return crop
    if crop.shape[1] == target_width and crop.shape[0] == target_height:
        return crop
    return cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_AREA)


def infer_circle_session_suffix(template_match, active_parts_name, radius):
    explicit_suffix = template_match or active_parts_name
    resolved_profile = resolve_circle_profile_from_name(explicit_suffix)
    if resolved_profile is not None:
        if radius is None:
            return resolved_profile[0]
        _, _, nominal_radius, _ = resolved_profile
        tolerance = max(14.0, nominal_radius * 0.30)
        if abs(float(radius) - float(nominal_radius)) <= tolerance:
            return resolved_profile[0]

    if radius is None:
        return None

    canonical_profiles = {}
    for _, profile in CIRCLE_PROFILE_BY_NAME.items():
        canonical_profiles[profile[0]] = profile

    best_name = None
    best_gap = float("inf")
    for canonical_name, (_, _, nominal_radius, _) in canonical_profiles.items():
        gap = abs(float(radius) - float(nominal_radius))
        if gap < best_gap:
            best_gap = gap
            best_name = canonical_name

    if best_name is None:
        return None

    _, _, nominal_radius, _ = canonical_profiles[best_name]
    tolerance = max(14.0, nominal_radius * 0.30)
    return best_name if best_gap <= tolerance else None


def should_reject_partial_small_circle(shape_type, template_kind, visible_names):
    return (
        template_kind == "small" and
        visible_names in (("top",), ("bottom",), ("left",), ("right",), ("bottom", "left"), ("bottom", "right"))
    )


def should_try_uncertain_circle_fallback(active_row, active_feeder_auto_tc, auto_shape_type, active_parts_name=None):
    if active_parts_name in KNOWN_CYLINDER_PART_NAME_SUFFIXES:
        return False
    return (
        auto_shape_type is None and
        active_feeder_auto_tc is not False
    )


def has_large_washer_evidence(edge_mask):
    if edge_mask is None or edge_mask.size == 0:
        return False

    height, width = edge_mask.shape[:2]
    scale = height / 254.0
    band = max(1, int(round(18 * scale)))
    center_start = max(0, int(round(60 * scale)))
    center_stop = min(height, int(round(194 * scale)))
    side_threshold = max(18, int(round(45 * scale)))
    if center_stop <= center_start:
        return False

    central_rows = slice(center_start, center_stop)
    central_cols = slice(center_start, min(width, int(round(194 * (width / 254.0)))))
    side_scores = [
        int(np.count_nonzero(edge_mask[:band, central_cols])),
        int(np.count_nonzero(edge_mask[-band:, central_cols])),
        int(np.count_nonzero(edge_mask[central_rows, :band])),
        int(np.count_nonzero(edge_mask[central_rows, -band:])),
    ]
    return max(side_scores) >= side_threshold


def resolve_large_center_mode_from_quadrants(top_mean, bottom_mean, left_mean, right_mean):
    darkest_side = min(
        (
            ("top", float(top_mean)),
            ("bottom", float(bottom_mean)),
            ("left", float(left_mean)),
            ("right", float(right_mean)),
        ),
        key=lambda item: item[1],
    )[0]
    opposite_side = {
        "top": "bottom",
        "bottom": "top",
        "left": "right",
        "right": "left",
    }
    return opposite_side[darkest_side]


def scale_detector_result_to_original_frame(result, scale_x, scale_y):
    if not isinstance(result, dict):
        return result

    if scale_x == 1.0 and scale_y == 1.0:
        return result

    scaled = dict(result)

    def scale_point(point):
        if point is None:
            return None
        return [
            int(round(point[0] * scale_x)),
            int(round(point[1] * scale_y)),
        ]

    for key in ("top", "bottom", "left", "right", "center"):
        if key in scaled:
            scaled[key] = scale_point(scaled[key])

    if "radius" in scaled and scaled["radius"] is not None:
        scaled["radius"] = float(scaled["radius"]) * ((scale_x + scale_y) / 2.0)

    return scaled


def fit_circle_algebraic(x, y):
    """Fits a circle to points (x, y) using algebraic least squares."""
    A = np.column_stack((x, y, np.ones_like(x)))
    b = x**2 + y**2
    # Solve A * [a, b, c]^T = b


    w, residues, rank, s = np.linalg.lstsq(A, b, rcond=None)
    xc = w[0] / 2.0
    yc = w[1] / 2.0
    val = w[2] + xc**2 + yc**2
    R = np.sqrt(val) if val > 0 else 0.0
    return xc, yc, R

def circle_from_3_points(p1, p2, p3):
    """Calculates center and radius of a circle passing through 3 points."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-6:
        return None

    ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
    uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
    R = np.sqrt((x1 - ux)**2 + (y1 - uy)**2)
    return ux, uy, R


@lru_cache(maxsize=None)
def _load_fuji_font(size):


    for fp in FUJI_FONT_CANDIDATES:
        if not fp:
            continue
        try:
            return ImageFont.truetype(fp, size)
        except OSError:
            continue
    return ImageFont.load_default()

@lru_cache(maxsize=1)
def _digit_templates():
    font = _load_fuji_font(11)

    templates = {}
    for digit in "0123456789":
        img = Image.new("L", (16, 16), 0)
        draw = ImageDraw.Draw(img)
        draw.text((2, 2), digit, font=font, fill=255)
        arr_d = np.array(img)
        ys, xs = np.where(arr_d > 0)
        if len(xs) > 0:
            templates[digit] = arr_d[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return templates

def _match_char_against_templates(char_img, templates):
    best_char = None
    min_diff = float("inf")
    h_c, w_c = char_img.shape
    for char, temp in templates.items():
        temp_resized = cv2.resize(temp, (w_c, h_c), interpolation=cv2.INTER_NEAREST)
        diff = float(np.mean(cv2.absdiff(char_img, temp_resized))) / 255.0
        if diff < min_diff:
            min_diff = diff
            best_char = char
    return best_char, min_diff


def _match_digit(char_img):
    return _match_char_against_templates(char_img, _digit_templates())


@lru_cache(maxsize=1)
def _parts_name_templates():


    font = _load_fuji_font(11)

    templates = {}
    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.*/":
        img = Image.new("L", (20, 20), 0)
        draw = ImageDraw.Draw(img)
        draw.text((2, 2), char, font=font, fill=255)
        arr_c = np.array(img)
        ys, xs = np.where(arr_c > 0)
        if len(xs) > 0:
            templates[char] = arr_c[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return templates


def _match_parts_name_char(char_img):
    return _match_char_against_templates(char_img, _parts_name_templates())


def _render_parts_name_mask(text, size, x_offset, y_offset):


    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), text, font=_load_fuji_font(11), fill=255)
    return np.array(img)


@lru_cache(maxsize=3)
def _load_template_mask(filename):
    template_path = os.path.join(TEMPLATE_DIR, filename)
    if not os.path.exists(template_path):
        return None
    with Image.open(template_path) as template_img:
        return np.array(template_img.convert("L"))


def _foreground_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def score_parts_name_template_match(actual_mask, template_mask):
    actual_bbox = _foreground_bbox(actual_mask)
    template_bbox = _foreground_bbox(template_mask)
    if actual_bbox is None or template_bbox is None:
        return float("inf"), float("inf")

    ax0, ay0, ax1, ay1 = actual_bbox
    tx0, ty0, tx1, ty1 = template_bbox
    actual_crop = actual_mask[ay0:ay1, ax0:ax1]
    template_crop = template_mask[ty0:ty1, tx0:tx1]

    actual_h, actual_w = actual_crop.shape[:2]
    template_h, template_w = template_crop.shape[:2]
    compare_w = max(actual_w, template_w)
    compare_h = max(actual_h, template_h)
    actual_canvas = np.zeros((compare_h, compare_w), dtype=np.uint8)
    template_canvas = np.zeros((compare_h, compare_w), dtype=np.uint8)
    actual_canvas[:actual_h, :actual_w] = actual_crop
    template_canvas[:template_h, :template_w] = template_crop

    raw_diff = float(np.mean(cv2.absdiff(actual_canvas, template_canvas)))
    actual_fg = actual_canvas > 0
    template_fg = template_canvas > 0
    intersection = float(np.count_nonzero(actual_fg & template_fg))
    actual_count = float(np.count_nonzero(actual_fg))
    template_count = float(np.count_nonzero(template_fg))
    if actual_count == 0.0 or template_count == 0.0:
        return float("inf"), float("inf")
    precision = intersection / template_count
    recall = intersection / actual_count
    if precision + recall == 0.0:
        f1_penalty = 1.0
    else:
        f1_penalty = 1.0 - ((2.0 * precision * recall) / (precision + recall))
    width_penalty = abs(template_w - actual_w) / float(max(template_w, actual_w, 1))
    height_penalty = abs(template_h - actual_h) / float(max(template_h, actual_h, 1))
    density_actual = float(np.count_nonzero(actual_canvas)) / float(actual_canvas.size)
    density_template = float(np.count_nonzero(template_canvas)) / float(template_canvas.size)
    density_penalty = abs(density_actual - density_template)
    score = (
        raw_diff +
        (55.0 * f1_penalty) +
        (22.0 * width_penalty) +
        (12.0 * height_penalty) +
        (28.0 * density_penalty)
    )
    return raw_diff, score


def _generate_text_masks(gray):
    """Normalizes a grayscale row-cell text region with local contrast and
    produces several binarized masks from both dark-on-light and light-on-dark
    assumptions. Masks are lightly cleaned (borders trimmed, small horizontal
    morphology to reconnect broken strokes) before deduplication."""
    gray = gray.astype(np.uint8)
    background_level = float(np.median(gray))
    masks = []
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))

    def add_mask(mask):
        mask = mask.astype(np.uint8)
        mask[:2, :] = 0
        mask[-2:, :] = 0
        mask[:, :2] = 0
        mask[:, -2:] = 0
        # Light morphology to reconnect broken strokes without merging
        # adjacent characters (horizontal-only close).
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        foreground = int(np.count_nonzero(mask))
        if foreground < 8 or foreground > int(mask.size * 0.65):
            return
        if any(np.array_equal(mask, existing) for existing in masks):
            return
        masks.append(mask)

    for offset in (8, 14, 20, 28):
        dark_mask = np.zeros_like(gray, dtype=np.uint8)
        dark_mask[gray <= max(25, min(220, int(background_level - offset)))] = 255
        add_mask(dark_mask)

        light_mask = np.zeros_like(gray, dtype=np.uint8)
        light_mask[gray >= min(245, max(35, int(background_level + offset)))] = 255
        add_mask(light_mask)

    _, thresh_dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    add_mask(thresh_dark)
    _, thresh_light = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    add_mask(thresh_light)

    # Local-contrast (CLAHE) normalized masks for anti-aliased or faint text
    # on uneven row backgrounds (e.g. selected gray rows).
    h_img, w_img = gray.shape[:2]
    tile_h = max(2, (h_img + 7) // 8)
    tile_w = max(2, (w_img + 7) // 8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile_h, tile_w))
    equalized = clahe.apply(gray)
    eq_background = float(np.median(equalized))
    for offset in (8, 16, 24):
        dark_eq = np.zeros_like(gray, dtype=np.uint8)
        dark_eq[equalized <= max(25, min(220, int(eq_background - offset)))] = 255
        add_mask(dark_eq)

        light_eq = np.zeros_like(gray, dtype=np.uint8)
        light_eq[equalized >= min(245, max(35, int(eq_background + offset)))] = 255
        add_mask(light_eq)

    _, eq_thresh_dark = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    add_mask(eq_thresh_dark)
    _, eq_thresh_light = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    add_mask(eq_thresh_light)

    return masks


def _rank_text_candidates(gray, candidates, max_width_delta, max_height_delta, text_masks=None):
    """Scores every candidate against every generated text mask and returns
    (best_name, best_score, second_score) where the margin is measured between
    the best and second-best *different* candidate (not the same candidate from
    a different mask)."""
    candidates = tuple(candidates)
    best_per_candidate = {}

    for text_mask in text_masks if text_masks is not None else _generate_text_masks(gray):
        actual_bbox = _foreground_bbox(text_mask)
        if actual_bbox is None:
            continue

        ax0, ay0, ax1, ay1 = actual_bbox
        actual_crop = text_mask[ay0:ay1, ax0:ax1]
        actual_h, actual_w = actual_crop.shape[:2]
        if actual_w < 8 or actual_h < 6:
            continue

        for candidate in candidates:
            template_mask = _render_parts_name_mask(candidate, (max(actual_w + 12, 64), max(actual_h + 12, 24)), 4, 4)
            template_bbox = _foreground_bbox(template_mask)
            if template_bbox is None:
                continue

            tx0, ty0, tx1, ty1 = template_bbox
            template_crop = template_mask[ty0:ty1, tx0:tx1]
            template_h, template_w = template_crop.shape[:2]
            if abs(template_w - actual_w) > max_width_delta or abs(template_h - actual_h) > max_height_delta:
                continue

            compare_w = max(template_w, actual_w)
            compare_h = max(template_h, actual_h)
            actual_canvas = np.zeros((compare_h, compare_w), dtype=np.uint8)
            template_canvas = np.zeros((compare_h, compare_w), dtype=np.uint8)
            actual_canvas[:actual_h, :actual_w] = actual_crop
            template_canvas[:template_h, :template_w] = template_crop

            diff = float(np.mean(cv2.absdiff(actual_canvas, template_canvas))) / 255.0
            width_penalty = abs(template_w - actual_w) / float(max(template_w, actual_w, 1))
            height_penalty = abs(template_h - actual_h) / float(max(template_h, actual_h, 1))
            density_actual = float(np.count_nonzero(actual_canvas)) / float(actual_canvas.size)
            density_template = float(np.count_nonzero(template_canvas)) / float(template_canvas.size)
            density_penalty = abs(density_actual - density_template)
            score = diff + (0.90 * width_penalty) + (0.25 * height_penalty) + (0.35 * density_penalty)

            prev = best_per_candidate.get(candidate)
            if prev is None or score < prev:
                best_per_candidate[candidate] = score

    if not best_per_candidate:
        return None, float("inf"), float("inf")

    ranked = sorted(best_per_candidate.items(), key=lambda item: item[1])
    best_name, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else float("inf")
    return best_name, best_score, second_score


def _match_known_text(gray, candidates, max_width_delta=6, max_height_delta=4, max_score=0.60, text_masks=None):
    candidates = tuple(candidates)
    best_name, best_score, second_score = _rank_text_candidates(
        gray, candidates, max_width_delta, max_height_delta, text_masks
    )

    if (
        best_name is not None and
        best_score <= max_score and
        (
            len(candidates) == 1 or
            second_score == float("inf") or
            (second_score - best_score) >= 0.01
        )
    ):
        return best_name
    return None


def has_teach_button(arr):
    """Returns True/False when the Teach control is recognized, else None."""
    x0, y0, x1, y1 = TEACH_BUTTON_REGION
    ix0, iy0, ix1, iy1 = TEACH_ICON_REGION
    if arr.shape[1] < max(x1, ix1) or arr.shape[0] < max(y1, iy1):
        return False

    icon = arr[iy0:iy1, ix0:ix1].astype(np.int16)
    red, green, blue = icon[:, :, 0], icon[:, :, 1], icon[:, :, 2]
    if np.count_nonzero((blue > 120) & (blue - red > 40) & (blue - green > 20)) >= 250:
        return True

    # The lower half contains the label; excluding the icon keeps OCR focused
    # on the word rather than button artwork.
    button_crop = arr[y0:y1, x0:x1]
    label_crop = button_crop[button_crop.shape[0] // 2:, :]
    ocr_crop = normalize_text_crop(label_crop, 96, 48)
    return _ocr_teach_label(cv2.cvtColor(ocr_crop, cv2.COLOR_RGB2GRAY))


def _ocr_teach_label(gray):
    tesseract = shutil.which("tesseract")
    if not tesseract:
        return None

    enlarged = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, thresholded = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ok, encoded = cv2.imencode(".png", thresholded)
    if not ok:
        return None

    try:
        result = subprocess.run(
            [tesseract, "stdin", "stdout", "--psm", "7"],
            input=encoded.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    return "".join(ch for ch in result.stdout.decode(errors="ignore") if ch.isalpha()).lower() == "teach"


def _sanitize_parts_name(text):
    if not text:
        return None
    cleaned = "".join(ch for ch in str(text).strip() if ch.isalnum() or ch in "-_.*")
    if sum(ch.isalnum() for ch in cleaned) < 4:
        return None
    if not any(ch.isalpha() for ch in cleaned):
        return None
    return cleaned


def _normalize_cylinder_ocr_text(text):
    return "".join(
        {"O": "G", "J": "3", "D": "B", "I": "4"}.get(ch, ch)
        for ch in text.upper()
        if ch.isalnum()
    )


def normalize_circle_parts_name(parts_name):
    if not parts_name:
        return None

    cleaned = _sanitize_parts_name(parts_name)
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if "*" not in lowered or not lowered.startswith("t"):
        return None

    prefix, suffix = lowered.split("*", 1)
    suffix = suffix.rstrip(".")
    if not suffix:
        return None

    try:
        thickness = float(prefix[1:])
        suffix_value = float(suffix)
    except ValueError:
        return None

    thickness_str = ("%g" % thickness)
    suffix_str = ("%g" % suffix_value)
    if "." not in suffix_str:
        suffix_str = f"{suffix_str}.0"
    return f"t{thickness_str}*{suffix_str}"


def resolve_circle_profile_from_name(parts_name):
    direct_match = CIRCLE_PROFILE_BY_NAME.get(parts_name)
    if direct_match is not None:
        return direct_match

    normalized = normalize_circle_parts_name(parts_name)
    if not normalized:
        return None

    direct_match = CIRCLE_PROFILE_BY_NAME.get(normalized)
    if direct_match is not None:
        return direct_match

    try:
        suffix_value = float(normalized.split("*", 1)[1])
    except (IndexError, ValueError):
        return None

    if abs(suffix_value - 2.0) <= 0.05:
        return ("t0.3*2.0", [(45, 60)], 54, "small")
    if abs(suffix_value - 9.0) <= 0.05:
        return (normalized, [(65, 90)], 75, "medium")
    if abs(suffix_value - 10.0) <= 0.05:
        return (normalized, [(20, 35)], 28, "small")
    if abs(suffix_value - 11.0) <= 0.05:
        return ("t0.2*11.0", [(200, 320)], 240, "large")
    return None


def resolve_circle_profile_fallback(
    best_template,
    best_10mm_name,
    best_10mm_score,
    second_10mm_score,
    best_9mm_name=None,
    best_9mm_score=float("inf"),
    second_9mm_score=float("inf"),
):
    if (
        best_9mm_name is not None and
        best_9mm_score <= 0.85 and
        (
            second_9mm_score == float("inf") or
            (second_9mm_score - best_9mm_score) >= 0.004
        )
    ):
        return resolve_circle_profile_from_name(best_9mm_name)

    if (
        best_10mm_name is not None and
        best_10mm_score <= 0.68 and
        (
            second_10mm_score == float("inf") or
            (second_10mm_score - best_10mm_score) >= 0.01
        )
    ):
        return resolve_circle_profile_from_name(best_10mm_name)

    if best_template is not None and best_template[1] == "template_11_0.png" and best_template[6] <= 12.0:
        return resolve_circle_profile_from_name("t0.2*11.0")

    return None


def resolve_circle_profile_with_template_override(active_parts_name, best_template):
    parts_name_profile = resolve_circle_profile_from_name(active_parts_name)
    if best_template is not None and best_template[1] == "template_11_0.png" and best_template[7] <= 10.0:
        # Whole-name OCR can collapse 11.0 rows into nearby medium profiles.
        # When the dedicated 11.0 row mask matches, keep the large profile.
        return resolve_circle_profile_from_name("t0.2*11.0")
    return parts_name_profile


def _ocr_part_number_from_crop(no_crop):
    gray = cv2.cvtColor(no_crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    candidates = []

    for threshold_mode in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        _, thresh = cv2.threshold(gray, 0, 255, threshold_mode | cv2.THRESH_OTSU)
        thresh[:2, :] = 0
        thresh[-2:, :] = 0
        thresh[:, :2] = 0
        thresh[:, -2:] = 0

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        char_boxes = []
        for c in contours:
            x, y, w_c, h_c = cv2.boundingRect(c)
            if not (2 <= w_c <= 18 and 5 <= h_c <= 18):
                continue
            if w_c < 5 and h_c < 6:
                continue
            char_boxes.append((x, y, w_c, h_c))

        char_boxes.sort(key=lambda box: box[0])
        if len(char_boxes) >= 2:
            left_box = char_boxes[0]
            next_box = char_boxes[1]
            x0, y0, w_c, h_c = left_box
            if (
                x0 <= 4 and
                w_c <= 5 and
                h_c <= 5 and
                next_box[0] >= x0 + w_c
            ):
                char_boxes = char_boxes[1:]

        digits = []
        scores = []
        for x, y, w_c, h_c in char_boxes:
            char_img = thresh[y:y + h_c, x:x + w_c]
            digit, score = _match_digit(char_img)
            if digit is not None and score <= 0.55:
                digits.append(digit)
                scores.append(score)

        if digits:
            candidates.append(("".join(digits), float(np.mean(scores))))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-len(item[0]), item[1]))
    return candidates[0][0]


def _classify_single_digit_token(token):
    token_small = cv2.resize(token, (4, 9), interpolation=cv2.INTER_NEAREST)
    token_bin = (token_small > 0).astype(np.uint8)

    left_count = int(token_bin[:, 0].sum())
    right_count = int(token_bin[:, -1].sum())
    top_count = int(token_bin[0].sum())
    bottom_count = int(token_bin[-1].sum())
    middle_row = int(token_bin[4].sum())

    if left_count <= 1 and right_count >= 3 and top_count <= 2 and bottom_count <= 2:
        return "1"
    if left_count <= 2 and top_count >= 3 and bottom_count >= 3 and middle_row <= 1:
        return "2"
    if left_count >= 5 and bottom_count >= 3:
        return "6"
    if left_count <= 2 and top_count >= 3 and bottom_count >= 3:
        return "3"
    if left_count >= 2 and right_count >= 3 and middle_row >= 2 and bottom_count <= 2:
        return "4"
    if left_count >= 4 and right_count <= 3 and top_count >= 3 and bottom_count >= 3:
        return "5"
    if left_count >= 5 and right_count >= 3 and middle_row >= 3:
        return "8"
    if left_count >= 5 and right_count >= 3 and middle_row <= 2:
        return "0"
    if top_count >= 3 and bottom_count < 3 and right_count >= 4:
        return "7"
    if left_count <= 2 and right_count >= 4 and bottom_count >= 3:
        return "9"
    return None


def _extract_focus_boxes_and_thresh(no_crop):
    gray = cv2.cvtColor(no_crop, cv2.COLOR_RGB2GRAY)
    focus = gray[2:18, 64:77]
    _, thresh = cv2.threshold(focus, 205, 255, cv2.THRESH_BINARY)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh, 8)

    boxes = []
    for idx in range(1, num_labels):
        x, y, w_c, h_c, area = stats[idx]
        if area < 2:
            continue
        if y == 0 and h_c <= 2:
            continue
        boxes.append((x, y, w_c, h_c, area))
    return thresh, boxes


def _read_active_row_part_number(no_crop):
    thresh, boxes = _extract_focus_boxes_and_thresh(no_crop)
    if not boxes:
        return None

    boxes.sort(key=lambda box: box[0])
    if len(boxes) >= 2:
        left_box = boxes[0]
        next_box = boxes[1]
        left_area = left_box[4]
        remaining_area = sum(box[4] for box in boxes[1:])
        # Selected rows can show "*5" in the No. column. Ignore the tiny
        # leading asterisk component so it does not get classified as "9".
        if (
            left_box[0] <= 4 and
            left_box[2] <= 5 and
            left_box[3] <= 5 and
            left_area <= 12 and
            remaining_area >= left_area and
            next_box[0] >= left_box[0] + left_box[2]
        ):
            boxes = boxes[1:]

    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[0] + box[2] for box in boxes)
    y1 = max(box[1] + box[3] for box in boxes)
    token = thresh[y0:y1, x0:x1]

    if token.shape[1] <= 5:
        digit = _classify_single_digit_token(token)
        if digit is not None:
            return digit

    return _ocr_part_number_from_crop(no_crop)


def _has_selected_row_marker(no_crop):
    _, boxes = _extract_focus_boxes_and_thresh(no_crop)
    if len(boxes) < 2:
        return False

    boxes.sort(key=lambda box: box[0])
    left_box = boxes[0]
    next_box = boxes[1]
    left_area = left_box[4]
    remaining_area = sum(box[4] for box in boxes[1:])
    return (
        left_box[0] <= 4 and
        left_box[2] <= 5 and
        left_box[3] <= 5 and
        left_area <= 12 and
        remaining_area >= left_area and
        next_box[0] >= left_box[0] + left_box[2]
    )


def _resolve_visible_row_numbers(values, excluded_indices=None):
    excluded = set(excluded_indices or ())
    anchors = [
        value - idx
        for idx, value in enumerate(values)
        if idx not in excluded and value is not None and (value - idx) >= 1
    ]
    if not anchors:
        return values

    start = Counter(anchors).most_common(1)[0][0]
    return [start + idx for idx in range(len(values))]


def _get_visible_row_anchor_span(values, excluded_indices=None):
    excluded = set(excluded_indices or ())
    anchors = [
        value - idx
        for idx, value in enumerate(values)
        if idx not in excluded and value is not None and (value - idx) >= 1
    ]
    if len(anchors) < 2:
        return 0
    return max(anchors) - min(anchors)


def _iter_active_row_y_offsets(arr):
    if get_frame_uniform_scale(arr) < 1.15:
        return (0,)
    return (0, -6, 6, -12, 12, -18, 18, -24)


def _resolve_active_part_number(active_digits, visible_numbers, active_row):
    neighbor_anchor_span = _get_visible_row_anchor_span(visible_numbers, excluded_indices={active_row})
    if neighbor_anchor_span > 3:
        if active_digits and active_digits.isdigit() and 1 <= int(active_digits) <= 9:
            return active_digits
        if all(value is None or value >= 10 for value in visible_numbers):
            return str(active_row + 1)

    resolved_from_neighbors = _resolve_visible_row_numbers(visible_numbers, excluded_indices={active_row})
    if 0 <= active_row < len(resolved_from_neighbors) and resolved_from_neighbors[active_row] is not None:
        return str(resolved_from_neighbors[active_row])

    resolved_numbers = _resolve_visible_row_numbers(visible_numbers)
    if 0 <= active_row < len(resolved_numbers) and resolved_numbers[active_row] is not None:
        return str(resolved_numbers[active_row])
    if active_digits:
        return active_digits
    return None


def _score_active_part_number_candidate(candidate, active_digits, visible_numbers, active_row):
    if candidate is None or not str(candidate).isdigit():
        return None

    candidate_int = int(candidate)
    recognized_count = sum(value is not None for value in visible_numbers)
    neighbor_anchor_span = _get_visible_row_anchor_span(visible_numbers, excluded_indices={active_row})
    resolved_from_neighbors = _resolve_visible_row_numbers(visible_numbers, excluded_indices={active_row})
    neighbor_value = (
        resolved_from_neighbors[active_row]
        if 0 <= active_row < len(resolved_from_neighbors)
        else None
    )
    resolved_numbers = _resolve_visible_row_numbers(visible_numbers)
    resolved_value = (
        resolved_numbers[active_row]
        if 0 <= active_row < len(resolved_numbers)
        else None
    )

    active_is_single_digit = active_digits is not None and active_digits.isdigit() and len(active_digits) == 1
    active_value = int(active_digits) if active_digits and active_digits.isdigit() else None
    active_gap = abs(active_value - candidate_int) if active_value is not None else 0

    return (
        -recognized_count,
        0 if neighbor_value == candidate_int else 1,
        0 if resolved_value == candidate_int else 1,
        0 if active_is_single_digit else 1,
        neighbor_anchor_span,
        active_gap,
        abs(candidate_int - (active_row + 1)),
        candidate_int,
    )


def _extract_text_boxes(thresh, min_width=2, max_width=22, min_height=6, max_height=20, min_area=8):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w_c, h_c = cv2.boundingRect(contour)
        area = int(cv2.contourArea(contour))
        if not (min_width <= w_c <= max_width and min_height <= h_c <= max_height):
            continue
        if area < min_area:
            continue
        boxes.append((x, y, w_c, h_c))
    boxes.sort(key=lambda box: box[0])
    return boxes
def _ocr_parts_name_from_crop(name_crop):
    gray = cv2.cvtColor(name_crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    candidates = []

    for text_mask in _generate_text_masks(gray):
        char_box_sets = []

        contour_boxes = _extract_text_boxes(
            text_mask,
            min_width=1,
            max_width=20,
            min_height=4,
            max_height=18,
            min_area=3,
        )
        if contour_boxes:
            char_box_sets.append(contour_boxes)

        # Column projection merges obviously split character components that
        # contours separated into multiple boxes (e.g. broken strokes after
        # thresholding).
        col_foreground = np.count_nonzero(text_mask > 0, axis=0)
        projection_boxes = []
        start = None
        for idx, count in enumerate(col_foreground):
            if count >= 1 and start is None:
                start = idx
            elif count == 0 and start is not None:
                end = idx
                if end - start >= 1:
                    char_slice = text_mask[:, start:end]
                    row_foreground = np.count_nonzero(char_slice > 0, axis=1)
                    ys = np.where(row_foreground >= 1)[0]
                    if len(ys) > 0:
                        y0 = int(ys[0])
                        y1 = int(ys[-1]) + 1
                        projection_boxes.append((start, y0, end - start, y1 - y0))
                start = None
        if start is not None:
            end = len(col_foreground)
            if end - start >= 1:
                char_slice = text_mask[:, start:end]
                row_foreground = np.count_nonzero(char_slice > 0, axis=1)
                ys = np.where(row_foreground >= 1)[0]
                if len(ys) > 0:
                    y0 = int(ys[0])
                    y1 = int(ys[-1]) + 1
                    projection_boxes.append((start, y0, end - start, y1 - y0))
        if projection_boxes:
            char_box_sets.append(projection_boxes)

        for char_boxes in char_box_sets:
            chars = []
            scores = []
            for x, y, w_c, h_c in char_boxes:
                if not (1 <= w_c <= 20 and 4 <= h_c <= 18):
                    continue
                char_img = text_mask[y:y + h_c, x:x + w_c]
                char, score = _match_parts_name_char(char_img)
                if char is not None and score <= 0.72:
                    chars.append(char)
                    scores.append(score)

            if chars:
                candidates.append(("".join(chars), float(np.mean(scores))))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-len(item[0]), item[1]))
    return candidates[0][0]


def _read_parts_name_from_crop(name_crop):
    gray = cv2.cvtColor(name_crop, cv2.COLOR_RGB2GRAY)
    text_masks = _generate_text_masks(gray)
    best_name, best_score, second_score = _rank_text_candidates(gray, ALL_KNOWN_PART_NAMES, 6, 4, text_masks)
    best_9mm_name, best_9mm_score, second_9mm_score = _rank_text_candidates(
        gray,
        ("t0.15*9.0", "t0.15*9", "t0.2*9.0", "t0.2*9"),
        8, 5, text_masks,
    )
    matched_9mm_name = _match_known_text(
        gray,
        ("t0.15*9.0", "t0.15*9", "t0.2*9.0", "t0.2*9"),
        max_width_delta=18,
        max_height_delta=10,
        max_score=0.95, text_masks=text_masks,
    )
    if matched_9mm_name is not None:
        normalized = normalize_circle_parts_name(matched_9mm_name)
        if normalized is not None and normalized in CIRCLE_PROFILE_BY_NAME:
            return normalized, (0, best_9mm_score, second_9mm_score)
    if (
        best_9mm_name is not None and
        best_9mm_score <= 0.85 and
        (second_9mm_score == float("inf") or (second_9mm_score - best_9mm_score) >= 0.004) and
        (
            best_name is None or
            best_name not in ("t0.15*9.0", "t0.15*9", "t0.2*9.0", "t0.2*9") or
            best_9mm_score <= best_score + 0.20
        )
    ):
        normalized = normalize_circle_parts_name(best_9mm_name)
        if normalized is not None and normalized in CIRCLE_PROFILE_BY_NAME:
            return normalized, (1, best_9mm_score, second_9mm_score)
    if (
        best_name is not None and
        best_score <= 0.60 and
        (second_score == float("inf") or (second_score - best_score) >= 0.01)
    ):
        if best_name in KNOWN_CYLINDER_PART_NAME_SUFFIXES:
            return best_name, (2, best_score, second_score)
        normalized = normalize_circle_parts_name(best_name)
        if normalized is not None and normalized in CIRCLE_PROFILE_BY_NAME:
            return normalized, (2, best_score, second_score)
        return best_name, (3, best_score, second_score)

    raw_name = _sanitize_parts_name(_ocr_parts_name_from_crop(name_crop))
    if raw_name is not None:
        normalized_raw_name = "".join(ch for ch in raw_name.upper() if ch.isalnum())
        cylinder_raw_name = _normalize_cylinder_ocr_text(raw_name)
        if len(normalized_raw_name) >= 6:
            best_ratio, second_ratio = 0.0, 0.0
            best_cylinder_name = None
            for candidate in KNOWN_CYLINDER_PART_NAME_SUFFIXES:
                normalized_candidate = "".join(ch for ch in candidate.upper() if ch.isalnum())
                ratio = max(
                    SequenceMatcher(None, normalized_raw_name, normalized_candidate).ratio(),
                    SequenceMatcher(None, cylinder_raw_name, normalized_candidate).ratio(),
                )
                if ratio > best_ratio:
                    second_ratio = best_ratio
                    best_ratio = ratio
                    best_cylinder_name = candidate
                elif ratio > second_ratio:
                    second_ratio = ratio
            if best_ratio >= 0.55 and (best_ratio - second_ratio) >= 0.18:
                return best_cylinder_name, (4, -best_ratio, -second_ratio)
        normalized = normalize_circle_parts_name(raw_name)
        if normalized is not None and normalized in CIRCLE_PROFILE_BY_NAME:
            return normalized, (5, 0.0, 0.0)
        return raw_name, (6, 0.0, 0.0)

    return None, None


def _probe_active_row_parts_name(arr, active_row, row_y_start, row_height, row_stride, parts_name_slice, parts_name_width):
    best_name = None
    best_score = None
    best_offset = None

    for y_offset in _iter_active_row_y_offsets(arr):
        active_y_start = row_y_start + active_row * row_stride + y_offset
        active_y_end = active_y_start + row_height
        if active_y_start < 0 or active_y_end > arr.shape[0]:
            continue

        name_crop = normalize_text_crop(
            arr[active_y_start:active_y_end, parts_name_slice],
            parts_name_width,
            29,
        )
        candidate_name, candidate_score = _read_parts_name_from_crop(name_crop)
        if candidate_name is None or candidate_score is None:
            continue
        if best_score is None or candidate_score < best_score:
            best_name = candidate_name
            best_score = candidate_score
            best_offset = y_offset

    return best_name, best_score, best_offset


def ocr_part_number(arr, active_row, preferred_offset=None):
    """Recognizes the part number for the active visible row from the No. column."""
    try:
        if active_row < 0:
            return None

        table_layout = get_table_layout(arr)
        row_y_start = table_layout["row_y_start"]
        row_height = table_layout["row_height"]
        row_stride = table_layout["row_stride"]
        no_col_slice = table_layout["no_col_slice"]
        no_col_width = table_layout["no_col_width"]
        parts_name_slice = table_layout["parts_name_slice"]
        parts_name_width = table_layout["parts_name_width"]

        if preferred_offset is None:
            _, _, preferred_offset = _probe_active_row_parts_name(
                arr,
                active_row,
                row_y_start,
                row_height,
                row_stride,
                parts_name_slice,
                parts_name_width,
            )

        active_y_start = row_y_start + active_row * row_stride
        active_y_end = active_y_start + row_height
        if get_frame_uniform_scale(arr) >= 1.15 and 0 <= active_y_start and active_y_end <= arr.shape[0]:
            active_crop = normalize_text_crop(
                arr[active_y_start:active_y_end, no_col_slice],
                no_col_width,
                29,
            )
            active_digits = _read_active_row_part_number(active_crop)
            if active_digits and active_digits.isdigit() and 1 <= int(active_digits) <= 9:
                return active_digits

        y_offsets = list(_iter_active_row_y_offsets(arr))
        if preferred_offset in y_offsets:
            y_offsets.remove(preferred_offset)
            y_offsets.insert(0, preferred_offset)

        best_candidate = None
        best_score = None
        prefer_active_digit = get_frame_uniform_scale(arr) >= 1.15

        for y_offset in y_offsets:
            active_digits = None
            active_y_start = row_y_start + active_row * row_stride + y_offset
            active_y_end = active_y_start + row_height
            if active_y_start < 0 or active_y_end > arr.shape[0]:
                continue

            active_crop = normalize_text_crop(
                arr[active_y_start:active_y_end, no_col_slice],
                no_col_width,
                29,
            )
            active_digits = _read_active_row_part_number(active_crop)

            visible_numbers = []
            for idx in range(TABLE_VISIBLE_ROWS):
                y_start = row_y_start + idx * row_stride + y_offset
                y_end = y_start + row_height
                if y_start < 0 or y_end > arr.shape[0]:
                    visible_numbers.append(None)
                    continue

                no_crop = normalize_text_crop(
                    arr[y_start:y_end, no_col_slice],
                    no_col_width,
                    29,
                )
                if idx == active_row and active_digits is not None:
                    digits = active_digits
                else:
                    digits = _ocr_part_number_from_crop(no_crop)
                visible_numbers.append(int(digits) if digits and digits.isdigit() else None)

            if prefer_active_digit and active_digits and active_digits.isdigit() and 1 <= int(active_digits) <= 9:
                candidate = active_digits
            else:
                candidate = _resolve_active_part_number(active_digits, visible_numbers, active_row)
            score = _score_active_part_number_candidate(candidate, active_digits, visible_numbers, active_row)
            if score is not None and (best_score is None or score < best_score):
                best_score = score
                best_candidate = candidate

        if best_candidate is not None:
            return best_candidate
    except Exception as e:
        print(f"OCR failed: {e}")
    return None


def ocr_parts_name(arr, active_row):
    """Recognizes the parts name for the active visible row."""
    try:
        if active_row < 0:
            return None

        table_layout = get_table_layout(arr)
        row_y_start = table_layout["row_y_start"]
        row_height = table_layout["row_height"]
        row_stride = table_layout["row_stride"]
        parts_name_slice = table_layout["parts_name_slice"]
        parts_name_width = table_layout["parts_name_width"]
        best_name, _, _ = _probe_active_row_parts_name(
            arr,
            active_row,
            row_y_start,
            row_height,
            row_stride,
            parts_name_slice,
            parts_name_width,
        )
        return best_name
    except Exception as e:
        print(f"Parts name OCR failed: {e}")
    return None

def detect_active_feeder_auto_tc(arr, active_row, preferred_offset=None):
    """Tri-state detector for the "Auto TC" feeder label on the active row.

    Returns True when the label is confidently "Auto TC", False only when a
    known non-Auto-TC label is confidently recognized, and None when feeder
    text is unreadable or ambiguous so downstream routing does not force an
    incorrect negative."""
    if active_row < 0:
        return None

    table_layout = get_table_layout(arr)
    row_y_start = table_layout["row_y_start"]
    row_height = table_layout["row_height"]
    row_stride = table_layout["row_stride"]
    feeder_slice = table_layout["feeder_slice"]
    feeder_width = table_layout["feeder_width"]
    parts_name_slice = table_layout["parts_name_slice"]
    parts_name_width = table_layout["parts_name_width"]

    def classify_feeder_crop(feeder_gray):
        for window in (
            feeder_gray[4:24, :56],
            feeder_gray[4:24, :64],
            feeder_gray[2:26, :56],
            feeder_gray[2:26, 4:68],
        ):
            if window.shape[0] < 6 or window.shape[1] < 8:
                continue
            matched_label = _match_known_text(
                window,
                KNOWN_FEEDER_TYPE_LABELS,
                max_width_delta=16,
                max_height_delta=8,
                max_score=0.50,
            )
            if matched_label is not None:
                return matched_label == "Auto TC"

        feeder_mask = np.zeros_like(feeder_gray, dtype=np.uint8)
        feeder_mask[feeder_gray > 220] = 255
        feeder_template = _load_template_mask(FEEDER_AUTO_TC_TEMPLATE)
        if feeder_template is None:
            return None
        if feeder_mask.shape != feeder_template.shape:
            return None

        mask_bbox = _foreground_bbox(feeder_mask)
        template_bbox = _foreground_bbox(feeder_template)
        if mask_bbox is None or template_bbox is None:
            return None

        mx0, my0, mx1, my1 = mask_bbox
        tx0, ty0, tx1, ty1 = template_bbox
        if (
            abs(mx0 - tx0) > 4 or abs(my0 - ty0) > 3 or
            abs(mx1 - tx1) > 4 or abs(my1 - ty1) > 3
        ):
            return None

        template_fg = int(np.count_nonzero(feeder_template))
        mask_fg = int(np.count_nonzero(feeder_mask))
        if abs(mask_fg - template_fg) > max(40, int(template_fg * 0.15)):
            return None

        overlap = int(np.count_nonzero(cv2.bitwise_and(feeder_mask, feeder_template)))
        if overlap >= int(template_fg * 0.85):
            return True

        return None

    if preferred_offset is None:
        _, _, preferred_offset = _probe_active_row_parts_name(
            arr,
            active_row,
            row_y_start,
            row_height,
            row_stride,
            parts_name_slice,
            parts_name_width,
        )
    y_offsets = list(_iter_active_row_y_offsets(arr))
    if preferred_offset in y_offsets:
        y_offsets.remove(preferred_offset)
        y_offsets.insert(0, preferred_offset)

    for y_offset in y_offsets:
        y_start_row = row_y_start + active_row * row_stride + y_offset
        y_end_row = y_start_row + row_height
        if (
            y_start_row < 0 or
            y_end_row > arr.shape[0] or
            feeder_slice.stop > arr.shape[1]
        ):
            continue

        feeder_crop = normalize_text_crop(
            arr[y_start_row:y_end_row, feeder_slice],
            feeder_width,
            29,
        )
        feeder_gray = cv2.cvtColor(feeder_crop, cv2.COLOR_RGB2GRAY)
        feeder_match = classify_feeder_crop(feeder_gray)
        if feeder_match is not None:
            return feeder_match

    return None


def resolve_auto_shape_type(active_row, active_feeder_auto_tc, template_match, active_parts_name=None):
    if active_row == -1:
        return None
    if active_parts_name in KNOWN_CYLINDER_PART_NAME_SUFFIXES:
        return "cylinder"
    if resolve_circle_profile_from_name(active_parts_name) is not None:
        return "circle"
    if active_feeder_auto_tc is True:
        return "circle"
    template_profile = resolve_circle_profile_from_name(template_match)
    if template_profile is not None and template_profile[3] in ("small", "medium"):
        return "circle"
    if active_feeder_auto_tc is False:
        return None
    if template_profile is not None:
        return "circle"
    return None

# Targeted bezel filter: the camera-window bezel has a specific, known
# signature measured on this machine. Only candidates matching this exact
# signature are rejected as the bezel. Real parts that are merely offset
# from the crosshairs (but not at the bezel's position) are NOT affected.
BEZEL_CENTER_DX = -9.5
BEZEL_CENTER_DY = 14.9
BEZEL_CENTER_TOL = 1.5
DEFAULT_CROSS_X = 147
DEFAULT_CROSS_Y = 341

def is_bezel(xc, yc, R, xc_cross, yc_cross):
    if abs(xc - (xc_cross + BEZEL_CENTER_DX)) > BEZEL_CENTER_TOL:
        return False
    if abs(yc - (yc_cross + BEZEL_CENTER_DY)) > BEZEL_CENTER_TOL:
        return False
    return True

def find_crosshair_center(arr):
    r = arr[:, :, 0].astype(int)
    g = arr[:, :, 1].astype(int)
    b = arr[:, :, 2].astype(int)

    # JPEG screenshots often wash out the cyan guide lines. Detect the long
    # horizontal/vertical guide lines by color dominance rather than strict RGB.
    cyan_mask = (g > 120) & (b > 120) & ((g - r) > 20) & ((b - r) > 20)
    search_limit_x = min(arr.shape[1], scale_frame_x(arr, 300))
    search_start_y = min(arr.shape[0], scale_frame_y(arr, 200))
    cyan_mask[:, search_limit_x:] = False
    cyan_mask[:search_start_y, :] = False

    x_hist = cyan_mask.sum(axis=0)
    y_hist = cyan_mask.sum(axis=1)
    if x_hist.max() >= 20 and y_hist.max() >= 20:
        return int(np.argmax(x_hist)), int(np.argmax(y_hist))

    y_cyan, x_cyan = np.where((r < 100) & (g > 150) & (b > 150))
    valid = x_cyan < search_limit_x
    x_cyan = x_cyan[valid]
    y_cyan = y_cyan[valid]
    valid_rows = y_cyan >= search_start_y
    x_cyan = x_cyan[valid_rows]
    y_cyan = y_cyan[valid_rows]
    if len(x_cyan) > 50:
        return int(Counter(x_cyan).most_common(1)[0][0]), int(Counter(y_cyan).most_common(1)[0][0])

    return scale_frame_x(arr, DEFAULT_CROSS_X), scale_frame_y(arr, DEFAULT_CROSS_Y)

def get_circle_anchor_error(xc, yc, R, xc_cross, yc_cross, anchor_mode=None):
    anchor_points = {
        "top": (xc, yc - R),
        "bottom": (xc, yc + R),
        "left": (xc - R, yc),
        "right": (xc + R, yc),
    }
    if anchor_mode in anchor_points:
        px, py = anchor_points[anchor_mode]
        return (px - xc_cross) ** 2 + (py - yc_cross) ** 2
    return min((px - xc_cross) ** 2 + (py - yc_cross) ** 2 for px, py in anchor_points.values())

# Deleted get_small_circle_anchor_error function to clean up dead code

def angular_distance(angles, target_angle):
    diff = np.abs(angles - target_angle)
    return np.minimum(diff, 2.0 * np.pi - diff)

def get_circle_support_points(x, y, xc, yc, R, tolerance=None):
    if len(x) == 0:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    if tolerance is None:
        tolerance = max(3.0, min(10.0, R * 0.12))
    residuals = np.abs(np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - R)
    support_mask = residuals <= tolerance
    return x[support_mask], y[support_mask], residuals[support_mask]

def get_circle_support_metrics(x, y, xc, yc, R, inliers=None, tolerance=None):
    if inliers is not None:
        x = x[inliers]
        y = y[inliers]
    support_x, support_y, support_residuals = get_circle_support_points(x, y, xc, yc, R, tolerance=tolerance)
    if len(support_x) == 0:
        return {
            "support_count": 0,
            "coverage_bins": 0,
            "cardinal_support": 0,
            "span_x_ratio": 0.0,
            "span_y_ratio": 0.0,
            "min_span_ratio": 0.0,
            "residual_mean": float("inf"),
            "residual_p90": float("inf"),
            "sector_peak": 0,
            "angular_balance": 1.0,
        }

    angles = np.mod(np.arctan2(support_y - yc, support_x - xc), 2.0 * np.pi)
    coverage_hist = np.bincount(
        np.floor(angles / (2.0 * np.pi / 24.0)).astype(int),
        minlength=24,
    )
    sector_hist = np.bincount(
        np.floor(angles / (2.0 * np.pi / 8.0)).astype(int),
        minlength=8,
    )

    support_threshold = max(3, min(8, int(0.05 * len(support_x))))
    cardinal_angles = (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0)
    cardinal_support = 0
    for target_angle in cardinal_angles:
        if np.count_nonzero(angular_distance(angles, target_angle) <= np.deg2rad(35.0)) >= support_threshold:
            cardinal_support += 1

    occupied_bins = coverage_hist > 0
    if np.any(occupied_bins):
        doubled_bins = np.concatenate((occupied_bins, occupied_bins))
        current_gap = 0
        largest_gap_bins = 0
        for is_occupied in doubled_bins:
            if is_occupied:
                current_gap = 0
            else:
                current_gap += 1
                largest_gap_bins = max(largest_gap_bins, current_gap)
        largest_gap_bins = min(largest_gap_bins, len(occupied_bins))
    else:
        largest_gap_bins = len(coverage_hist)

    diameter = max(2.0 * R, 1.0)
    span_x_ratio = float((np.max(support_x) - np.min(support_x)) / diameter)
    span_y_ratio = float((np.max(support_y) - np.min(support_y)) / diameter)
    min_span_ratio = min(span_x_ratio, span_y_ratio)
    angular_balance = float(np.hypot(np.mean(np.cos(angles)), np.mean(np.sin(angles))))

    return {
        "support_count": int(len(support_x)),
        "coverage_bins": int(np.count_nonzero(coverage_hist)),
        "cardinal_support": cardinal_support,
        "largest_gap_bins": int(largest_gap_bins),
        "span_x_ratio": span_x_ratio,
        "span_y_ratio": span_y_ratio,
        "min_span_ratio": min_span_ratio,
        "residual_mean": float(np.mean(support_residuals)),
        "residual_p90": float(np.percentile(support_residuals, 90)),
        "sector_peak": int(sector_hist.max()) if len(sector_hist) else 0,
        "angular_balance": angular_balance,
    }

# Deleted resolve_sector_support_pixel function to clean up dead code

def extract_connected_components(mask, min_pixels=18):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    components = []
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_pixels:
            ys, xs = np.where(labels == label)
            components.append((ys, xs))
    return components

def fit_circle_ransac_unanchored(
    x,
    y,
    allowed_ranges,
    max_iterations=800,
    threshold=3.0,
    min_inliers=8,
):
    """Fits a circle from candidate edge points without crosshair anchoring."""
    if len(x) < 3:
        return None

    rng = random.Random(137)
    points = np.column_stack((x, y))
    candidates = []
    sample_count = max_iterations + len(points) // 4

    for _ in range(sample_count):
        idx = rng.sample(range(len(points)), 3)
        circle = circle_from_3_points(points[idx[0]], points[idx[1]], points[idx[2]])
        if circle is None:
            continue

        xc, yc, R = circle
        if not any(low <= R <= high for low, high in allowed_ranges):
            continue

        dists = np.abs(np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - R)
        inliers = np.where(dists <= threshold)[0]
        if len(inliers) < min_inliers:
            continue

        metrics = get_circle_support_metrics(x, y, xc, yc, R, inliers)
        candidates.append((
            -len(inliers),
            metrics["angular_balance"],
            -metrics["coverage_bins"],
            metrics["residual_mean"],
            metrics["residual_p90"],
            (xc, yc, R),
            inliers,
        ))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[:5])
    _, _, _, _, _, circle, inliers = candidates[0]
    inlier_x = x[inliers]
    inlier_y = y[inliers]
    try:
        refined = fit_circle_algebraic(inlier_x, inlier_y)
        if any(low <= refined[2] <= high for low, high in allowed_ranges):
            return refined
    except Exception:
        pass
    return circle

def fit_circle_ransac(
    x,
    y,
    xc_cross=DEFAULT_CROSS_X,
    yc_cross=DEFAULT_CROSS_Y,
    max_iterations=500,
    threshold=2.0,
    allowed_ranges=None,
    center_mode=None,
    anchor_mode=None,
    preferred_radius=None,
):
    """Fits a circle using RANSAC to handle noise and select the best circle close to crosshair center."""
    if allowed_ranges is None:
        allowed_ranges = [(20, 65), (200, 320)]

    is_large_target = allowed_ranges is not None and any(high > 150 for low, high in allowed_ranges)
    if is_large_target:
        max_iterations = max(max_iterations, 2000)

    num_points = len(x)
    if num_points < 3:
        return None

    # Use a fixed seed for determinism
    rng = random.Random(42)

    candidates = []
    points = np.column_stack((x, y))

    for _ in range(max_iterations):
        # Sample 3 random points
        idx = rng.sample(range(num_points), 3)
        p1, p2, p3 = points[idx[0]], points[idx[1]], points[idx[2]]

        circle = circle_from_3_points(p1, p2, p3)
        if circle is None:
            continue

        xc, yc, R = circle
        # Filter for circle candidate with radius within strict target ranges
        in_range = any(low <= R <= high for low, high in allowed_ranges)
        if not in_range:
            continue

        # Enforce quadrant-based center and boundary constraints for large washer parts
        if R > 120 and center_mode is not None:
            R_final = R + 5.5 if R > 120 else R
            tol = 15.0
            if center_mode == "top":
                if yc <= yc_cross or abs(yc - R_final - yc_cross) > tol or abs(xc - xc_cross) > tol:
                    continue
            elif center_mode == "bottom":
                if yc >= yc_cross or abs(yc + R_final - yc_cross) > tol or abs(xc - xc_cross) > tol:
                    continue
            elif center_mode == "left":
                if xc <= xc_cross or abs(xc - R_final - xc_cross) > tol or abs(yc - yc_cross) > tol:
                    continue
            elif center_mode == "right":
                if xc >= xc_cross or abs(xc + R_final - xc_cross) > tol or abs(yc - yc_cross) > tol:
                    continue
        else:
            # Small parts are taught from one of the fitted circle boundaries.
            # Keep candidates whose geometry can plausibly touch the guide-line center.
            if get_circle_anchor_error(xc, yc, R, xc_cross, yc_cross, anchor_mode=anchor_mode) > 75.0 ** 2:
                continue

        dists = np.abs(np.sqrt((x - xc)**2 + (y - yc)**2) - R)
        inliers = np.where(dists < threshold)[0]

        if len(inliers) >= 15:
            candidates.append((len(inliers), (xc, yc, R), inliers))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    max_inliers = candidates[0][0]

    def get_deviation(circle_tuple):
        xc_c, yc_c, R_c = circle_tuple
        R_f = R_c + 5.5 if R_c > 120 else R_c
        radius_bias = abs(R_c - preferred_radius) if preferred_radius is not None else 0.0
        if R_c > 120 and center_mode is not None:
            if center_mode == "top":
                return (xc_c - xc_cross)**2 + (yc_c - R_f - yc_cross)**2, radius_bias
            elif center_mode == "bottom":
                return (xc_c - xc_cross)**2 + (yc_c + R_f - yc_cross)**2, radius_bias
            elif center_mode == "left":
                return (xc_c - R_f - xc_cross)**2 + (yc_c - yc_cross)**2, radius_bias
            elif center_mode == "right":
                return (xc_c + R_f - xc_cross)**2 + (yc_c - yc_cross)**2, radius_bias
        return get_circle_anchor_error(
            xc_c,
            yc_c,
            R_c,
            xc_cross,
            yc_cross,
            anchor_mode=anchor_mode,
        ), radius_bias

    # Nearby clutter can produce a stronger arc than the taught target. Keep a
    # looser inlier floor for small parts, then rank candidates by anchor
    # alignment before falling back to support strength.
    min_inlier_ratio = 0.7 if is_large_target else 0.4
    candidate_floor = max(15, int(np.ceil(min_inlier_ratio * max_inliers)))
    top_candidates = [c for c in candidates if c[0] >= candidate_floor]

    def get_candidate_sort_key(candidate):
        inlier_count, circle_tuple, inliers = candidate
        anchor_error, radius_bias = get_deviation(circle_tuple)
        metrics = get_circle_support_metrics(
            x,
            y,
            circle_tuple[0],
            circle_tuple[1],
            circle_tuple[2],
            inliers,
        )
        if is_large_target:
            return (
                anchor_error,
                radius_bias,
                metrics["residual_mean"],
                metrics["residual_p90"],
                -metrics["coverage_bins"],
                -metrics["support_count"],
                -inlier_count,
            )
        sector_ratio = metrics["sector_peak"] / max(metrics["support_count"], 1)
        return (
            radius_bias,
            -metrics["cardinal_support"],
            metrics["largest_gap_bins"],
            -metrics["coverage_bins"],
            -metrics["min_span_ratio"],
            sector_ratio,
            metrics["residual_mean"],
            metrics["residual_p90"],
            anchor_error,
            -metrics["support_count"],
            -inlier_count,
        )

    top_candidates.sort(key=get_candidate_sort_key)
    best_candidate = (top_candidates[0][1], top_candidates[0][2])

    # Re-fit algebraically using only the inliers of the selected candidate
    circle_geom, inliers_indices = best_candidate
    inlier_x = x[inliers_indices]
    inlier_y = y[inliers_indices]
    try:
        xc, yc, R = fit_circle_algebraic(inlier_x, inlier_y)
        if any(low <= R <= high for low, high in allowed_ranges):
            # Verify refitted circle also satisfies strict constraints
            if R > 120 and center_mode is not None:
                R_final = R + 5.5 if R > 120 else R
                tol_edge = 2.0
                tol_center = 1.5
                if center_mode == "top":
                    if yc < yc_cross and abs(yc + R_final - yc_cross) <= tol_edge and abs(xc - xc_cross) <= tol_center:
                        return xc, yc, R
                elif center_mode == "bottom":
                    if yc > yc_cross and abs(yc - R_final - yc_cross) <= tol_edge and abs(xc - xc_cross) <= tol_center:
                        return xc, yc, R
                elif center_mode == "left":
                    if xc < xc_cross and abs(xc + R_final - xc_cross) <= tol_edge and abs(yc - yc_cross) <= tol_center:
                        return xc, yc, R
                elif center_mode == "right":
                    if xc > xc_cross and abs(xc - R_final - xc_cross) <= tol_edge and abs(yc - yc_cross) <= tol_center:
                        return xc, yc, R
                return circle_geom
            else:
                if get_circle_anchor_error(xc, yc, R, xc_cross, yc_cross, anchor_mode=anchor_mode) > 75.0 ** 2:
                    return circle_geom
                return xc, yc, R
            return xc, yc, R
        return circle_geom
    except:
        return circle_geom

def run_inference(image_b64, shape_type=None):
    """
    Decodes the base64 image, crops to camera box, filters gray pixels,
    detects the circle/cylinder boundary pixels, and returns results.

    Args:
        image_b64: Base64-encoded screenshot string.
        shape_type: Optional hint from the caller.
            'cylinder' – force cylinder detection mode (both-cap or partial left-cap).
            None / 'circle' – auto-detect (default circle/RANSAC path).
    """
    try:
        def make_no_detect_result():
            return {
                "top": None,
                "bottom": None,
                "left": None,
                "right": None,
            }

        def make_idle_result():
            result = make_no_detect_result()
            result["screen_state"] = "idle"
            return result

        def build_session_id(session_suffix_override=None, prefer_cylinder=False):
            if prefer_cylinder and active_parts_name in KNOWN_CYLINDER_PART_NAME_SUFFIXES:
                session_suffix = active_parts_name
            elif prefer_cylinder and active_feeder_auto_tc is False:
                session_suffix = "KG3B_35_5F4Z"
            else:
                session_suffix = session_suffix_override or template_match or active_parts_name
            if not session_suffix:
                return None
            if session_suffix not in KNOWN_CYLINDER_PART_NAME_SUFFIXES:
                resolved_profile = resolve_circle_profile_from_name(session_suffix)
                if resolved_profile is not None:
                    session_suffix = resolved_profile[0]
            if session_suffix in {"t0.2*9.0", "t0.2*9"}:
                session_suffix = "t0.15*9.0"
            part_num = ocr_part_number(table_arr, active_row, preferred_parts_name_offset)
            if part_num:
                return f"session_{part_num}_{session_suffix}"
            return f"session_{active_row + 1}_{session_suffix}" if active_row != -1 else None

        def finalize_result(result, is_cylinder=False):
            if result is None:
                return None
            if (
                is_cylinder and
                isinstance(result, dict) and
                "session_id" not in result and
                any(result.get(name) is not None for name in ("top", "bottom", "left", "right"))
            ):
                session_id = build_session_id(prefer_cylinder=True)
                if session_id is not None:
                    result = dict(result)
                    result["session_id"] = session_id
            if (
                not is_cylinder and
                isinstance(result, dict) and
                "session_id" not in result and
                any(result.get(name) is not None for name in ("top", "bottom", "left", "right"))
            ):
                session_suffix_override = None
                if (
                    all(result.get(name) is None for name in ("top", "bottom", "left", "right")) and
                    best_template is not None and
                    best_template[1] == "template_11_0.png" and
                    best_template[6] <= 21.0
                ):
                    session_suffix_override = "t0.2*11.0"
                session_id = build_session_id(session_suffix_override)
                if session_id is not None:
                    result = dict(result)
                    result["session_id"] = session_id
            return scale_detector_result_to_original_frame(result, frame_scale_x, frame_scale_y)

        def has_visible_cylinder_points(result):
            if not isinstance(result, dict):
                return False
            return result.get("left") is not None or result.get("right") is not None

        # Decode base64 image
        if "base64," in image_b64:
            image_b64 = image_b64.split("base64,")[1]

        img_data = base64.b64decode(image_b64)
        img = cv2.imdecode(np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Failed to decode image data."}
        if img.shape[1] != 1024 or img.shape[0] != 768:
            return {"error": f"Unsupported image resolution: {(img.shape[1], img.shape[0])}. Expected 1024x768."}
        original_arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if has_teach_button(original_arr) is False:
            return make_idle_result()

        table_arr = original_arr
        frame_scale_x = get_frame_scale_x(original_arr)
        frame_scale_y = get_frame_scale_y(original_arr)
        arr = normalize_fuji_frame_size(original_arr)

        # 1. Dynamically locate the cyan guide-line center
        xc_cross, yc_cross = find_crosshair_center(arr)

        # Camera crop box boundaries centered at crosshairs
        crop_half = scale_frame_len(arr, 127)
        x_start = xc_cross - crop_half
        x_end = xc_cross + crop_half
        y_start = yc_cross - crop_half
        y_end = yc_cross + crop_half

        # Clamp bounds to handle edge of screen safely while keeping crop size 254x254
        if x_start < 0:
            x_end -= x_start
            x_start = 0
        if y_start < 0:
            y_end -= y_start
            y_start = 0
        if x_end > arr.shape[1]:
            x_start -= (x_end - arr.shape[1])
            x_end = arr.shape[1]
        if y_end > arr.shape[0]:
            y_start -= (y_end - arr.shape[0])
            y_end = arr.shape[0]

        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(arr.shape[1], x_end)
        y_end = min(arr.shape[0], y_end)

        # Check boundary size
        if (x_end - x_start) < 200 or (y_end - y_start) < 200:
            raise ValueError(f"Image dimensions {arr.shape[:2]} are smaller than camera box")

        dialog_title = arr[250:310, 300:500].astype(np.int16)
        if dialog_title.size and np.count_nonzero(
            (dialog_title[:, :, 2] - dialog_title[:, :, 0] > 25) &
            (dialog_title[:, :, 2] - dialog_title[:, :, 1] > 10)
        ) > 800:
            return finalize_result(make_no_detect_result())

        crop = arr[y_start:y_end, x_start:x_end]

        template_match = None
        template_ranges = None
        template_nominal = None
        template_kind = None
        active_feeder_auto_tc = None
        active_parts_name = None
        preferred_parts_name_offset = None
        best_template = None

        # Prefer the visible "*" marker in the No. column when present;
        # otherwise fall back to the gray selection background heuristic.
        table_layout = get_table_layout(table_arr)
        row_y_start = table_layout["row_y_start"]
        row_height = table_layout["row_height"]
        row_stride = table_layout["row_stride"]
        parts_name_slice = table_layout["parts_name_slice"]
        parts_name_width = table_layout["parts_name_width"]
        no_col_slice = table_layout["no_col_slice"]
        no_col_width = table_layout["no_col_width"]
        row_probe_left = min(table_arr.shape[1], scale_frame_x(table_arr, 360))
        row_probe_right = min(table_arr.shape[1], scale_frame_x(table_arr, 620))
        row_averages = []
        row_midtone_scores = []
        row_medians = []
        row_selection_deltas = []
        marker_rows = []
        for idx in range(TABLE_VISIBLE_ROWS):
            y_start_row = row_y_start + idx * row_stride
            y_end_row = y_start_row + row_height
            if y_end_row <= table_arr.shape[0]:
                row_slice = table_arr[y_start_row:y_end_row, row_probe_left:row_probe_right]
                row_gray = np.mean(row_slice, axis=2)
                row_mean = float(np.mean(row_gray))
                row_median = float(np.median(row_gray))
                row_averages.append(row_mean)
                row_medians.append(row_median)
                row_midtone_scores.append(float(np.mean((row_gray >= 165.0) & (row_gray <= 225.0))))
                row_selection_deltas.append(abs(row_median - 194.0))
                marker_crop = normalize_text_crop(
                    table_arr[y_start_row:y_end_row, no_col_slice],
                    no_col_width,
                    29,
                )
                marker_rows.append(_has_selected_row_marker(marker_crop))
            else:
                row_averages.append(255.0)
                row_medians.append(255.0)
                row_midtone_scores.append(0.0)
                row_selection_deltas.append(float("inf"))
                marker_rows.append(False)

        active_row = -1
        if marker_rows.count(True) == 1:
            active_row = marker_rows.index(True)
        elif row_averages:
            best_idx = min(
                range(len(row_selection_deltas)),
                key=lambda idx: (row_selection_deltas[idx], -row_midtone_scores[idx], row_averages[idx]),
            )
            if row_selection_deltas[best_idx] <= 28.0 and row_midtone_scores[best_idx] >= 0.04:
                active_row = best_idx
            else:
                min_idx = int(np.argmin(row_averages))
                if row_averages[min_idx] < 210:
                    active_row = min_idx
        if active_row != -1:
            y_start_row = row_y_start + active_row * row_stride
            y_end_row = y_start_row + row_height
            if y_end_row <= table_arr.shape[0] and parts_name_slice.stop <= table_arr.shape[1]:
                # Normalize the selected-row parts-name crop back to the base
                # template size so the fixed row templates still match scaled
                # screenshots such as 1024x768 captures.
                crop_rgb = normalize_text_crop(
                    table_arr[y_start_row:y_end_row, parts_name_slice],
                    parts_name_width,
                    29,
                )
                crop_gray = np.mean(crop_rgb, axis=2)
                mask_active = (crop_gray > 220).astype(np.uint8) * 255
                best_match = None
                best_template = None
                template_configs = [
                    ("t0.2*11.0", "template_11_0.png", [(200, 320)], 240, "large", 3.0),
                    ("t0.15*9.0", None, [(65, 90)], 75, "medium", 18.0),
                    ("t0.3*2.0", "template_2_0.png", [(45, 60)], 54, "small", 15.0),
                ]

                for part_name, filename, ranges, r_nom, kind, max_diff in template_configs:
                    if filename is None:
                        template_arr = _render_parts_name_mask(part_name, (mask_active.shape[1], mask_active.shape[0]), 0, 0)
                    else:
                        template_arr = _load_template_mask(filename)
                        if template_arr is None:
                            continue

                    if mask_active.shape == template_arr.shape:
                        diff, match_score = score_parts_name_template_match(mask_active, template_arr)
                        if best_template is None or match_score < best_template[7]:
                            best_template = (part_name, filename, ranges, r_nom, kind, max_diff, diff, match_score)
                        if diff <= max_diff and (best_match is None or match_score < best_match[5]):
                            best_match = (part_name, ranges, r_nom, kind, diff, match_score)

                weak_small_template = None
                weak_medium_template = None
                camera_crop_mean_hint = float(np.mean(crop))
                camera_crop_std_hint = float(np.std(crop))
                if best_template is not None and best_template[4] != "small":
                    for part_name, filename, ranges, r_nom, kind, _max_diff in template_configs:
                        if kind == "medium" and weak_medium_template is None:
                            weak_medium_template = (part_name, ranges, r_nom, kind)
                        if kind != "small":
                            continue
                        if filename is None:
                            template_arr = _render_parts_name_mask(part_name, (mask_active.shape[1], mask_active.shape[0]), 0, 0)
                        else:
                            template_arr = _load_template_mask(filename)
                            if template_arr is None:
                                continue
                        if mask_active.shape != template_arr.shape:
                            continue
                        diff, match_score = score_parts_name_template_match(mask_active, template_arr)
                        if match_score <= best_template[7] + 3.0 and match_score <= 65.0:
                            weak_small_template = (part_name, ranges, r_nom, kind)
                            break

                text_masks = _generate_text_masks(crop_gray)
                best_9mm_name, best_9mm_score, second_9mm_score = _rank_text_candidates(
                    crop_gray,
                    ("t0.15*9.0", "t0.15*9", "t0.2*9.0", "t0.2*9"),
                    8,
                    5, text_masks,
                )
                best_2mm_name, best_2mm_score, second_2mm_score = _rank_text_candidates(
                    crop_gray,
                    ("t0.3*2.0",),
                    8,
                    5, text_masks,
                )
                best_10mm_name, best_10mm_score, second_10mm_score = _rank_text_candidates(
                    crop_gray,
                    ("t0.15*10", "t0.15*10.0", "t0.2*10", "t0.2*10.0"),
                    8,
                    5, text_masks,
                )
                fallback_profile = resolve_circle_profile_fallback(
                    best_template,
                    best_10mm_name,
                    best_10mm_score,
                    second_10mm_score,
                    best_9mm_name,
                    best_9mm_score,
                    second_9mm_score,
                )
                active_parts_name, _, preferred_parts_name_offset = _probe_active_row_parts_name(
                    table_arr,
                    active_row,
                    row_y_start,
                    row_height,
                    row_stride,
                    parts_name_slice,
                    parts_name_width,
                )
                parts_name_profile = resolve_circle_profile_with_template_override(active_parts_name, best_template)
                if (
                    best_template is not None and
                    best_template[4] == "large" and
                    best_template[7] <= 21.0 and
                    (parts_name_profile is None or parts_name_profile[3] == "medium")
                ):
                    template_match, template_ranges, template_nominal, template_kind = (
                        best_template[0],
                        best_template[2],
                        best_template[3],
                        best_template[4],
                    )
                elif (
                    weak_small_template is not None and
                    (
                        parts_name_profile is None or
                        parts_name_profile[3] == "medium"
                    )
                ):
                    if best_template is not None and best_template[4] == "large":
                        if camera_crop_mean_hint >= 230.0 or (
                            camera_crop_mean_hint < 190.0 and
                            camera_crop_std_hint < 35.0
                        ):
                            template_match, template_ranges, template_nominal, template_kind = weak_small_template
                        elif camera_crop_mean_hint >= 190.0 and weak_medium_template is not None:
                            template_match, template_ranges, template_nominal, template_kind = weak_medium_template
                        else:
                            template_match, template_ranges, template_nominal, template_kind = (
                                best_template[0],
                                best_template[2],
                                best_template[3],
                                best_template[4],
                            )
                    else:
                        template_match, template_ranges, template_nominal, template_kind = weak_small_template
                elif (
                    (parts_name_profile is None or parts_name_profile[3] == "medium") and
                    best_template is not None and
                    best_template[4] == "small" and
                    best_template[7] <= 65.0
                ):
                    template_match, template_ranges, template_nominal, template_kind = (
                        best_template[0],
                        best_template[2],
                        best_template[3],
                        best_template[4],
                    )
                elif (
                    parts_name_profile is not None and
                    parts_name_profile[3] == "medium" and
                    best_2mm_name is not None and
                    best_2mm_score <= 0.65 and
                    best_2mm_score <= best_9mm_score + 0.04
                ):
                    template_match, template_ranges, template_nominal, template_kind = resolve_circle_profile_from_name(best_2mm_name)
                elif (
                    parts_name_profile is not None and
                    parts_name_profile[3] == "medium" and
                    best_match is not None and
                    best_match[3] == "small"
                ):
                    template_match, template_ranges, template_nominal, template_kind, _, _ = best_match
                elif parts_name_profile is not None:
                    template_match, template_ranges, template_nominal, template_kind = parts_name_profile
                elif fallback_profile is not None:
                    template_match, template_ranges, template_nominal, template_kind = fallback_profile
                elif best_match:
                    template_match, template_ranges, template_nominal, template_kind, _, _ = best_match

            active_feeder_auto_tc = detect_active_feeder_auto_tc(
                table_arr, active_row, preferred_parts_name_offset
            )
            if active_parts_name in KNOWN_CYLINDER_PART_NAME_SUFFIXES:
                active_feeder_auto_tc = False

        # Filter pixels
        r = crop[:, :, 0].astype(float)
        g = crop[:, :, 1].astype(float)
        b = crop[:, :, 2].astype(float)

        mean_val = (r + g + b) / 3.0
        is_gray = (np.abs(r - g) < 15) & (np.abs(g - b) < 15) & (np.abs(r - b) < 15)

        # 2. Precedence and Detection Paths:
        # Define nested helper functions for circle and cylinder detection.
        # Nested functions inherit all cropped parameters from the enclosing scope.

        def detect_cylinder():
            h_crop, w_crop = crop.shape[:2]
            y_grid, x_grid = np.ogrid[:h_crop, :w_crop]
            cyl_margin = 25
            mid_band = (y_grid >= 65) & (y_grid <= 195)

            crop_mean = np.mean(mean_val)
            cyl_thresh = max(130.0, crop_mean + 15.0)

            dark_body_mask_cyl = (
                (mean_val > 50) & (mean_val < cyl_thresh) & is_gray & mid_band &
                (x_grid >= cyl_margin) & (x_grid < w_crop - cyl_margin)
            )
            bright_cap_mask_cyl = (
                (mean_val >= cyl_thresh) & is_gray & mid_band &
                (x_grid >= cyl_margin) & (x_grid < w_crop - cyl_margin)
            )

            dark_col_cyl = dark_body_mask_cyl.sum(axis=0).astype(float)
            bright_col_cyl = bright_cap_mask_cyl.sum(axis=0).astype(float)
            net_col_cyl = bright_col_cyl - dark_col_cyl
            kernel5 = np.ones(5) / 5.0
            net_smooth_cyl = np.convolve(net_col_cyl, kernel5, mode='same')

            # Find valley in the body zone (x=60..180)
            body_search_cyl = net_smooth_cyl[60:180]
            valley_idx_cyl = int(np.argmin(body_search_cyl)) + 60

            # Left cap search: find peak in [35, valley_idx_cyl - 15]
            search_start_l = 35
            search_end_l = max(search_start_l + 1, valley_idx_cyl - 15)
            peak_idx_l = int(np.argmax(bright_col_cyl[search_start_l:search_end_l])) + search_start_l
            peak_val_l = bright_col_cyl[peak_idx_l]

            # Right cap search: find peak in [valley_idx_cyl + 15, 210]
            search_start_r = min(w_crop - cyl_margin - 1, valley_idx_cyl + 15)
            search_end_r = 210
            peak_idx_r = int(np.argmax(bright_col_cyl[search_start_r:search_end_r])) + search_start_r
            peak_val_r = bright_col_cyl[peak_idx_r]

            # Centering and peak validity checks
            left_peak_ok = (peak_val_l >= 15)
            right_peak_ok = (peak_val_r >= 15)
            midpoint_x = (peak_idx_l + peak_idx_r) / 2.0
            peaks_centered = (95 <= midpoint_x <= 155)

            # Balance check
            left_bright_sum = bright_col_cyl[cyl_margin:valley_idx_cyl].sum()
            right_bright_sum = bright_col_cyl[valley_idx_cyl:w_crop - cyl_margin].sum()
            balance_ok = (min(left_bright_sum, right_bright_sum) > 0 and
                          max(left_bright_sum, right_bright_sum) / max(min(left_bright_sum, right_bright_sum), 1.0) < 4.0)

            # Continuity check: count consecutive columns between peaks where bright count <= 5
            consec_low = 0
            max_consec_low = 0
            for x in range(int(peak_idx_l) + 1, int(peak_idx_r)):
                if bright_col_cyl[x] <= 5:
                    consec_low += 1
                    max_consec_low = max(max_consec_low, consec_low)
                else:
                    consec_low = 0

            is_continuous = (max_consec_low < 10)
            is_full_cyl = (left_peak_ok and right_peak_ok and peaks_centered and is_continuous and balance_ok and net_smooth_cyl[valley_idx_cyl] < -30)

            gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            blur_crop = cv2.GaussianBlur(gray_crop, (5, 5), 0)
            edge_mask = cv2.Canny(blur_crop, 35, 110)
            edge_mask[~is_gray] = 0
            edge_mask[:65, :] = 0
            edge_mask[196:, :] = 0
            edge_mask[:, :cyl_margin] = 0
            edge_mask[:, w_crop - cyl_margin:] = 0

            cross_x_local = int(round(xc_cross - x_start))
            cross_y_local = int(round(yc_cross - y_start))
            edge_mask[:, max(0, cross_x_local - 2):min(edge_mask.shape[1], cross_x_local + 3)] = 0
            edge_mask[max(0, cross_y_local - 2):min(edge_mask.shape[0], cross_y_local + 3), :] = 0

            def build_cap_candidates(side_name, peak_idx, peak_value):
                if peak_value < 18:
                    return []

                x0 = max(15, int(peak_idx - 50))
                x1 = min(w_crop - 15, int(peak_idx + 50))
                bright_slice = bright_cap_mask_cyl[60:200, x0:x1]
                bright_y, _ = np.where(bright_slice)
                side_center_y = int(np.median(bright_y) + 60) if len(bright_y) >= 20 else cross_y_local
                y0 = max(25, side_center_y - 65)
                y1 = min(h_crop - 15, side_center_y + 65)

                roi_edges = edge_mask[y0:y1, x0:x1]
                edge_y, edge_x = np.where(roi_edges > 0)
                if len(edge_x) < 80:
                    return []

                circles = cv2.HoughCircles(
                    blur_crop[y0:y1, x0:x1],
                    cv2.HOUGH_GRADIENT,
                    dp=1.2,
                    minDist=20,
                    param1=90,
                    param2=15,
                    minRadius=18,
                    maxRadius=65,
                )

                seed_candidates = []
                if circles is not None:
                    seed_candidates.extend(circles[0][:12].tolist())
                seed_candidates.append(None)
                results = []

                for seed in seed_candidates:
                    if seed is None:
                        point_mask = np.ones(len(edge_x), dtype=bool)
                        seed_cx = float(peak_idx)
                        seed_cy = float(side_center_y)
                        seed_r = 32.0
                    else:
                        seed_cx = float(seed[0]) + x0
                        seed_cy = float(seed[1]) + y0
                        seed_r = float(seed[2])
                        dists = np.sqrt((edge_x + x0 - seed_cx) ** 2 + (edge_y + y0 - seed_cy) ** 2)
                        point_mask = np.abs(dists - seed_r) <= max(4.0, seed_r * 0.12)
                        if side_name == "left":
                            point_mask &= (edge_x + x0) <= seed_cx + seed_r * 0.55
                        else:
                            point_mask &= (edge_x + x0) >= seed_cx - seed_r * 0.55

                    x_fit = edge_x[point_mask] + x0 + x_start
                    y_fit = edge_y[point_mask] + y0 + y_start
                    if len(x_fit) < 80:
                        continue

                    circle_fit = fit_circle_ransac_unanchored(
                        x_fit.astype(float),
                        y_fit.astype(float),
                        [(20, 65)],
                        max_iterations=1200,
                        threshold=3.5,
                        min_inliers=10,
                    )
                    if circle_fit is None:
                        continue

                    fit_x, fit_y, fit_r = circle_fit
                    fit_local_x = fit_x - x_start
                    fit_local_y = fit_y - y_start
                    if side_name == "left" and fit_local_x > peak_idx + 40:
                        continue
                    if side_name == "right" and fit_local_x < peak_idx - 40:
                        continue

                    metrics = get_circle_support_metrics(
                        x_fit.astype(float),
                        y_fit.astype(float),
                        fit_x,
                        fit_y,
                        fit_r,
                    )
                    if metrics["support_count"] < 80:
                        continue
                    if metrics["coverage_bins"] < 12:
                        continue
                    if metrics["residual_mean"] > 2.8 or metrics["residual_p90"] > 5.8:
                        continue
                    if metrics["span_y_ratio"] < 0.78:
                        continue

                    support_x, support_y, _ = get_circle_support_points(
                        x_fit.astype(float),
                        y_fit.astype(float),
                        fit_x,
                        fit_y,
                        fit_r,
                    )
                    if len(support_x) == 0:
                        continue

                    seed_penalty = abs(fit_local_x - seed_cx) + abs(fit_local_y - seed_cy) + abs(fit_r - seed_r)
                    center_penalty = abs(fit_local_x - peak_idx) + abs(fit_local_y - side_center_y)
                    valley_penalty = 0.0
                    if side_name == "left":
                        valley_penalty = max(0.0, (fit_local_x + fit_r) - (valley_idx_cyl + 12))
                    else:
                        valley_penalty = max(0.0, (valley_idx_cyl - 12) - (fit_local_x - fit_r))

                    sort_key = (
                        metrics["residual_mean"],
                        metrics["residual_p90"],
                        valley_penalty,
                        center_penalty,
                        seed_penalty,
                        -metrics["coverage_bins"],
                        -metrics["cardinal_support"],
                        -metrics["support_count"],
                    )
                    results.append({
                        "side": side_name,
                        "circle": circle_fit,
                        "support_x": support_x,
                        "support_y": support_y,
                        "metrics": metrics,
                        "sort_key": sort_key,
                    })

                results.sort(key=lambda item: item["sort_key"])
                return results

            left_candidates = build_cap_candidates("left", peak_idx_l, peak_val_l)
            right_candidates = build_cap_candidates("right", peak_idx_r, peak_val_r)

            left_candidate = left_candidates[0] if left_candidates else None
            right_candidate = right_candidates[0] if right_candidates else None
            had_partial_candidate = left_candidate is not None or right_candidate is not None

            if left_candidate is not None and right_candidate is not None:
                best_pair = None
                for left_option in left_candidates[:4]:
                    for right_option in right_candidates[:4]:
                        left_x, left_y, left_r = left_option["circle"]
                        right_x, right_y, right_r = right_option["circle"]
                        if right_x <= left_x:
                            continue
                        pair_overlap = max(0.0, (left_x + left_r) - (right_x - right_r))
                        center_separation = right_x - left_x
                        pair_key = (
                            abs(left_y - right_y),
                            abs(left_r - right_r),
                            pair_overlap,
                            -center_separation,
                            left_option["sort_key"],
                            right_option["sort_key"],
                        )
                        if best_pair is None or pair_key < best_pair[0]:
                            best_pair = (pair_key, left_option, right_option)
                if best_pair is not None:
                    pair_key, left_option, right_option = best_pair
                    left_x, left_y, left_r = left_option["circle"]
                    right_x, right_y, right_r = right_option["circle"]
                    pair_overlap = max(0.0, (left_x + left_r) - (right_x - right_r))
                    center_separation = right_x - left_x
                    right_near_cross = abs(right_x - xc_cross) <= 12.0
                    left_far_from_cross = left_x <= xc_cross - 45.0
                    pair_sits_low = left_y >= yc_cross + 25.0 and right_y >= yc_cross + 25.0
                    if (
                        pair_key[0] <= 45.0 and
                        pair_key[1] <= 20.0 and
                        pair_overlap <= 10.0 and
                        center_separation >= 0.6 * max(left_r, right_r) and
                        not (right_near_cross and left_far_from_cross) and
                        not (
                            pair_sits_low and
                            active_feeder_auto_tc is not False and
                            active_parts_name not in KNOWN_CYLINDER_PART_NAME_SUFFIXES
                        )
                    ):
                        left_candidate = left_option
                        right_candidate = right_option
                    else:
                        left_candidate = None
                        right_candidate = None
                else:
                    left_candidate = None
                    right_candidate = None

            if left_candidate is None and right_candidate is None:
                if had_partial_candidate:
                    return make_no_detect_result()
                return None

            if left_candidate is None or right_candidate is None:
                return make_no_detect_result()

            def get_candidate_box(side_name, candidate):
                fit_x, fit_y, fit_r = candidate["circle"]
                support_x = candidate["support_x"]
                support_y = candidate["support_y"]
                if side_name == "left":
                    edge_x = int(round(np.percentile(support_x, 10))) if len(support_x) else int(round(fit_x - fit_r))
                else:
                    edge_x = int(round(np.percentile(support_x, 90))) if len(support_x) else int(round(fit_x + fit_r))
                top_y = int(round(np.percentile(support_y, 8))) if len(support_y) else int(round(fit_y - fit_r))
                bottom_y = int(round(np.percentile(support_y, 92))) if len(support_y) else int(round(fit_y + fit_r))
                return {
                    "edge_x": edge_x,
                    "top_y": top_y,
                    "bottom_y": bottom_y,
                    "center_x": int(round(fit_x)),
                    "center_y": int(round(fit_y)),
                    "radius": float(fit_r),
                }

            left_box = get_candidate_box("left", left_candidate) if left_candidate is not None else None
            right_box = get_candidate_box("right", right_candidate) if right_candidate is not None else None

            if left_box is not None and right_box is not None:
                raw_rect_left = left_box["edge_x"]
                raw_rect_right = right_box["edge_x"]
                raw_rect_top = min(left_box["top_y"], right_box["top_y"])
                raw_rect_bottom = max(left_box["bottom_y"], right_box["bottom_y"])
                base_rect_left = raw_rect_left
                base_rect_right = raw_rect_right
                base_rect_top = raw_rect_top
                base_rect_bottom = raw_rect_bottom

                outer_thresh = max(160.0, min(190.0, crop_mean + 5.0))
                outer_mask = (
                    (mean_val < outer_thresh) & is_gray & mid_band &
                    (x_grid >= cyl_margin) & (x_grid < w_crop - cyl_margin)
                )
                outer_y, outer_x = np.where(outer_mask)
                if len(outer_x) >= 4000:
                    left_percentile = 2.0
                    right_percentile = 90.0
                    top_percentile = 15.0
                    bottom_percentile = 95.0

                    if crop_mean < 150.0:
                        left_percentile = 2.0
                        right_percentile = 90.0
                        top_percentile = 10.0
                        bottom_percentile = 97.0
                    elif crop_mean >= 200.0:
                        outer_thresh = 220.0
                        outer_mask = (
                            (mean_val < outer_thresh) & is_gray & mid_band &
                            (x_grid >= cyl_margin) & (x_grid < w_crop - cyl_margin)
                        )
                        outer_y, outer_x = np.where(outer_mask)
                        left_percentile = 5.0
                        right_percentile = 95.0
                        top_percentile = 4.0
                        bottom_percentile = 99.0
                    elif crop_mean >= 170.0:
                        outer_thresh = max(180.0, min(200.0, crop_mean))
                        outer_mask = (
                            (mean_val < outer_thresh) & is_gray & mid_band &
                            (x_grid >= cyl_margin) & (x_grid < w_crop - cyl_margin)
                        )
                        outer_y, outer_x = np.where(outer_mask)
                        left_percentile = 6.0
                        right_percentile = 95.0
                        top_percentile = 2.0
                        bottom_percentile = 97.0

                    base_rect_left = int(round(np.percentile(outer_x + x_start, left_percentile)))
                    base_rect_right = int(round(np.percentile(outer_x + x_start, right_percentile)))
                    base_rect_top = int(round(np.percentile(outer_y + y_start, top_percentile)))
                    base_rect_bottom = int(round(np.percentile(outer_y + y_start, bottom_percentile)))

                rect_left = base_rect_left
                rect_right = base_rect_right
                rect_top = base_rect_top
                rect_bottom = base_rect_bottom
                left_local_x = int(round(left_candidate["circle"][0] - x_start))
                right_local_x = int(round(right_candidate["circle"][0] - x_start))
                base_left_local = int(base_rect_left - x_start)
                base_right_local = int(base_rect_right - x_start)
                base_top_local = int(base_rect_top - y_start)
                base_bottom_local = int(base_rect_bottom - y_start)

                raw_box_width = max(1, base_right_local - base_left_local)
                raw_box_height = max(1, base_bottom_local - base_top_local)
                prelim_left = max(cyl_margin, base_left_local - 18)
                prelim_right = min(w_crop - cyl_margin - 1, base_right_local + 18)
                prelim_top = max(65, base_top_local - 16)
                prelim_bottom = min(195, base_bottom_local + 16)

                edge_binary = edge_mask > 0
                expected_top_local = max(prelim_top, min(prelim_bottom, base_top_local))
                expected_bottom_local = max(prelim_top, min(prelim_bottom, base_bottom_local))

                row_inner_margin = max(12, int(round(raw_box_width * 0.16)))
                row_x0 = min(prelim_right, base_left_local + row_inner_margin)
                row_x1 = max(row_x0, base_right_local - row_inner_margin)

                def choose_horizontal_edge(expected_row, search_above, search_below):
                    search_top = max(prelim_top, expected_row - search_above)
                    search_bottom = min(prelim_bottom, expected_row + search_below)
                    if search_bottom < search_top or row_x1 < row_x0:
                        return None

                    best_row = None
                    best_score = None
                    min_support = max(18, int(round((row_x1 - row_x0 + 1) * 0.24)))
                    for row_idx in range(search_top, search_bottom + 1):
                        band_top = max(prelim_top, row_idx - 1)
                        band_bottom = min(prelim_bottom, row_idx + 1)
                        row_band = edge_binary[band_top:band_bottom + 1, row_x0:row_x1 + 1]
                        support_cols = np.where(np.any(row_band, axis=0))[0]
                        if len(support_cols) < min_support:
                            continue

                        span = float(support_cols[-1] - support_cols[0] + 1)
                        density = float(np.count_nonzero(row_band))
                        score = (
                            span * 1.4 +
                            len(support_cols) * 0.8 +
                            density * 0.2 -
                            abs(row_idx - expected_row) * 2.2
                        )
                        candidate = (score, -abs(row_idx - expected_row), -density)
                        if best_score is None or candidate > best_score:
                            best_score = candidate
                            best_row = row_idx
                    return best_row

                refined_top_local = choose_horizontal_edge(expected_top_local, 10, 16)
                refined_bottom_local = choose_horizontal_edge(expected_bottom_local, 32, 10)
                if refined_top_local is not None:
                    rect_top = refined_top_local + y_start
                if refined_bottom_local is not None:
                    rect_bottom = refined_bottom_local + y_start

                band_top_local = max(prelim_top, min(prelim_bottom, rect_top - y_start))
                band_bottom_local = max(prelim_top, min(prelim_bottom, rect_bottom - y_start))
                if band_bottom_local <= band_top_local + 20:
                    band_top_local = max(prelim_top, min(prelim_bottom - 21, expected_top_local))
                    band_bottom_local = min(prelim_bottom, max(prelim_top + 21, expected_bottom_local))

                band_height = band_bottom_local - band_top_local + 1
                if band_height >= 24:
                    top_band_end = min(
                        band_bottom_local,
                        band_top_local + max(6, int(round(band_height * 0.24))),
                    )
                    mid_band_top = min(
                        band_bottom_local,
                        band_top_local + max(5, int(round(band_height * 0.30))),
                    )
                    mid_band_bottom = max(
                        mid_band_top,
                        min(band_bottom_local, band_top_local + int(round(band_height * 0.74))),
                    )
                    def get_column_support(col_idx):
                        strip_left = max(prelim_left, col_idx - 2)
                        strip_right = min(prelim_right, col_idx + 2)

                        full_strip = edge_binary[band_top_local:band_bottom_local + 1, strip_left:strip_right + 1]
                        top_strip = edge_binary[band_top_local:top_band_end + 1, strip_left:strip_right + 1]
                        mid_strip = edge_binary[mid_band_top:mid_band_bottom + 1, strip_left:strip_right + 1]

                        full_rows = int(np.count_nonzero(np.any(full_strip, axis=1)))
                        top_rows = int(np.count_nonzero(np.any(top_strip, axis=1)))
                        mid_rows = int(np.count_nonzero(np.any(mid_strip, axis=1)))
                        full_density = int(np.count_nonzero(full_strip))
                        score = full_rows + (1.15 * top_rows) + (0.95 * mid_rows) + (0.04 * full_density)
                        return {
                            "full_rows": full_rows,
                            "top_rows": top_rows,
                            "mid_rows": mid_rows,
                            "score": score,
                        }

                    def collect_vertical_candidates(x_min, x_max):
                        candidates = []
                        if x_max < x_min:
                            return candidates

                        min_full_rows = max(10, int(round(band_height * 0.18)))
                        min_top_rows = max(3, int(round((top_band_end - band_top_local + 1) * 0.12)))
                        min_mid_rows = max(4, int(round((mid_band_bottom - mid_band_top + 1) * 0.10)))
                        for col_idx in range(x_min, x_max + 1):
                            support = get_column_support(col_idx)
                            full_rows = support["full_rows"]
                            top_rows = support["top_rows"]
                            mid_rows = support["mid_rows"]
                            if (
                                full_rows < min_full_rows or
                                top_rows < min_top_rows or
                                mid_rows < min_mid_rows
                            ):
                                continue

                            candidates.append({
                                "x": col_idx,
                                "score": support["score"],
                            })
                        return candidates

                    def choose_vertical_edge(candidates, raw_edge_local):
                        if not candidates:
                            return None

                        max_score = max(candidate["score"] for candidate in candidates)
                        min_score = max_score * 0.72
                        strong_candidates = [candidate for candidate in candidates if candidate["score"] >= min_score]
                        if not strong_candidates:
                            strong_candidates = [candidate for candidate in candidates if candidate["score"] >= max_score * 0.55]
                        if not strong_candidates:
                            return None

                        best_candidate = None
                        best_key = None
                        for candidate in strong_candidates:
                            distance = abs(candidate["x"] - raw_edge_local)
                            key = (
                                candidate["score"] - (0.35 * distance),
                                -distance,
                            )
                            if best_key is None or key > best_key:
                                best_key = key
                                best_candidate = candidate["x"]

                        if best_candidate is None:
                            return None
                        if abs(best_candidate - raw_edge_local) > 24:
                            return None
                        return best_candidate

                    left_candidates_persistent = collect_vertical_candidates(
                        max(prelim_left, base_left_local - 12),
                        min(prelim_right, base_left_local + 12),
                    )
                    right_candidates_persistent = collect_vertical_candidates(
                        max(prelim_left, base_right_local - 12),
                        min(prelim_right, base_right_local + 12),
                    )

                    persistent_left = choose_vertical_edge(left_candidates_persistent, base_left_local)
                    persistent_right = choose_vertical_edge(right_candidates_persistent, base_right_local)

                    if persistent_left is not None:
                        rect_left = persistent_left + x_start
                    if persistent_right is not None:
                        rect_right = persistent_right + x_start

                if abs(rect_left - base_rect_left) > 4:
                    rect_left = base_rect_left
                if abs(rect_right - base_rect_right) > 4:
                    rect_right = base_rect_right
                if rect_top != base_rect_top:
                    rect_top = base_rect_top
                if 150.0 <= crop_mean < 170.0:
                    rect_top = max(rect_top, yc_cross - 40)
                    rect_bottom = max(rect_bottom, yc_cross + 63)
                    rect_right = min(rect_right, xc_cross + 75)
                if crop_mean >= 200.0:
                    rect_top = max(rect_top, yc_cross - 40)
                    rect_bottom = max(rect_bottom, yc_cross + 63)
                    rect_right = min(rect_right, xc_cross + 75)

                refined_width = rect_right - rect_left
                refined_height = rect_bottom - rect_top
                if refined_height > refined_width * 0.65:
                    return make_no_detect_result()
                if (
                    refined_width < raw_box_width - 8 or
                    refined_width > raw_box_width + 8 or
                    refined_height < raw_box_height - 28 or
                    refined_height > raw_box_height + 8
                ):
                    rect_left = base_rect_left
                    rect_right = base_rect_right
                    rect_top = base_rect_top
                    rect_bottom = base_rect_bottom

                left_point = [rect_left, rect_bottom]
                right_point = [rect_right, rect_top]
                center_x = int(round((rect_left + rect_right) / 2.0))
                center_y = int(round((rect_top + rect_bottom) / 2.0))
            visible_sides = []
            fit_meta = {"visible_sides": visible_sides}
            radii = []
            box_meta = {
                "top_left": None,
                "top_right": None,
                "bottom_left": None,
                "bottom_right": None,
                "width": None,
                "height": None,
            }

            if left_candidate is not None:
                visible_sides.append("left")
                fit_meta["left_cap_center"] = [
                    int(round(left_candidate["circle"][0])),
                    int(round(left_candidate["circle"][1])),
                ]
                fit_meta["left_cap_radius"] = float(left_candidate["circle"][2])
                radii.append(float(left_candidate["circle"][2]))
            if right_candidate is not None:
                visible_sides.append("right")
                fit_meta["right_cap_center"] = [
                    int(round(right_candidate["circle"][0])),
                    int(round(right_candidate["circle"][1])),
                ]
                fit_meta["right_cap_radius"] = float(right_candidate["circle"][2])
                radii.append(float(right_candidate["circle"][2]))
            if radii:
                fit_meta["cap_radius"] = float(np.mean(radii))

            box_meta["top_left"] = [left_point[0], right_point[1]]
            box_meta["top_right"] = right_point
            box_meta["bottom_left"] = left_point
            box_meta["bottom_right"] = [right_point[0], left_point[1]]
            box_meta["width"] = int(right_point[0] - left_point[0])
            box_meta["height"] = int(left_point[1] - right_point[1])
            fit_meta["box"] = box_meta

            return {
                "top": None,
                "bottom": None,
                "left": left_point,
                "right": right_point,
            }

        def detect_circle(mean_val=mean_val):
            crop_mean = np.mean(mean_val)
            crop_std = np.std(mean_val)
            crop_p5 = np.percentile(mean_val, 5)

            # Compute quadrant averages to determine target orientation for large washers.
            crop_gray_q = np.mean(crop, axis=2)
            h_q, w_q = crop_gray_q.shape[:2]
            cy_q, cx_q = h_q // 2, w_q // 2
            top_q_mean = np.mean(crop_gray_q[:cy_q, :])
            bottom_q_mean = np.mean(crop_gray_q[cy_q:, :])
            left_q_mean = np.mean(crop_gray_q[:, :cx_q])
            right_q_mean = np.mean(crop_gray_q[:, cx_q:])
            large_center_mode = resolve_large_center_mode_from_quadrants(
                top_q_mean,
                bottom_q_mean,
                left_q_mean,
                right_q_mean,
            )

            # Detect active teaching mode for small circles based on SMT UI label text.
            cyan_pixels = (
                (crop[:, :, 1] > 120) &
                (crop[:, :, 2] > 120) &
                ((crop[:, :, 1].astype(int) - crop[:, :, 0].astype(int)) > 20) &
                ((crop[:, :, 2].astype(int) - crop[:, :, 0].astype(int)) > 20)
            )
            crop_scale = crop.shape[0] / 254.0
            crop_px = lambda value: max(0, int(round(value * crop_scale)))
            top_region = cyan_pixels[crop_px(20):crop_px(100), crop_px(110):crop_px(144)].copy()
            bottom_region = cyan_pixels[crop_px(154):crop_px(234), crop_px(110):crop_px(144)].copy()
            left_region = cyan_pixels[crop_px(110):crop_px(144), crop_px(20):crop_px(100)].copy()
            right_region = cyan_pixels[crop_px(110):crop_px(144), crop_px(154):crop_px(234)].copy()

            top_region[:, crop_px(12):crop_px(22)] = False
            bottom_region[:, crop_px(12):crop_px(22)] = False
            left_region[crop_px(12):crop_px(22), :] = False
            right_region[crop_px(12):crop_px(22), :] = False

            overlay_counts = {
                "top": int(np.count_nonzero(top_region)),
                "bottom": int(np.count_nonzero(bottom_region)),
                "left": int(np.count_nonzero(left_region)),
                "right": int(np.count_nonzero(right_region)),
            }
            top_label_present = overlay_counts["top"] >= 20
            bottom_label_present = overlay_counts["bottom"] >= 20
            left_label_present = overlay_counts["left"] >= 20
            right_label_present = overlay_counts["right"] >= 20
            label_overlay_count = sum(
                int(flag)
                for flag in (
                    top_label_present,
                    bottom_label_present,
                    left_label_present,
                    right_label_present,
                )
            )
            has_label_overlay = label_overlay_count >= 2

            if top_label_present:
                small_anchor_mode = "top"
            elif bottom_label_present:
                small_anchor_mode = "bottom"
            elif left_label_present:
                small_anchor_mode = "left"
            elif right_label_present:
                small_anchor_mode = "right"
            else:
                small_anchor_mode = None

            small_search_ranges = template_ranges if template_kind in ("small", "medium") else [(20, 65)]
            small_nominal = template_nominal if template_kind in ("small", "medium") and template_nominal is not None else None
            medium_search_ranges = [(65, 90)]
            medium_nominal = 75
            large_search_ranges = template_ranges if template_kind == "large" else [(200, 320)]
            large_nominal = template_nominal if template_kind == "large" and template_nominal is not None else 240

            # Circular camera window mask: only consider pixels within the camera
            # window (radius ~120 pixels, centered at 127, 127 in the crop). Pixels
            # outside the camera window (e.g., the Executing dialog) are excluded
            # from detection.
            cy_center, cx_center = crop.shape[0] // 2, crop.shape[1] // 2
            camera_radius = crop_px(120)
            yy, xx = np.ogrid[:crop.shape[0], :crop.shape[1]]
            camera_mask = (xx - cx_center) ** 2 + (yy - cy_center) ** 2 <= camera_radius ** 2
            masked_mean = np.where(camera_mask, mean_val, crop_mean)

            # Mask out the SMT Camera UI labels around the small-part teaching markers.
            if template_kind != "large":
                small_mask_nominal = small_nominal if small_nominal is not None else 54
                cx_in_crop = xc_cross - x_start
                cy_in_crop = yc_cross - y_start

                def mask_rect(y0, y1, x0, x1):
                    y0 = max(0, int(y0))
                    y1 = min(masked_mean.shape[0], int(y1))
                    x0 = max(0, int(x0))
                    x1 = min(masked_mean.shape[1], int(x1))
                    if y0 < y1 and x0 < x1:
                        masked_mean[y0:y1, x0:x1] = crop_mean

                if top_label_present:
                    mask_rect(cy_in_crop - small_mask_nominal - 30, cy_in_crop - small_mask_nominal - 2, cx_in_crop - 25, cx_in_crop + 25)
                    mask_rect(cy_in_crop - small_mask_nominal - 12, cy_in_crop - small_mask_nominal + 12, cx_in_crop - 12, cx_in_crop + 12)
                if bottom_label_present:
                    mask_rect(cy_in_crop + small_mask_nominal - 25, cy_in_crop + small_mask_nominal - 2, cx_in_crop - 30, cx_in_crop + 30)
                    mask_rect(cy_in_crop + small_mask_nominal - 12, cy_in_crop + small_mask_nominal + 12, cx_in_crop - 12, cx_in_crop + 12)
                if left_label_present:
                    mask_rect(cy_in_crop - small_mask_nominal - 25, cy_in_crop - small_mask_nominal + 15, cx_in_crop - small_mask_nominal - 15, cx_in_crop - small_mask_nominal / 2 + 10)
                    mask_rect(cy_in_crop - 12, cy_in_crop + 12, cx_in_crop - small_mask_nominal - 12, cx_in_crop - small_mask_nominal + 12)
                if right_label_present:
                    mask_rect(cy_in_crop - small_mask_nominal - 25, cy_in_crop - small_mask_nominal + 15, cx_in_crop + small_mask_nominal / 2 - 10, cx_in_crop + small_mask_nominal + 15)
                    mask_rect(cy_in_crop - 12, cy_in_crop + 12, cx_in_crop + small_mask_nominal - 12, cx_in_crop + small_mask_nominal + 12)

            # Executing dialog mask: the dialog is always in the upper-right
            # portion of the screen. Set those pixels to background to
            # prevent false edges from interfering with RANSAC.
            DIALOG_Y_LO = crop_px(80)
            DIALOG_Y_HI = crop_px(220)
            DIALOG_X_LO = crop_px(170)
            DIALOG_X_HI = crop.shape[1]

            # Mask out any UI pixels (which are colored/saturated) and their black outlines/shadows.
            # Real camera pixels are grayscale (R, G, B channels are very close).
            r_crop = crop[:, :, 0].astype(float)
            g_crop = crop[:, :, 1].astype(float)
            b_crop = crop[:, :, 2].astype(float)
            is_ui = (np.maximum(np.maximum(r_crop, g_crop), b_crop) - np.minimum(np.minimum(r_crop, g_crop), b_crop)) > 20
            dialog_region = is_ui[DIALOG_Y_LO:DIALOG_Y_HI, DIALOG_X_LO:DIALOG_X_HI]
            dialog_present = np.count_nonzero(dialog_region) > 800

            if dialog_present:
                return finalize_result(make_no_detect_result())

            preserve_faint_upper_right_target = (
                template_kind == "small" and
                crop_p5 >= 230.0 and
                crop_std < 18.0
            )
            if dialog_present or not preserve_faint_upper_right_target:
                masked_mean[DIALOG_Y_LO:DIALOG_Y_HI, DIALOG_X_LO:DIALOG_X_HI] = crop_mean

            # Dilate the UI mask by 6 pixels to completely cover the black borders/outlines of the UI
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
            dilated_ui = cv2.dilate(is_ui.astype(np.uint8), kernel).astype(bool)

            masked_mean[dilated_ui] = crop_mean

            is_not_cyan = ~is_ui

            # Reassign mean_val to masked_mean for adaptive threshold check
            # This ensures center red dots, axes lines, and text labels do not trigger false contrast indicators.
            mean_val = masked_mean
            center_disk_mask = ((xx - cx_center) ** 2 + (yy - cy_center) ** 2) <= 24 ** 2
            center_disk_mean = float(np.mean(mean_val[center_disk_mask & camera_mask]))
            center_object_present = center_disk_mean < (crop_mean - 12.0)
            bright_edge_mode = (
                template_kind == "small" and
                crop_mean >= 220.0 and
                not center_object_present and
                crop_std >= 18.0
            )

            if crop_mean < 155:
                thresholds = [crop_mean - 20]
            elif crop_mean < 200:
                thresholds = [crop_mean - 15]
            elif crop_mean < 220:
                thresholds = [150]
            else:
                # Rule 3: High-Contrast Adaptive Thresholding
                # If background is bright and a high-contrast target is present, use 180 instead of 245.
                # Otherwise, ensure the upper threshold is always below crop_mean to prevent background noise capture.
                has_high_contrast = (np.min(mean_val) < 150)
                upper_thresh = 180 if has_high_contrast else min(245, int(crop_mean - 15))
                thresholds = [180, upper_thresh]

            is_gray_local = (np.abs(r - g) < 15) & (np.abs(g - b) < 15) & (np.abs(r - b) < 15)
            padded_mean = np.pad(mean_val, 1, mode='edge')
            grad_x = np.abs(padded_mean[1:-1, 2:] - padded_mean[1:-1, :-2])
            grad_y = np.abs(padded_mean[2:, 1:-1] - padded_mean[:-2, 1:-1])
            contrast_threshold = max(12.0, min(28.0, crop_std * 0.6))
            contrast_edge_mask = (
                camera_mask &
                is_gray_local &
                is_not_cyan &
                (~dilated_ui) &
                (np.maximum(grad_x, grad_y) > contrast_threshold)
            )

            def resolve_circle_pixel_visibility(xc, yc, R, support_x, support_y):
                if support_x is None or support_y is None or len(support_x) == 0:
                    support_angles = np.array([], dtype=float)
                    support_x_sel = np.array([], dtype=float)
                    support_y_sel = np.array([], dtype=float)
                else:
                    support_tolerance = max(4.0, min(10.0, R * 0.12))
                    support_error = np.abs(np.sqrt((support_x - xc) ** 2 + (support_y - yc) ** 2) - R)
                    support_mask = support_error <= support_tolerance
                    support_x_sel = support_x[support_mask]
                    support_y_sel = support_y[support_mask]
                    if len(support_x_sel) < 12:
                        support_x_sel = support_x
                        support_y_sel = support_y
                    support_angles = np.mod(np.arctan2(support_y_sel - yc, support_x_sel - xc), 2.0 * np.pi)

                ring_tolerance = max(4.0, min(9.0, R * 0.12))
                angle_tolerance = np.deg2rad(40.0 if R < 120 else 32.0)
                arc_angle_tolerance = np.deg2rad(22.0 if R < 120 else 16.0)
                point_tolerance = max(14.0, min(28.0, R * 0.4))
                min_support_points = 3 if R < 120 else 2
                min_local_points = 1
                min_arc_edge_points = 2 if R < 120 else 2
                min_arc_tone_points = 4 if R < 120 else 3
                overlay_visibility_mode = has_label_overlay

                local_xc = xc - x_start
                local_yc = yc - y_start
                ring_error = np.abs(np.sqrt((xx - local_xc) ** 2 + (yy - local_yc) ** 2) - R)
                ring_mask = (
                    camera_mask &
                    (~dilated_ui) &
                    (ring_error <= ring_tolerance)
                )
                ring_tone_threshold = crop_mean - max(8.0, crop_std * 0.35)
                ring_tone_mask = ring_mask & (mean_val <= ring_tone_threshold)

                raw_pixels = {
                    "top": [int(round(xc)), int(round(yc - R))],
                    "bottom": [int(round(xc)), int(round(yc + R))],
                    "left": [int(round(xc - R)), int(round(yc))],
                    "right": [int(round(xc + R)), int(round(yc))],
                }
                point_angles = {
                    "top": 3.0 * np.pi / 2.0,
                    "bottom": np.pi / 2.0,
                    "left": np.pi,
                    "right": 0.0,
                }

                resolved_pixels = {}
                for name, pixel in raw_pixels.items():
                    px, py = pixel
                    if px < 0 or py < 0 or px >= arr.shape[1] or py >= arr.shape[0]:
                        resolved_pixels[name] = None
                        continue

                    local_x = int(round(px - x_start))
                    local_y = int(round(py - y_start))
                    if local_x < 0 or local_y < 0 or local_x >= crop.shape[1] or local_y >= crop.shape[0]:
                        resolved_pixels[name] = None
                        continue

                    if support_angles.size > 0:
                        local_mask = ((support_x_sel - px) ** 2 + (support_y_sel - py) ** 2) <= point_tolerance ** 2
                        if np.count_nonzero(local_mask) >= min_local_points:
                            resolved_pixels[name] = pixel
                            continue

                        sector_mask = angular_distance(support_angles, point_angles[name]) <= angle_tolerance
                        if np.count_nonzero(sector_mask) >= min_support_points:
                            sector_x = support_x_sel[sector_mask]
                            sector_y = support_y_sel[sector_mask]
                            point_distance = np.sqrt((sector_x - px) ** 2 + (sector_y - py) ** 2)
                            if len(point_distance) > 0 and float(np.min(point_distance)) <= point_tolerance:
                                resolved_pixels[name] = pixel
                                continue
                            if overlay_visibility_mode and np.count_nonzero(sector_mask) >= (min_support_points + 1):
                                resolved_pixels[name] = pixel
                                continue

                    arc_mask = ring_mask & (
                        angular_distance(
                            np.mod(np.arctan2(yy - local_yc, xx - local_xc), 2.0 * np.pi),
                            point_angles[name],
                        ) <= arc_angle_tolerance
                    )
                    if (
                        np.count_nonzero(contrast_edge_mask & arc_mask) >= min_arc_edge_points or
                        np.count_nonzero(ring_tone_mask & arc_mask) >= min_arc_tone_points
                    ):
                        resolved_pixels[name] = pixel
                        continue

                    resolved_pixels[name] = None

                if (
                    resolved_pixels["top"] is None and
                    resolved_pixels["left"] is not None and
                    resolved_pixels["right"] is not None and
                    yc <= yc_cross + max(8.0, R * 0.1)
                ):
                    top_x, top_y = raw_pixels["top"]
                    top_local_x = int(round(top_x - x_start))
                    top_local_y = int(round(top_y - y_start))
                    if (
                        0 <= top_local_x < crop.shape[1] and
                        0 <= top_local_y < crop.shape[0]
                    ):
                        resolved_pixels["top"] = raw_pixels["top"]

                if (
                    R < 120 and
                    resolved_pixels["bottom"] is None and
                    resolved_pixels["top"] is not None and
                    resolved_pixels["left"] is not None and
                    resolved_pixels["right"] is not None
                ):
                    bottom_x, bottom_y = raw_pixels["bottom"]
                    bottom_local_x = int(round(bottom_x - x_start))
                    bottom_local_y = int(round(bottom_y - y_start))
                    if (
                        0 <= bottom_local_x < crop.shape[1] and
                        0 <= bottom_local_y < crop.shape[0]
                    ):
                        resolved_pixels["bottom"] = raw_pixels["bottom"]

                if R < 120:
                    visible_names = tuple(
                        name for name in ("top", "bottom", "left", "right")
                        if resolved_pixels[name] is not None
                    )
                    if len(visible_names) == 3:
                        for missing_name in ("top", "bottom", "left", "right"):
                            if resolved_pixels[missing_name] is not None:
                                continue
                            raw_x, raw_y = raw_pixels[missing_name]
                            local_x = int(round(raw_x - x_start))
                            local_y = int(round(raw_y - y_start))
                            if 0 <= local_x < crop.shape[1] and 0 <= local_y < crop.shape[0]:
                                resolved_pixels[missing_name] = raw_pixels[missing_name]
                            break

                centered_vertical_support = (
                    resolved_pixels["top"] is not None and
                    resolved_pixels["bottom"] is not None and
                    abs(yc - yc_cross) <= max(18.0, R * 0.35)
                )
                if centered_vertical_support and resolved_pixels["left"] is not None and resolved_pixels["right"] is None:
                    right_x, right_y = raw_pixels["right"]
                    right_local_x = int(round(right_x - x_start))
                    right_local_y = int(round(right_y - y_start))
                    if (
                        0 <= right_local_x < crop.shape[1] and
                        0 <= right_local_y < crop.shape[0]
                    ):
                        resolved_pixels["right"] = raw_pixels["right"]
                if centered_vertical_support and resolved_pixels["right"] is not None and resolved_pixels["left"] is None:
                    left_x, left_y = raw_pixels["left"]
                    left_local_x = int(round(left_x - x_start))
                    left_local_y = int(round(left_y - y_start))
                    if (
                        0 <= left_local_x < crop.shape[1] and
                        0 <= left_local_y < crop.shape[0]
                    ):
                        resolved_pixels["left"] = raw_pixels["left"]

                if (
                    resolved_pixels["bottom"] is None and
                    resolved_pixels["top"] is not None and
                    resolved_pixels["left"] is not None and
                    resolved_pixels["right"] is not None and
                    R < 120
                ):
                    bottom_x, bottom_y = raw_pixels["bottom"]
                    bottom_local_x = int(round(bottom_x - x_start))
                    bottom_local_y = int(round(bottom_y - y_start))
                    if (
                        0 <= bottom_local_x < crop.shape[1] and
                        0 <= bottom_local_y < crop.shape[0]
                    ):
                        resolved_pixels["bottom"] = raw_pixels["bottom"]

                return resolved_pixels

            def count_visible_circle_points(candidate):
                if candidate is None:
                    return -1
                circle_fit = candidate.get("circle")
                if circle_fit is None:
                    return -1
                visible_pixels = resolve_circle_pixel_visibility(
                    circle_fit[0],
                    circle_fit[1],
                    circle_fit[2],
                    candidate.get("support_x"),
                    candidate.get("support_y"),
                )
                return sum(
                    1
                    for name in ("top", "bottom", "left", "right")
                    if visible_pixels[name] is not None
                )

            def choose_medium_circle_candidate(refined_candidate, hough_candidate):
                if hough_candidate is None:
                    return refined_candidate
                if refined_candidate is None:
                    return hough_candidate

                refined_visible_count = count_visible_circle_points(refined_candidate)
                hough_visible_count = count_visible_circle_points(hough_candidate)
                if refined_visible_count > hough_visible_count:
                    return refined_candidate
                if hough_visible_count > refined_visible_count:
                    return hough_candidate
                return choose_candidate(refined_candidate, hough_candidate)

            def should_prefer_unprofiled_medium_candidate(small_candidate, medium_candidate):
                if medium_candidate is None:
                    return False
                if active_row == -1 or active_feeder_auto_tc is not True:
                    return False

                medium_circle = medium_candidate.get("circle")
                if medium_circle is None or medium_circle[2] < 80.0:
                    return False

                medium_visible_count = count_visible_circle_points(medium_candidate)
                if medium_visible_count < 3:
                    return False

                if small_candidate is None:
                    return True

                small_circle = small_candidate.get("circle")
                if small_circle is None:
                    return True
                if resolve_circle_profile_from_name(
                    infer_circle_session_suffix(template_match, active_parts_name, small_circle[2])
                ) is not None:
                    return False

                if medium_circle[2] < small_circle[2] + 18.0:
                    return False

                small_visible_count = count_visible_circle_points(small_candidate)
                return medium_visible_count >= small_visible_count

            def get_large_alignment_error(xc, yc, R):
                R_final = R + 5.5 if R > 120 else R
                if large_center_mode == "top":
                    return (xc - xc_cross) ** 2 + (yc - R_final - yc_cross) ** 2
                if large_center_mode == "bottom":
                    return (xc - xc_cross) ** 2 + (yc + R_final - yc_cross) ** 2
                if large_center_mode == "left":
                    return (xc - R_final - xc_cross) ** 2 + (yc - yc_cross) ** 2
                return (xc + R_final - xc_cross) ** 2 + (yc - yc_cross) ** 2

            def choose_candidate(best_candidate, candidate):
                if candidate is None:
                    return best_candidate
                if best_candidate is None or candidate["sort_key"] < best_candidate["sort_key"]:
                    return candidate
                return best_candidate

            def choose_small_circle_candidate(edge_candidate, opencv_candidate):
                if opencv_candidate is None:
                    return edge_candidate
                if edge_candidate is None:
                    return opencv_candidate

                if template_kind == "small" and (small_nominal is None or small_nominal >= 45.0):
                    edge_circle = edge_candidate.get("circle")
                    opencv_circle = opencv_candidate.get("circle")
                    if edge_circle is not None and opencv_circle is not None:
                        edge_x, edge_y, edge_r = edge_circle
                        opencv_x, opencv_y, opencv_r = opencv_circle
                        center_gap = float(np.hypot(edge_x - opencv_x, edge_y - opencv_y))
                        radius_gain = edge_r - opencv_r
                        if (
                            center_gap <= max(10.0, max(edge_r, opencv_r) * 0.18) and
                            radius_gain >= max(8.0, max(edge_r, opencv_r) * 0.12)
                        ):
                            edge_metrics = edge_candidate.get("metrics") or {}
                            opencv_metrics = opencv_candidate.get("metrics") or {}
                            if (
                                edge_metrics.get("cardinal_support", 0) + 1 >= opencv_metrics.get("cardinal_support", 0) and
                                edge_metrics.get("coverage_bins", 0) + 2 >= opencv_metrics.get("coverage_bins", 0) and
                                edge_metrics.get("residual_mean", float("inf")) <= opencv_metrics.get("residual_mean", float("inf")) + 1.2 and
                                edge_metrics.get("residual_p90", float("inf")) <= opencv_metrics.get("residual_p90", float("inf")) + 1.5
                            ):
                                return edge_candidate

                return opencv_candidate

            def get_medium_visibility_profile(xc_fit, yc_fit, R_fit):
                total_bins = 24
                visible_bins = 0
                angles = np.linspace(0.0, 2.0 * np.pi, total_bins, endpoint=False)
                local_xc = xc_fit - x_start
                local_yc = yc_fit - y_start

                for angle in angles:
                    px = int(round(local_xc + R_fit * np.cos(angle)))
                    py = int(round(local_yc + R_fit * np.sin(angle)))
                    if px < 0 or py < 0 or px >= crop.shape[1] or py >= crop.shape[0]:
                        continue
                    if not camera_mask[py, px]:
                        continue
                    if dilated_ui[py, px]:
                        continue
                    visible_bins += 1

                visible_cardinals = 0
                for px, py in (
                    (int(round(local_xc)), int(round(local_yc - R_fit))),
                    (int(round(local_xc)), int(round(local_yc + R_fit))),
                    (int(round(local_xc - R_fit)), int(round(local_yc))),
                    (int(round(local_xc + R_fit)), int(round(local_yc))),
                ):
                    if px < 0 or py < 0 or px >= crop.shape[1] or py >= crop.shape[0]:
                        continue
                    if not camera_mask[py, px]:
                        continue
                    if dilated_ui[py, px]:
                        continue
                    visible_cardinals += 1

                return {
                    "total_bins": total_bins,
                    "visible_bins": visible_bins,
                    "visible_cardinals": visible_cardinals,
                    "is_occluded": visible_bins < total_bins or visible_cardinals < 4,
                }

            def get_medium_ring_evidence_profile(xc_fit, yc_fit, R_fit):
                total_bins = 24
                local_xc = xc_fit - x_start
                local_yc = yc_fit - y_start
                ring_offset = max(5.0, min(10.0, R_fit * 0.10))
                sample_radius = int(round(max(2.0, min(4.0, R_fit * 0.035))))
                darkness_margin = max(5.0, min(12.0, crop_std * 0.45 + 2.0))
                valid_sample_mask = camera_mask & (~dilated_ui)
                dark_bins = 0
                visible_bins = 0
                dark_cardinals = 0
                strengths = []
                cardinal_bins = {0, 6, 12, 18}

                def sample_patch_mean(px, py):
                    x_c = int(round(px))
                    y_c = int(round(py))
                    x0 = max(0, x_c - sample_radius)
                    x1 = min(mean_val.shape[1], x_c + sample_radius + 1)
                    y0 = max(0, y_c - sample_radius)
                    y1 = min(mean_val.shape[0], y_c + sample_radius + 1)
                    if x0 >= x1 or y0 >= y1:
                        return None
                    patch_mask = valid_sample_mask[y0:y1, x0:x1]
                    if np.count_nonzero(patch_mask) < 4:
                        return None
                    return float(np.mean(mean_val[y0:y1, x0:x1][patch_mask]))

                for idx in range(total_bins):
                    angle = idx * (2.0 * np.pi / total_bins)
                    cos_a = np.cos(angle)
                    sin_a = np.sin(angle)
                    inner_mean = sample_patch_mean(local_xc + cos_a * (R_fit - ring_offset), local_yc + sin_a * (R_fit - ring_offset))
                    ring_mean = sample_patch_mean(local_xc + cos_a * R_fit, local_yc + sin_a * R_fit)
                    outer_mean = sample_patch_mean(local_xc + cos_a * (R_fit + ring_offset), local_yc + sin_a * (R_fit + ring_offset))
                    if inner_mean is None or ring_mean is None or outer_mean is None:
                        continue

                    visible_bins += 1
                    adjacent_mean = 0.5 * (inner_mean + outer_mean)
                    paired_strength = min(inner_mean, outer_mean) - ring_mean
                    contrast_strength = adjacent_mean - ring_mean
                    if contrast_strength >= darkness_margin and paired_strength >= (darkness_margin * 0.35):
                        dark_bins += 1
                        strengths.append(min(contrast_strength, paired_strength))
                        if idx in cardinal_bins:
                            dark_cardinals += 1

                return {
                    "visible_bins": visible_bins,
                    "dark_bins": dark_bins,
                    "dark_cardinals": dark_cardinals,
                    "dark_ratio": (float(dark_bins) / float(visible_bins)) if visible_bins > 0 else 0.0,
                    "mean_strength": float(np.mean(strengths)) if strengths else 0.0,
                }

            def is_valid_medium_candidate(metrics, visibility_profile, ring_profile):
                visible_bins = max(visibility_profile["visible_bins"], 1)
                visible_cardinals = visibility_profile["visible_cardinals"]
                coverage_ratio = metrics["coverage_bins"] / float(visible_bins)
                min_cardinal_support = 2 if visible_cardinals >= 3 else 1

                if visible_bins >= 18:
                    # Full / near-full visibility: require good arc coverage
                    min_coverage_bins = 11
                    min_span_ratio = 0.58
                    max_gap_bins = 15
                    min_coverage_ratio = 0.45
                    min_dark_bins = 8
                    min_dark_ratio = 0.34
                    min_dark_cardinals = 2
                    max_angular_balance = 0.36
                elif visible_bins >= 12:
                    # Partial visibility: moderate requirements
                    min_coverage_bins = 8
                    min_span_ratio = 0.42
                    max_gap_bins = 18
                    min_coverage_ratio = 0.40
                    min_dark_bins = 6
                    min_dark_ratio = 0.32
                    min_dark_cardinals = 1
                    max_angular_balance = 0.46
                else:
                    # Low visibility — ring_profile evidence is the primary guard
                    # against spurious fits from noise/dialog-edges.
                    min_coverage_bins = 6
                    min_span_ratio = 0.30
                    max_gap_bins = 20
                    min_coverage_ratio = 0.30
                    min_dark_bins = 4
                    min_dark_ratio = 0.30
                    min_dark_cardinals = 1
                    max_angular_balance = 0.58

                if metrics["cardinal_support"] < min(min_cardinal_support, max(1, visible_cardinals)):
                    return False
                if metrics["coverage_bins"] < min_coverage_bins:
                    return False
                if coverage_ratio < min_coverage_ratio:
                    return False
                if metrics["largest_gap_bins"] > max_gap_bins:
                    return False
                if metrics["min_span_ratio"] < min_span_ratio:
                    return False
                if metrics["angular_balance"] > max_angular_balance:
                    return False
                if ring_profile["dark_bins"] < min_dark_bins:
                    return False
                if ring_profile["dark_ratio"] < min_dark_ratio:
                    return False
                if ring_profile["dark_cardinals"] < min_dark_cardinals:
                    return False
                if ring_profile["mean_strength"] < 4.5:
                    return False
                return True

            def get_medium_outer_ring_sort_key(metrics, R_fit, nominal_radius, boundary_margin, residual_mean, residual_p90, circumference_error, center_distance, visibility_profile, ring_profile, interior_mean):
                radius_bias = abs(R_fit - nominal_radius) if nominal_radius is not None else 0.0
                boundary_penalty = max(0.0, 8.0 - boundary_margin)
                outer_ring_priority = 0 if nominal_radius is None or R_fit >= (nominal_radius - 4.0) else 1
                visible_bins = max(visibility_profile["visible_bins"], 1)
                coverage_ratio = metrics["coverage_bins"] / float(visible_bins)
                min_expected_cardinals = 2 if visibility_profile["visible_cardinals"] >= 3 else 1
                coverage_priority = 0 if metrics["cardinal_support"] >= min_expected_cardinals and coverage_ratio >= 0.24 else 1
                clipped_priority = 0 if boundary_margin >= 0.0 else 1
                ring_priority = 0 if ring_profile["dark_cardinals"] >= min_expected_cardinals and ring_profile["dark_ratio"] >= 0.32 else 1
                return (
                    outer_ring_priority,
                    coverage_priority,
                    ring_priority,
                    clipped_priority,
                    radius_bias,
                    metrics["angular_balance"],
                    -metrics["cardinal_support"],
                    -ring_profile["dark_cardinals"],
                    metrics["largest_gap_bins"],
                    -coverage_ratio,
                    -ring_profile["dark_ratio"],
                    -metrics["coverage_bins"],
                    -ring_profile["dark_bins"],
                    -metrics["min_span_ratio"],
                    -ring_profile["mean_strength"],
                    -interior_mean,
                    residual_mean,
                    residual_p90,
                    boundary_penalty,
                    circumference_error,
                    center_distance,
                    -metrics["support_count"],
                )

            def build_opencv_circle_candidate(circle_fit, support_x, support_y, nominal_radius, target_kind):
                if circle_fit is None or support_x is None or support_y is None or len(support_x) == 0:
                    return None

                xc_fit, yc_fit, R_fit = circle_fit
                search_ranges = small_search_ranges if target_kind == "small" else large_search_ranges
                if not any(low <= R_fit <= high for low, high in search_ranges):
                    return None
                if is_bezel(xc_fit, yc_fit, R_fit, xc_cross, yc_cross):
                    return None

                metrics = get_circle_support_metrics(
                    support_x,
                    support_y,
                    xc_fit,
                    yc_fit,
                    R_fit,
                    tolerance=max(4.0, min(9.0, R_fit * 0.12)),
                )
                if metrics["support_count"] < (8 if target_kind == "small" else 18):
                    return None
                if metrics["coverage_bins"] < (2 if target_kind == "small" else 3):
                    return None
                if metrics["residual_mean"] > (5.5 if target_kind == "small" else 6.5):
                    return None
                if metrics["residual_p90"] > (10.0 if target_kind == "small" else 12.0):
                    return None

                medium_ring_profile = None
                if template_kind == "medium":
                    medium_visibility_profile = get_medium_visibility_profile(xc_fit, yc_fit, R_fit)
                    medium_ring_profile = get_medium_ring_evidence_profile(xc_fit, yc_fit, R_fit)
                    if not is_valid_medium_candidate(metrics, medium_visibility_profile, medium_ring_profile):
                        return None

                radius_bias = abs(R_fit - nominal_radius) if nominal_radius is not None else 0.0
                center_distance = float(np.hypot(xc_fit - xc_cross, yc_fit - yc_cross))
                circumference_error = abs(center_distance - R_fit)
                medium_mode = template_kind == "medium"
                boundary_margin = min(
                    xc_fit - R_fit - x_start,
                    x_end - (xc_fit + R_fit),
                    yc_fit - R_fit - y_start,
                    y_end - (yc_fit + R_fit),
                )
                local_xc = xc_fit - x_start
                local_yc = yc_fit - y_start
                interior_mask = (
                    ((xx - local_xc) ** 2 + (yy - local_yc) ** 2) <= max(3.0, R_fit * 0.65) ** 2
                ) & camera_mask
                interior_mean = float(np.mean(mean_val[interior_mask])) if np.count_nonzero(interior_mask) >= 20 else 0.0
                medium_visibility_profile = get_medium_visibility_profile(xc_fit, yc_fit, R_fit) if template_kind == "medium" else None

                if target_kind == "small":
                    inferred_small_profile = infer_circle_session_suffix(template_match, active_parts_name, R_fit)
                    bright_faint_mode = template_kind == "small" and crop_p5 >= 230.0 and crop_std < 18.0
                    center_teaching_mode = (
                        template_kind == "small" and
                        center_object_present and
                        center_distance <= max(12.0, R_fit * 0.25)
                    )
                    inferred_modes = infer_small_anchor_modes(support_x, support_y)
                    if inferred_modes:
                        anchor_error = min(
                            get_circle_anchor_error(
                                xc_fit,
                                yc_fit,
                                R_fit,
                                xc_cross,
                                yc_cross,
                                anchor_mode=anchor_mode,
                            )
                            for anchor_mode in inferred_modes
                        )
                    else:
                        anchor_error = get_circle_anchor_error(
                            xc_fit,
                            yc_fit,
                            R_fit,
                            xc_cross,
                            yc_cross,
                            anchor_mode=None,
                        )
                    if (
                        not medium_mode and
                        not bright_faint_mode and
                        not center_teaching_mode and
                        circumference_error > max(18.0, R_fit * 0.4)
                    ):
                        return None
                    if bright_faint_mode:
                        coverage_ok = 0 if metrics["coverage_bins"] >= 4 else 1
                        sort_key = (
                            0,
                            coverage_ok,
                            radius_bias,
                            -metrics["support_count"],
                            -metrics["coverage_bins"],
                            metrics["largest_gap_bins"],
                            metrics["residual_mean"],
                            metrics["residual_p90"],
                            center_distance,
                        )
                    elif center_teaching_mode:
                        sort_key = (
                            0,
                            center_distance,
                            radius_bias,
                            metrics["residual_mean"],
                            -metrics["support_count"],
                            circumference_error,
                        )
                    elif medium_mode:
                        if not is_valid_medium_candidate(metrics, medium_visibility_profile, medium_ring_profile):
                            return None
                        sort_key = get_medium_outer_ring_sort_key(
                            metrics,
                            R_fit,
                            nominal_radius,
                            boundary_margin,
                            metrics["residual_mean"],
                            metrics["residual_p90"],
                            circumference_error,
                            center_distance,
                            medium_visibility_profile,
                            medium_ring_profile,
                            interior_mean,
                        )
                    else:
                        sort_key = (
                            1,
                            circumference_error,
                            radius_bias,
                            max(0.0, -boundary_margin),
                            -metrics["cardinal_support"],
                            metrics["largest_gap_bins"],
                            -metrics["coverage_bins"],
                            metrics["residual_mean"],
                            -metrics["support_count"],
                            center_distance,
                        )
                else:
                    sort_key = (
                        get_large_alignment_error(xc_fit, yc_fit, R_fit),
                        radius_bias,
                        metrics["residual_mean"],
                        -metrics["support_count"],
                    )

                return {
                    "circle": circle_fit,
                    "support_x": support_x,
                    "support_y": support_y,
                    "metrics": metrics,
                    "sort_key": sort_key,
                    "source": "opencv",
                }

            def detect_medium_hough_circle_candidate():
                relaxed_active_auto_tc = template_kind != "medium" and active_row != -1 and active_feeder_auto_tc is True
                if template_kind == "medium":
                    search_ranges = medium_search_ranges
                    nominal_radius = medium_nominal
                elif active_row != -1 and active_feeder_auto_tc is True:
                    search_ranges = medium_search_ranges
                    nominal_radius = medium_nominal
                else:
                    return None

                min_radius = int(max(1, min(low for low, _ in search_ranges)))
                max_radius = int(max(high for _, high in search_ranges))
                valid_mask = camera_mask & is_gray_local & is_not_cyan & (~dilated_ui)
                if np.count_nonzero(valid_mask) < 1000:
                    return None

                gray_raw = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                fill_value = int(np.median(gray_raw[valid_mask]))
                work = gray_raw.copy()
                work[~valid_mask] = fill_value
                blur = cv2.medianBlur(work, 5)

                edge_mask_hough = cv2.Canny(blur, 35, 100)
                edge_mask_hough[~valid_mask] = 0
                edge_y, edge_x = np.where(edge_mask_hough > 0)
                if len(edge_x) < 50:
                    return None

                edge_x_full = edge_x.astype(float) + x_start
                edge_y_full = edge_y.astype(float) + y_start
                best_candidate = None
                best_key = None

                for param2 in (20, 24, 28, 16, 12):
                    circles = cv2.HoughCircles(
                        blur,
                        cv2.HOUGH_GRADIENT,
                        dp=1.1,
                        minDist=40,
                        param1=80,
                        param2=param2,
                        minRadius=min_radius,
                        maxRadius=max_radius,
                    )
                    if circles is None:
                        continue

                    for cx_local, cy_local, radius_seed in circles[0][:8]:
                        cx_seed = float(cx_local + x_start)
                        cy_seed = float(cy_local + y_start)
                        radius_seed = float(radius_seed)
                        if not any(low <= radius_seed <= high for low, high in search_ranges):
                            continue

                        residuals = np.abs(
                            np.sqrt((edge_x_full - cx_seed) ** 2 + (edge_y_full - cy_seed) ** 2) - radius_seed
                        )
                        support_mask = residuals <= max(4.0, radius_seed * 0.10)
                        if np.count_nonzero(support_mask) < 20:
                            continue

                        support_x = edge_x_full[support_mask]
                        support_y = edge_y_full[support_mask]
                        metrics = get_circle_support_metrics(
                            support_x,
                            support_y,
                            cx_seed,
                            cy_seed,
                            radius_seed,
                            tolerance=max(4.0, min(9.0, radius_seed * 0.12)),
                        )
                        min_support = 70 if dialog_present else (90 if relaxed_active_auto_tc else 120)
                        min_coverage = 10 if dialog_present else (14 if relaxed_active_auto_tc else 16)
                        min_cardinals = 2 if dialog_present else (2 if relaxed_active_auto_tc else 3)
                        max_residual_mean = 5.4 if dialog_present else (5.2 if relaxed_active_auto_tc else 4.5)
                        max_residual_p90 = 9.2 if dialog_present else (9.0 if relaxed_active_auto_tc else 8.0)
                        if metrics["support_count"] < min_support:
                            continue
                        if metrics["coverage_bins"] < min_coverage:
                            continue
                        if metrics["cardinal_support"] < min_cardinals:
                            continue
                        if metrics["residual_mean"] > max_residual_mean or metrics["residual_p90"] > max_residual_p90:
                            continue

                        radius_bias = abs(radius_seed - nominal_radius) if nominal_radius is not None else 0.0
                        key = (
                            radius_bias,
                            metrics["residual_mean"],
                            metrics["residual_p90"],
                            -metrics["cardinal_support"],
                            -metrics["coverage_bins"],
                            -metrics["support_count"],
                            param2,
                        )
                        if best_key is None or key < best_key:
                            best_key = key
                            best_candidate = {
                                "circle": (cx_seed, cy_seed, radius_seed),
                                "support_x": support_x,
                                "support_y": support_y,
                                "metrics": metrics,
                                "sort_key": key,
                                "source": "medium_hough",
                            }

                    if best_candidate is not None:
                        return best_candidate

                return best_candidate

            def detect_profile_hough_circle_candidate():
                if template_kind not in ("small", "medium"):
                    return None

                search_ranges = small_search_ranges if template_kind == "small" else medium_search_ranges
                nominal_radius = small_nominal if template_kind == "small" else medium_nominal
                min_radius = int(max(1, min(low for low, _ in search_ranges)))
                max_radius = int(max(high for _, high in search_ranges))

                gray_raw = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                blur = cv2.medianBlur(gray_raw, 5)
                edge_mask = cv2.Canny(blur, 35, 100)
                edge_mask[~camera_mask] = 0
                edge_mask[dilated_ui] = 0
                edge_y, edge_x = np.where(edge_mask > 0)
                if len(edge_x) < 50:
                    return None

                edge_x_full = edge_x.astype(float) + x_start
                edge_y_full = edge_y.astype(float) + y_start
                best_candidate = None
                best_key = None

                for param2 in (18, 14, 10):
                    circles = cv2.HoughCircles(
                        blur,
                        cv2.HOUGH_GRADIENT,
                        dp=1.1,
                        minDist=40,
                        param1=80,
                        param2=param2,
                        minRadius=min_radius,
                        maxRadius=max_radius,
                    )
                    if circles is None:
                        continue

                    for cx_local, cy_local, radius_seed in circles[0][:8]:
                        cx_seed = float(cx_local + x_start)
                        cy_seed = float(cy_local + y_start)
                        radius_seed = float(radius_seed)
                        if not any(low <= radius_seed <= high for low, high in search_ranges):
                            continue

                        residuals = np.abs(
                            np.sqrt((edge_x_full - cx_seed) ** 2 + (edge_y_full - cy_seed) ** 2) - radius_seed
                        )
                        support_mask = residuals <= max(4.0, radius_seed * 0.10)
                        if np.count_nonzero(support_mask) < 40:
                            continue

                        support_x = edge_x_full[support_mask]
                        support_y = edge_y_full[support_mask]
                        metrics = get_circle_support_metrics(
                            support_x,
                            support_y,
                            cx_seed,
                            cy_seed,
                            radius_seed,
                            tolerance=max(4.0, min(9.0, radius_seed * 0.12)),
                        )
                        if metrics["coverage_bins"] < 8 or metrics["cardinal_support"] < 2:
                            continue
                        if metrics["residual_mean"] > 4.5 or metrics["residual_p90"] > 8.0:
                            continue

                        radius_bias = abs(radius_seed - nominal_radius) if nominal_radius is not None else 0.0
                        center_distance = float(np.hypot(cx_seed - xc_cross, cy_seed - yc_cross))
                        key = (
                            radius_bias,
                            center_distance if template_kind == "small" else 0.0,
                            metrics["residual_mean"],
                            metrics["residual_p90"],
                            -metrics["coverage_bins"],
                            -metrics["support_count"],
                            param2,
                        )
                        if best_key is None or key < best_key:
                            best_key = key
                            best_candidate = {
                                "circle": (cx_seed, cy_seed, radius_seed),
                                "support_x": support_x,
                                "support_y": support_y,
                                "metrics": metrics,
                                "sort_key": key,
                                "source": "profile_hough",
                            }

                    if best_candidate is not None:
                        return best_candidate

                return best_candidate

            def detect_crosshair_small_hough_candidate():
                if active_row == -1:
                    return None
                if template_kind != "small":
                    return None
                if active_feeder_auto_tc is not True and resolve_circle_profile_from_name(active_parts_name) is None:
                    return None

                nominal_radius = small_nominal if small_nominal is not None else 54.0
                ring_tone_threshold = crop_mean - max(8.0, crop_std * 0.35)
                best_candidate = None
                local_xc = xc_cross - x_start
                local_yc = yc_cross - y_start

                for low, high in small_search_ranges:
                    for trial_radius in np.arange(float(low), float(high) + 0.1, 1.0):
                        ring_error = np.abs(np.sqrt((xx - local_xc) ** 2 + (yy - local_yc) ** 2) - trial_radius)
                        ring_band_mask = (
                            camera_mask &
                            is_gray_local &
                            is_not_cyan &
                            (~dilated_ui) &
                            (ring_error <= max(4.0, min(8.0, trial_radius * 0.10))) &
                            ((mean_val <= ring_tone_threshold) | contrast_edge_mask)
                        )
                        y_band, x_band = np.where(ring_band_mask)
                        if len(x_band) < 20:
                            continue

                        x_band_full = x_band.astype(float) + x_start
                        y_band_full = y_band.astype(float) + y_start
                        fit_threshold = max(3.0, min(7.0, trial_radius * 0.10))
                        fitted_circle = fit_circle_ransac_unanchored(
                            x_band_full,
                            y_band_full,
                            small_search_ranges,
                            max_iterations=800,
                            threshold=fit_threshold,
                            min_inliers=20,
                        )
                        if fitted_circle is None and len(x_band_full) >= 3:
                            fitted_circle = fit_circle_algebraic(x_band_full, y_band_full)
                        if fitted_circle is None:
                            continue
                        if not any(low <= fitted_circle[2] <= high for low, high in small_search_ranges):
                            continue

                        candidate = build_circle_candidate(
                            fitted_circle,
                            x_band_full,
                            y_band_full,
                            nominal_radius,
                            "small",
                        )
                        if candidate is None:
                            continue
                        candidate["source"] = "crosshair_small_hough"

                        if best_candidate is None or candidate["sort_key"] < best_candidate["sort_key"]:
                            best_candidate = candidate

                return best_candidate

            def detect_profiled_small_raw_hough_candidate():
                if active_row == -1 or template_kind != "small":
                    return None
                if dialog_present:
                    return None

                gray_raw = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                dialog_box = gray_raw[crop_px(75):crop_px(165), crop_px(95):crop_px(230)]
                if (
                    dialog_box.size and
                    np.count_nonzero((dialog_box >= 90) & (dialog_box < 180)) > 300 and
                    np.count_nonzero(dialog_box > 230) > 9000
                ):
                    return None
                blur = cv2.medianBlur(gray_raw, 5)
                edge_mask_raw = cv2.Canny(blur, 30, 90)
                edge_mask_raw[~camera_mask] = 0
                edge_y, edge_x = np.where(edge_mask_raw > 0)
                if len(edge_x) < 40:
                    return None

                edge_x_full = edge_x.astype(float) + x_start
                edge_y_full = edge_y.astype(float) + y_start
                min_radius = int(max(1, min(low for low, _ in small_search_ranges)))
                max_radius = int(max(high for _, high in small_search_ranges))
                best_candidate = None
                best_key = None
                bright_upper_small = crop_mean >= 230.0 and 12.0 <= crop_std <= 23.0 and crop_p5 < 220.0

                for param1, param2 in ((30, 10), (50, 12), (80, 14)):
                    circles = cv2.HoughCircles(
                        blur,
                        cv2.HOUGH_GRADIENT,
                        dp=1.2,
                        minDist=30,
                        param1=param1,
                        param2=param2,
                        minRadius=min_radius,
                        maxRadius=max_radius,
                    )
                    if circles is None:
                        continue
                    for cx_local, cy_local, radius in circles[0]:
                        cx_full = float(cx_local) + x_start
                        cy_full = float(cy_local) + y_start
                        radius = float(radius)
                        center_distance = float(np.hypot(cx_full - xc_cross, cy_full - yc_cross))
                        upper_off_crosshair = (
                            bright_upper_small and
                            abs(cx_full - xc_cross) <= 30.0 and
                            cy_full < yc_cross
                        )
                        if not upper_off_crosshair and center_distance > max(24.0, radius * 0.55):
                            continue

                        residuals = np.abs(np.hypot(edge_x_full - cx_full, edge_y_full - cy_full) - radius)
                        support_mask = residuals <= max(5.0, radius * 0.12)
                        if np.count_nonzero(support_mask) < 20:
                            continue

                        support_x = edge_x_full[support_mask]
                        support_y = edge_y_full[support_mask]
                        metrics = get_circle_support_metrics(support_x, support_y, cx_full, cy_full, radius)
                        if metrics["coverage_bins"] < 4 or metrics["residual_mean"] > 6.0 or metrics["residual_p90"] > 11.0:
                            continue

                        radius_bias = abs(radius - small_nominal) if small_nominal is not None else 0.0
                        if upper_off_crosshair:
                            expected_top_y = yc_cross - max(96.0, radius * 2.0)
                            key = (
                                0,
                                abs((cy_full - radius) - expected_top_y),
                                radius_bias,
                                abs(cx_full - xc_cross),
                                -metrics["coverage_bins"],
                                metrics["residual_mean"],
                                -metrics["support_count"],
                                param2,
                            )
                        else:
                            key = (
                                1,
                                radius_bias,
                                center_distance,
                                -metrics["coverage_bins"],
                                metrics["residual_mean"],
                                -metrics["support_count"],
                                param2,
                            )
                        if best_key is None or key < best_key:
                            best_key = key
                            best_candidate = {
                                "circle": (cx_full, cy_full, radius),
                                "support_x": support_x,
                                "support_y": support_y,
                                "metrics": metrics,
                                "sort_key": key,
                                "source": "profiled_small_raw_hough",
                            }
                    if best_candidate is not None:
                        return best_candidate

                return best_candidate

            def detect_large_raw_hough_candidate():
                if template_kind != "large":
                    return None

                gray_raw = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                blur = cv2.medianBlur(gray_raw, 5)
                edge_mask_raw = cv2.Canny(blur, 25, 90)
                edge_mask_raw[~camera_mask] = 0
                edge_y, edge_x = np.where(edge_mask_raw > 0)
                if len(edge_x) < 80:
                    return None

                edge_x_full = edge_x.astype(float) + x_start
                edge_y_full = edge_y.astype(float) + y_start
                min_radius = int(max(1, min(low for low, _ in large_search_ranges)))
                max_radius = int(max(high for _, high in large_search_ranges))
                best_candidate = None
                best_key = None

                for param1, param2 in ((20, 6), (30, 8), (50, 10)):
                    circles = cv2.HoughCircles(
                        blur,
                        cv2.HOUGH_GRADIENT,
                        dp=1.2,
                        minDist=40,
                        param1=param1,
                        param2=param2,
                        minRadius=min_radius,
                        maxRadius=max_radius,
                    )
                    if circles is None:
                        continue
                    for cx_local, cy_local, radius in circles[0]:
                        cx_full = float(cx_local) + x_start
                        cy_full = float(cy_local) + y_start
                        radius = float(radius)
                        residuals = np.abs(np.hypot(edge_x_full - cx_full, edge_y_full - cy_full) - radius)
                        support_mask = residuals <= max(6.0, radius * 0.05)
                        if np.count_nonzero(support_mask) < 40:
                            continue
                        support_x = edge_x_full[support_mask]
                        support_y = edge_y_full[support_mask]
                        metrics = get_circle_support_metrics(support_x, support_y, cx_full, cy_full, radius)
                        if metrics["coverage_bins"] < 3 or metrics["residual_mean"] > 8.0:
                            continue
                        visible_candidates = [
                            (cx_full, cy_full - radius),
                            (cx_full, cy_full + radius),
                            (cx_full - radius, cy_full),
                            (cx_full + radius, cy_full),
                        ]
                        if not any(0 <= x < arr.shape[1] and 0 <= y < arr.shape[0] for x, y in visible_candidates):
                            continue
                        key = (
                            abs(radius - large_nominal),
                            -metrics["coverage_bins"],
                            metrics["residual_mean"],
                            -metrics["support_count"],
                            param2,
                        )
                        if best_key is None or key < best_key:
                            best_key = key
                            best_candidate = {
                                "circle": (cx_full, cy_full, radius),
                                "support_x": support_x,
                                "support_y": support_y,
                                "metrics": metrics,
                                "sort_key": key,
                                "source": "large_raw_hough",
                            }
                    if best_candidate is not None:
                        return best_candidate

                return best_candidate

            def detect_opencv_circle_candidate(target_kind):
                search_ranges = small_search_ranges if target_kind == "small" else large_search_ranges
                nominal_radius = small_nominal if target_kind == "small" else large_nominal
                medium_candidate = target_kind == "small" and template_kind == "medium"
                bright_faint_candidate = template_kind == "small" and crop_p5 >= 230.0 and crop_std < 18.0
                centered_bright_candidate = template_kind == "small" and crop_mean >= 220.0 and center_object_present
                bright_edge_candidate = (
                    target_kind == "small" and
                    template_kind == "small" and
                    crop_mean >= 220.0 and
                    not center_object_present and
                    not dialog_present and
                    crop_p5 >= 195.0 and
                    crop_std >= 18.0
                )
                overlay_candidate = (
                    target_kind == "small" and
                    template_kind == "small" and
                    has_label_overlay
                )
                profiled_small_candidate = (
                    target_kind == "small" and
                    template_kind == "small" and
                    active_row != -1 and
                    active_feeder_auto_tc is True
                )
                if (
                    target_kind == "large" or
                    (has_label_overlay and not overlay_candidate) or
                    (
                        template_kind != "medium" and
                        not bright_faint_candidate and
                        not centered_bright_candidate and
                        not bright_edge_candidate and
                        not overlay_candidate and
                        not profiled_small_candidate
                    )
                ):
                    return None

                valid_mask = camera_mask & is_gray_local & is_not_cyan & (~dilated_ui)
                gray_u8 = np.clip(mean_val, 0, 255).astype(np.uint8)
                fill_value = int(np.clip(crop_mean, 0, 255))
                work = gray_u8.copy()
                work[~valid_mask] = fill_value

                blur = cv2.GaussianBlur(work, (3, 3), 0)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                equalized = clahe.apply(blur)

                edge_mask_cv = np.zeros_like(work, dtype=np.uint8)
                for low, high in ((10, 32), (18, 54), (28, 84)):
                    edge_mask_cv = cv2.bitwise_or(edge_mask_cv, cv2.Canny(blur, low, high))
                    edge_mask_cv = cv2.bitwise_or(edge_mask_cv, cv2.Canny(equalized, low, high))

                edge_mask_cv = cv2.bitwise_or(edge_mask_cv, (contrast_edge_mask.astype(np.uint8) * 255))
                edge_mask_cv[~valid_mask] = 0

                cross_x_local = int(round(xc_cross - x_start))
                cross_y_local = int(round(yc_cross - y_start))
                if 0 <= cross_x_local < edge_mask_cv.shape[1]:
                    edge_mask_cv[:, max(0, cross_x_local - 1):min(edge_mask_cv.shape[1], cross_x_local + 2)] = 0
                if 0 <= cross_y_local < edge_mask_cv.shape[0]:
                    edge_mask_cv[max(0, cross_y_local - 1):min(edge_mask_cv.shape[0], cross_y_local + 2), :] = 0

                contours, _ = cv2.findContours(edge_mask_cv, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                contour_points = []
                for contour in contours:
                    pts = contour.reshape(-1, 2)
                    if len(pts) >= 2:
                        contour_points.append(pts)
                if not contour_points:
                    return None

                all_points = np.vstack(contour_points)
                if len(all_points) > 5000:
                    sample_idx = np.linspace(0, len(all_points) - 1, 5000).astype(int)
                    all_points = all_points[sample_idx]

                edge_x_local = all_points[:, 0].astype(float)
                edge_y_local = all_points[:, 1].astype(float)
                edge_x_full = edge_x_local + x_start
                edge_y_full = edge_y_local + y_start

                best_hough_candidate = None
                best_hough_key = None
                if bright_faint_candidate or bright_edge_candidate or overlay_candidate or medium_candidate or profiled_small_candidate:
                    hough_median = cv2.medianBlur(work, 5)
                    hough_equalized = cv2.equalizeHist(hough_median)
                    hough_inputs = (
                        (hough_median, 0, 12),
                        (255 - hough_median, 0, 12),
                        (hough_equalized, 1, 14),
                        (equalized, 2, 14),
                    )
                    min_radius = int(max(1, min(low for low, _ in search_ranges)))
                    max_radius = int(max(high for _, high in search_ranges))
                    for hough_image, source_priority, param2 in hough_inputs:
                        circles = cv2.HoughCircles(
                            hough_image,
                            cv2.HOUGH_GRADIENT,
                            dp=1.2,
                            minDist=55,
                            param1=35,
                            param2=param2,
                            minRadius=min_radius,
                            maxRadius=max_radius,
                        )
                        if circles is None:
                            continue
                        for cx_seed_local, cy_seed_local, radius_seed in circles[0]:
                            cx_seed = float(cx_seed_local + x_start)
                            cy_seed = float(cy_seed_local + y_start)
                            radius_seed = float(radius_seed)
                            if not any(low <= radius_seed <= high for low, high in search_ranges):
                                continue

                            seed_residuals = np.abs(
                                np.sqrt((edge_x_full - cx_seed) ** 2 + (edge_y_full - cy_seed) ** 2) - radius_seed
                            )
                            seed_support = seed_residuals <= max(4.0, radius_seed * 0.14)
                            if np.count_nonzero(seed_support) < 8:
                                continue

                            support_x = edge_x_full[seed_support]
                            support_y = edge_y_full[seed_support]
                            candidate = build_opencv_circle_candidate(
                                (cx_seed, cy_seed, radius_seed),
                                support_x,
                                support_y,
                                nominal_radius,
                                target_kind,
                            )
                            if candidate is None:
                                continue

                            center_distance = float(np.hypot(cx_seed - xc_cross, cy_seed - yc_cross))
                            radius_bias = abs(radius_seed - nominal_radius) if nominal_radius is not None else 0.0
                            if medium_candidate:
                                hough_key = (source_priority,) + tuple(candidate["sort_key"])
                            elif overlay_candidate or bright_edge_candidate:
                                hough_key = (
                                    abs(cx_seed - xc_cross),
                                    abs((cy_seed + radius_seed) - yc_cross),
                                    radius_bias,
                                    source_priority,
                                    center_distance,
                                    -candidate["metrics"]["coverage_bins"],
                                    candidate["metrics"]["largest_gap_bins"],
                                    -candidate["metrics"]["support_count"],
                                )
                            else:
                                upper_priority = 0 if cy_seed <= yc_cross - max(14.0, radius_seed * 0.25) else 1
                                hough_key = (
                                    upper_priority,
                                    source_priority,
                                    cy_seed if upper_priority == 0 else center_distance,
                                    radius_bias,
                                    center_distance,
                                    -candidate["metrics"]["coverage_bins"],
                                    candidate["metrics"]["largest_gap_bins"],
                                    -candidate["metrics"]["support_count"],
                                )
                            if best_hough_key is None or hough_key < best_hough_key:
                                best_hough_key = hough_key
                                best_hough_candidate = candidate

                if (
                    bright_faint_candidate or
                    bright_edge_candidate or
                    overlay_candidate or
                    medium_candidate or
                    profiled_small_candidate
                ) and best_hough_candidate is not None:
                    return best_hough_candidate

                best_candidate = None
                for pts in contour_points:
                    if len(pts) < 8:
                        continue
                    x_contour = pts[:, 0].astype(float) + x_start
                    y_contour = pts[:, 1].astype(float) + y_start
                    contour_fit = fit_circle_ransac_unanchored(
                        x_contour,
                        y_contour,
                        search_ranges,
                        max_iterations=350,
                        threshold=3.5,
                        min_inliers=min(8, max(3, len(x_contour) // 3)),
                    )
                    best_candidate = choose_candidate(
                        best_candidate,
                        build_opencv_circle_candidate(
                            contour_fit,
                            x_contour,
                            y_contour,
                            nominal_radius,
                            target_kind,
                        ),
                    )

                grad_x_cv = cv2.Sobel(blur.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
                grad_y_cv = cv2.Sobel(blur.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
                ix = np.clip(edge_x_local.astype(int), 0, blur.shape[1] - 1)
                iy = np.clip(edge_y_local.astype(int), 0, blur.shape[0] - 1)
                gx = grad_x_cv[iy, ix]
                gy = grad_y_cv[iy, ix]
                mag = np.sqrt(gx ** 2 + gy ** 2)
                grad_valid = mag > max(4.0, np.percentile(mag, 35) if len(mag) else 4.0)
                if np.count_nonzero(grad_valid) < 8:
                    return best_candidate

                edge_x_vote = edge_x_local[grad_valid]
                edge_y_vote = edge_y_local[grad_valid]
                gx = gx[grad_valid] / mag[grad_valid]
                gy = gy[grad_valid] / mag[grad_valid]

                low_radius = min(low for low, _ in search_ranges)
                high_radius = max(high for _, high in search_ranges)
                if nominal_radius is not None:
                    low_radius = max(low_radius, nominal_radius - 10)
                    high_radius = min(high_radius, nominal_radius + 10)
                radii = np.arange(low_radius, high_radius + 0.1, 1.5)
                if len(radii) == 0:
                    return best_candidate

                bin_size = 2
                h_bins = int(np.ceil(edge_mask_cv.shape[0] / bin_size))
                w_bins = int(np.ceil(edge_mask_cv.shape[1] / bin_size))
                accumulator = np.zeros((len(radii), h_bins, w_bins), dtype=np.uint16)

                for radius_idx, radius in enumerate(radii):
                    for sign in (-1.0, 1.0):
                        center_x = edge_x_vote + sign * gx * radius
                        center_y = edge_y_vote + sign * gy * radius
                        valid_centers = (
                            (center_x >= 0) & (center_x < edge_mask_cv.shape[1]) &
                            (center_y >= 0) & (center_y < edge_mask_cv.shape[0])
                        )
                        if not np.any(valid_centers):
                            continue
                        x_bins = np.floor(center_x[valid_centers] / bin_size).astype(int)
                        y_bins = np.floor(center_y[valid_centers] / bin_size).astype(int)
                        np.add.at(accumulator[radius_idx], (y_bins, x_bins), 1)

                flat_votes = accumulator.ravel()
                if flat_votes.size == 0 or int(flat_votes.max()) < 4:
                    return best_candidate

                top_count = min(80, flat_votes.size)
                top_indices = np.argpartition(flat_votes, -top_count)[-top_count:]
                top_indices = top_indices[np.argsort(flat_votes[top_indices])[::-1]]
                seen_seeds = []

                for flat_idx in top_indices:
                    votes = int(flat_votes[flat_idx])
                    if votes < 4:
                        break
                    radius_idx, y_bin, x_bin = np.unravel_index(flat_idx, accumulator.shape)
                    radius_seed = float(radii[radius_idx])
                    cx_seed = (x_bin + 0.5) * bin_size + x_start
                    cy_seed = (y_bin + 0.5) * bin_size + y_start

                    duplicate = False
                    for prev_x, prev_y, prev_r in seen_seeds:
                        if np.hypot(cx_seed - prev_x, cy_seed - prev_y) < 5.0 and abs(radius_seed - prev_r) < 3.0:
                            duplicate = True
                            break
                    if duplicate:
                        continue
                    seen_seeds.append((cx_seed, cy_seed, radius_seed))

                    seed_residuals = np.abs(
                        np.sqrt((edge_x_full - cx_seed) ** 2 + (edge_y_full - cy_seed) ** 2) - radius_seed
                    )
                    seed_support = seed_residuals <= max(4.0, radius_seed * 0.12)
                    if np.count_nonzero(seed_support) < 8:
                        continue

                    support_x = edge_x_full[seed_support]
                    support_y = edge_y_full[seed_support]
                    refined_fit = fit_circle_ransac_unanchored(
                        support_x,
                        support_y,
                        search_ranges,
                        max_iterations=600,
                        threshold=max(3.0, radius_seed * 0.08),
                        min_inliers=8,
                    )
                    if refined_fit is None:
                        try:
                            refined_fit = fit_circle_algebraic(support_x, support_y)
                        except Exception:
                            refined_fit = None
                    if refined_fit is None:
                        continue
                    if np.hypot(refined_fit[0] - cx_seed, refined_fit[1] - cy_seed) > 18.0:
                        continue
                    if abs(refined_fit[2] - radius_seed) > 10.0:
                        continue

                    best_candidate = choose_candidate(
                        best_candidate,
                        build_opencv_circle_candidate(
                            refined_fit,
                            support_x,
                            support_y,
                            nominal_radius,
                            target_kind,
                        ),
                    )

                return best_candidate

            def infer_small_anchor_modes(x_pts, y_pts):
                if len(x_pts) == 0:
                    return []

                dx = float(np.mean(x_pts) - xc_cross)
                dy = float(np.mean(y_pts) - yc_cross)
                modes = []

                if dy < -10.0:
                    modes.append("bottom")
                elif dy > 10.0:
                    modes.append("top")

                if dx < -10.0:
                    modes.append("right")
                elif dx > 10.0:
                    modes.append("left")

                if len(modes) == 2:
                    if abs(dx) > abs(dy) * 1.4:
                        return [modes[1]]
                    if abs(dy) > abs(dx) * 1.4:
                        return [modes[0]]
                return modes

            def fit_circle_for_mode(x_pts, y_pts, search_ranges, nominal_radius, target_kind):
                if len(x_pts) < 15:
                    return None
                medium_mode = target_kind == "small" and template_kind == "medium"
                if medium_mode:
                    circle_fit = fit_circle_ransac_unanchored(
                        x_pts,
                        y_pts,
                        search_ranges,
                        max_iterations=1400,
                        threshold=3.5,
                        min_inliers=10,
                    )
                    if circle_fit is None:
                        return None

                    metrics = get_circle_support_metrics(
                        x_pts,
                        y_pts,
                        circle_fit[0],
                        circle_fit[1],
                        circle_fit[2],
                    )
                    radius_bias = abs(circle_fit[2] - nominal_radius) if nominal_radius is not None else 0.0
                    center_distance = float(np.hypot(circle_fit[0] - xc_cross, circle_fit[1] - yc_cross))
                    fit_key = (
                        radius_bias,
                        -metrics["coverage_bins"],
                        metrics["angular_balance"],
                        -metrics["cardinal_support"],
                        metrics["largest_gap_bins"],
                        metrics["residual_mean"],
                        metrics["residual_p90"],
                        -metrics["support_count"],
                        center_distance,
                    )

                    refined_fit = None
                    try:
                        refined_fit = fit_circle_algebraic(x_pts, y_pts)
                    except Exception:
                        refined_fit = None

                    if (
                        refined_fit is not None and
                        any(low <= refined_fit[2] <= high for low, high in search_ranges)
                    ):
                        refined_metrics = get_circle_support_metrics(
                            x_pts,
                            y_pts,
                            refined_fit[0],
                            refined_fit[1],
                            refined_fit[2],
                        )
                        refined_radius_bias = abs(refined_fit[2] - nominal_radius) if nominal_radius is not None else 0.0
                        refined_center_distance = float(np.hypot(refined_fit[0] - xc_cross, refined_fit[1] - yc_cross))
                        refined_key = (
                            refined_radius_bias,
                            -refined_metrics["coverage_bins"],
                            refined_metrics["angular_balance"],
                            -refined_metrics["cardinal_support"],
                            refined_metrics["largest_gap_bins"],
                            refined_metrics["residual_mean"],
                            refined_metrics["residual_p90"],
                            -refined_metrics["support_count"],
                            refined_center_distance,
                        )
                        if refined_key < fit_key:
                            return refined_fit

                    return circle_fit

                anchor_modes = [None]
                if target_kind == "small" and not medium_mode:
                    inferred_modes = infer_small_anchor_modes(x_pts, y_pts)
                    anchor_modes = inferred_modes + [None] if inferred_modes else [None]

                circle_fit = None
                best_fit_key = None
                for anchor_mode in anchor_modes:
                    fit_candidate = fit_circle_ransac(
                        x_pts,
                        y_pts,
                        xc_cross,
                        yc_cross,
                        allowed_ranges=search_ranges,
                        center_mode=large_center_mode if target_kind == "large" else None,
                        anchor_mode=anchor_mode,
                        preferred_radius=nominal_radius,
                    )
                    if fit_candidate is None:
                        continue

                    metrics = get_circle_support_metrics(
                        x_pts,
                        y_pts,
                        fit_candidate[0],
                        fit_candidate[1],
                        fit_candidate[2],
                    )
                    if medium_mode:
                        radius_bias = abs(fit_candidate[2] - nominal_radius) if nominal_radius is not None else 0.0
                        fit_key = (
                            radius_bias,
                            -metrics["coverage_bins"],
                            -metrics["cardinal_support"],
                            metrics["largest_gap_bins"],
                            metrics["residual_mean"],
                            metrics["residual_p90"],
                            -metrics["support_count"],
                        )
                    else:
                        anchor_error = get_circle_anchor_error(
                            fit_candidate[0],
                            fit_candidate[1],
                            fit_candidate[2],
                            xc_cross,
                            yc_cross,
                            anchor_mode=anchor_mode,
                        )
                        fit_key = (
                            metrics["residual_mean"],
                            metrics["residual_p90"],
                            anchor_error,
                            -metrics["support_count"],
                        )
                    if best_fit_key is None or fit_key < best_fit_key:
                        best_fit_key = fit_key
                        circle_fit = fit_candidate

                if circle_fit is None and target_kind == "large" and large_center_mode is not None:
                    circle_fit = fit_circle_ransac(
                        x_pts,
                        y_pts,
                        xc_cross,
                        yc_cross,
                        allowed_ranges=search_ranges,
                        center_mode=None,
                        preferred_radius=nominal_radius,
                    )
                return circle_fit

            def build_circle_candidate(circle_fit, support_x, support_y, nominal_radius, target_kind):
                if circle_fit is None or support_x is None or support_y is None or len(support_x) == 0:
                    return None

                xc_fit, yc_fit, R_fit = circle_fit
                if is_bezel(xc_fit, yc_fit, R_fit, xc_cross, yc_cross):
                    return None

                metrics = get_circle_support_metrics(support_x, support_y, xc_fit, yc_fit, R_fit)
                radius_bias = abs(R_fit - nominal_radius) if nominal_radius is not None else 0.0
                if target_kind == "small":
                    inferred_small_profile = infer_circle_session_suffix(template_match, active_parts_name, R_fit)
                    medium_mode = template_kind == "medium"
                    boundary_margin = min(
                        xc_fit - R_fit - x_start,
                        x_end - (xc_fit + R_fit),
                        yc_fit - R_fit - y_start,
                        y_end - (yc_fit + R_fit),
                    )
                    local_xc = xc_fit - x_start
                    local_yc = yc_fit - y_start
                    interior_mask = (
                        ((xx - local_xc) ** 2 + (yy - local_yc) ** 2) <= max(3.0, R_fit * 0.65) ** 2
                    ) & camera_mask
                    interior_mean = float(np.mean(mean_val[interior_mask])) if np.count_nonzero(interior_mask) >= 20 else 0.0
                    center_distance = float(np.hypot(xc_fit - xc_cross, yc_fit - yc_cross))
                    circumference_error = abs(center_distance - R_fit)
                    sector_ratio = metrics["sector_peak"] / max(metrics["support_count"], 1)
                    inferred_modes = infer_small_anchor_modes(support_x, support_y)
                    if inferred_modes:
                        anchor_error = min(
                            get_circle_anchor_error(
                                xc_fit,
                                yc_fit,
                                R_fit,
                                xc_cross,
                                yc_cross,
                                anchor_mode=anchor_mode,
                            )
                            for anchor_mode in inferred_modes
                        )
                    else:
                        anchor_error = get_circle_anchor_error(
                            xc_fit,
                            yc_fit,
                            R_fit,
                            xc_cross,
                            yc_cross,
                            anchor_mode=None,
                        )
                    if metrics["support_count"] < 15:
                        return None
                    if metrics["residual_mean"] > 4.5 or metrics["residual_p90"] > 8.0:
                        return None
                    if metrics["coverage_bins"] < 3 and metrics["support_count"] < 26:
                        return None
                    if not medium_mode and anchor_error > 80.0 ** 2:
                        return None
                    medium_visibility_profile = get_medium_visibility_profile(xc_fit, yc_fit, R_fit) if medium_mode else None
                    medium_ring_profile = get_medium_ring_evidence_profile(xc_fit, yc_fit, R_fit) if medium_mode else None
                    if medium_mode:
                        if not is_valid_medium_candidate(metrics, medium_visibility_profile, medium_ring_profile):
                            return None
                    elif has_label_overlay:
                        if metrics["support_count"] < 12:
                            return None
                        if anchor_error > 65.0 ** 2:
                            return None
                    else:
                        if metrics["cardinal_support"] == 0 and metrics["coverage_bins"] < 5:
                            return None
                        if boundary_margin >= 8.0:
                            if metrics["cardinal_support"] < 2:
                                return None
                            if metrics["largest_gap_bins"] > 14:
                                return None
                            if metrics["min_span_ratio"] < 0.62:
                                return None
                            if sector_ratio > 0.52:
                                return None
                        if (
                            not bright_edge_mode and
                            circumference_error > max(18.0, R_fit * 0.4)
                        ):
                            return None
                    boundary_penalty = max(0.0, 8.0 - boundary_margin)
                    if medium_mode:
                        sort_key = get_medium_outer_ring_sort_key(
                            metrics,
                            R_fit,
                            nominal_radius,
                            boundary_margin,
                            metrics["residual_mean"],
                            metrics["residual_p90"],
                            circumference_error,
                            center_distance,
                            medium_visibility_profile,
                            medium_ring_profile,
                            interior_mean,
                        )
                    elif has_label_overlay:
                        sort_key = (
                            anchor_error,
                            radius_bias,
                            -metrics["cardinal_support"],
                            metrics["largest_gap_bins"],
                            -metrics["coverage_bins"],
                            -metrics["min_span_ratio"],
                            sector_ratio,
                            metrics["residual_mean"],
                            metrics["residual_p90"],
                            boundary_penalty,
                            -metrics["support_count"],
                        )
                    elif bright_edge_mode:
                        sort_key = (
                            radius_bias,
                            -metrics["support_count"],
                            -metrics["coverage_bins"],
                            metrics["largest_gap_bins"],
                            -metrics["min_span_ratio"],
                            metrics["residual_mean"],
                            metrics["residual_p90"],
                            anchor_error,
                            max(0.0, -boundary_margin),
                            sector_ratio,
                        )
                    else:
                        interior_priority = 0 if boundary_margin >= 8.0 else 1
                        clipped_priority = 0 if boundary_margin >= 0.0 else 1
                        sort_key = (
                            interior_priority,
                            clipped_priority,
                            radius_bias,
                            -metrics["cardinal_support"],
                            metrics["largest_gap_bins"],
                            -metrics["coverage_bins"],
                            sector_ratio,
                            metrics["residual_mean"],
                            metrics["residual_p90"],
                            boundary_penalty,
                            anchor_error,
                            -metrics["support_count"],
                        )
                else:
                    if metrics["support_count"] < 15 or metrics["coverage_bins"] < 2:
                        return None
                    if metrics["residual_mean"] > 4.5 or metrics["residual_p90"] > 7.5:
                        return None
                    sort_key = (
                        metrics["residual_mean"],
                        metrics["residual_p90"],
                        get_large_alignment_error(xc_fit, yc_fit, R_fit),
                        radius_bias,
                        -metrics["support_count"],
                        -metrics["coverage_bins"],
                    )

                return {
                    "circle": circle_fit,
                    "support_x": support_x,
                    "support_y": support_y,
                    "metrics": metrics,
                    "sort_key": sort_key,
                }

            def refine_medium_ring_candidate(candidate):
                if candidate is None or template_kind != "medium":
                    return None

                xc_fit, yc_fit, R_fit = candidate["circle"]
                local_xc = xc_fit - x_start
                local_yc = yc_fit - y_start
                ring_tolerance = max(5.0, min(9.0, R_fit * 0.12))
                ring_error = np.abs(np.sqrt((xx - local_xc) ** 2 + (yy - local_yc) ** 2) - R_fit)
                ring_tone_threshold = crop_mean - max(8.0, crop_std * 0.35)
                ring_band_mask = (
                    camera_mask &
                    is_gray_local &
                    is_not_cyan &
                    (~dilated_ui) &
                    (ring_error <= ring_tolerance) &
                    ((mean_val <= ring_tone_threshold) | contrast_edge_mask)
                )

                y_band, x_band = np.where(ring_band_mask)
                if len(x_band) < 40:
                    return None

                x_band_full = x_band.astype(float) + x_start
                y_band_full = y_band.astype(float) + y_start
                refined_ranges = small_search_ranges
                if small_nominal is not None:
                    tight_ranges = []
                    for low, high in small_search_ranges:
                        tight_low = max(low, small_nominal - 8)
                        tight_high = min(high, small_nominal + 8)
                        if tight_low <= tight_high:
                            tight_ranges.append((tight_low, tight_high))
                    if tight_ranges:
                        refined_ranges = tight_ranges

                refined_fit = fit_circle_ransac_unanchored(
                    x_band_full,
                    y_band_full,
                    refined_ranges,
                    max_iterations=1200,
                    threshold=max(3.0, R_fit * 0.07),
                    min_inliers=12,
                )
                if refined_fit is None:
                    return None

                return build_circle_candidate(
                    refined_fit,
                    x_band_full,
                    y_band_full,
                    small_nominal,
                    "small",
                )

            def refine_medium_cardinal_geometry(candidate):
                if candidate is None or template_kind != "medium":
                    return None

                xc_fit, yc_fit, R_fit = candidate["circle"]
                local_xc = xc_fit - x_start
                local_yc = yc_fit - y_start
                ring_tolerance = max(4.0, min(8.0, R_fit * 0.10))
                ring_tone_threshold = crop_mean - max(8.0, crop_std * 0.35)
                ring_error = np.abs(np.sqrt((xx - local_xc) ** 2 + (yy - local_yc) ** 2) - R_fit)
                ring_band_mask = (
                    camera_mask &
                    is_gray_local &
                    is_not_cyan &
                    (~dilated_ui) &
                    (ring_error <= ring_tolerance) &
                    ((mean_val <= ring_tone_threshold) | contrast_edge_mask)
                )

                y_band, x_band = np.where(ring_band_mask)
                if len(x_band) < 32:
                    return None

                x_full = x_band.astype(float) + x_start
                y_full = y_band.astype(float) + y_start
                angles = np.mod(np.arctan2(y_full - yc_fit, x_full - xc_fit), 2.0 * np.pi)
                sector_tolerance = np.deg2rad(28.0 if R_fit < 120 else 22.0)

                def sector_point(target_angle, axis_name):
                    sector_mask = angular_distance(angles, target_angle) <= sector_tolerance
                    if np.count_nonzero(sector_mask) < 6:
                        return None
                    xs = x_full[sector_mask]
                    ys = y_full[sector_mask]
                    if axis_name == "top":
                        return (float(np.median(xs)), float(np.percentile(ys, 14)))
                    if axis_name == "bottom":
                        return (float(np.median(xs)), float(np.percentile(ys, 86)))
                    if axis_name == "left":
                        return (float(np.percentile(xs, 14)), float(np.median(ys)))
                    return (float(np.percentile(xs, 86)), float(np.median(ys)))

                top_pt = sector_point(3.0 * np.pi / 2.0, "top")
                bottom_pt = sector_point(np.pi / 2.0, "bottom")
                left_pt = sector_point(np.pi, "left")
                right_pt = sector_point(0.0, "right")

                x_estimates = []
                y_estimates = []
                r_estimates = []

                radius_lr = None
                radius_tb = None
                if left_pt is not None and right_pt is not None:
                    xc_lr = 0.5 * (left_pt[0] + right_pt[0])
                    yc_lr = 0.5 * (left_pt[1] + right_pt[1])
                    radius_lr = 0.5 * (right_pt[0] - left_pt[0])
                    if radius_lr > 0:
                        x_estimates.append(xc_lr)
                        y_estimates.append(yc_lr)
                        r_estimates.append(radius_lr)

                if top_pt is not None and bottom_pt is not None:
                    xc_tb = 0.5 * (top_pt[0] + bottom_pt[0])
                    yc_tb = 0.5 * (top_pt[1] + bottom_pt[1])
                    radius_tb = 0.5 * (bottom_pt[1] - top_pt[1])
                    if radius_tb > 0:
                        x_estimates.append(xc_tb)
                        y_estimates.append(yc_tb)
                        r_estimates.append(radius_tb)

                if radius_lr is not None:
                    if top_pt is not None:
                        y_estimates.append(top_pt[1] + radius_lr)
                    if bottom_pt is not None:
                        y_estimates.append(bottom_pt[1] - radius_lr)
                if radius_tb is not None:
                    if left_pt is not None:
                        x_estimates.append(left_pt[0] + radius_tb)
                    if right_pt is not None:
                        x_estimates.append(right_pt[0] - radius_tb)

                if not r_estimates:
                    return None

                xc_refined = float(np.mean(x_estimates)) if x_estimates else xc_fit
                yc_refined = float(np.mean(y_estimates)) if y_estimates else yc_fit
                R_refined = float(np.mean(r_estimates))
                if not any(low <= R_refined <= high for low, high in small_search_ranges):
                    return None

                return build_circle_candidate(
                    (xc_refined, yc_refined, R_refined),
                    x_full,
                    y_full,
                    small_nominal,
                    "small",
                )

            def refine_medium_axis_geometry(candidate):
                if candidate is None or template_kind != "medium":
                    return None

                xc_fit, yc_fit, R_fit = candidate["circle"]
                local_xc = xc_fit - x_start
                local_yc = yc_fit - y_start
                ring_tone_threshold = crop_mean - max(8.0, crop_std * 0.35)
                scan_pad = int(round(max(10.0, min(18.0, R_fit * 0.22))))
                axis_half_width = int(round(max(8.0, min(18.0, R_fit * 0.18))))
                ring_mask = (
                    camera_mask &
                    is_gray_local &
                    is_not_cyan &
                    (~dilated_ui) &
                    ((mean_val <= ring_tone_threshold) | contrast_edge_mask)
                )

                def score_pixels(xs, ys):
                    if len(xs) == 0:
                        return None
                    intensities = mean_val[ys, xs]
                    edges = contrast_edge_mask[ys, xs]
                    dark_score = float(np.sum(np.clip(ring_tone_threshold - intensities, 0.0, 255.0)))
                    edge_score = float(np.count_nonzero(edges)) * 12.0
                    return dark_score + edge_score + (2.0 * float(len(xs)))

                def find_vertical_intersection(expected_y, label):
                    best = None
                    y0 = max(0, int(round(expected_y - scan_pad)))
                    y1 = min(crop.shape[0] - 1, int(round(expected_y + scan_pad)))
                    x0 = max(0, int(round(local_xc - axis_half_width)))
                    x1 = min(crop.shape[1] - 1, int(round(local_xc + axis_half_width)))
                    for local_y in range(y0, y1 + 1):
                        xs = np.where(ring_mask[local_y, x0:x1 + 1])[0]
                        if len(xs) < 2:
                            continue
                        xs = xs + x0
                        ys = np.full(len(xs), local_y, dtype=int)
                        score = score_pixels(xs, ys)
                        if score is None:
                            continue
                        candidate_key = (-score, abs(local_y - expected_y))
                        if best is None or candidate_key < best[0]:
                            x_value = float(np.median(xs))
                            if label == "top":
                                y_value = float(np.percentile(ys, 14))
                            else:
                                y_value = float(np.percentile(ys, 86))
                            best = (candidate_key, (x_value + x_start, y_value + y_start))
                    return None if best is None else best[1]

                def find_horizontal_intersection(expected_x, label):
                    best = None
                    x0 = max(0, int(round(expected_x - scan_pad)))
                    x1 = min(crop.shape[1] - 1, int(round(expected_x + scan_pad)))
                    y0 = max(0, int(round(local_yc - axis_half_width)))
                    y1 = min(crop.shape[0] - 1, int(round(local_yc + axis_half_width)))
                    for local_x in range(x0, x1 + 1):
                        ys = np.where(ring_mask[y0:y1 + 1, local_x])[0]
                        if len(ys) < 2:
                            continue
                        ys = ys + y0
                        xs = np.full(len(ys), local_x, dtype=int)
                        score = score_pixels(xs, ys)
                        if score is None:
                            continue
                        candidate_key = (-score, abs(local_x - expected_x))
                        if best is None or candidate_key < best[0]:
                            y_value = float(np.median(ys))
                            if label == "left":
                                x_value = float(np.percentile(xs, 14))
                            else:
                                x_value = float(np.percentile(xs, 86))
                            best = (candidate_key, (x_value + x_start, y_value + y_start))
                    return None if best is None else best[1]

                top_pt = find_vertical_intersection(local_yc - R_fit, "top")
                bottom_pt = find_vertical_intersection(local_yc + R_fit, "bottom")
                left_pt = find_horizontal_intersection(local_xc - R_fit, "left")
                right_pt = find_horizontal_intersection(local_xc + R_fit, "right")

                points = [pt for pt in (top_pt, bottom_pt, left_pt, right_pt) if pt is not None]
                if len(points) < 3:
                    return None

                pts = np.array(points, dtype=float)
                try:
                    refined_fit = fit_circle_algebraic(pts[:, 0], pts[:, 1])
                except Exception:
                    return None

                if refined_fit is None:
                    return None
                if not any(low <= refined_fit[2] <= high for low, high in small_search_ranges):
                    return None

                local_refined_xc = refined_fit[0] - x_start
                local_refined_yc = refined_fit[1] - y_start
                refined_ring_tolerance = max(4.0, min(8.0, refined_fit[2] * 0.10))
                refined_ring_error = np.abs(
                    np.sqrt((xx - local_refined_xc) ** 2 + (yy - local_refined_yc) ** 2) - refined_fit[2]
                )
                support_mask = ring_mask & (refined_ring_error <= refined_ring_tolerance)
                y_support, x_support = np.where(support_mask)
                if len(x_support) < 20:
                    return None

                x_support_full = x_support.astype(float) + x_start
                y_support_full = y_support.astype(float) + y_start

                return build_circle_candidate(
                    refined_fit,
                    x_support_full,
                    y_support_full,
                    small_nominal,
                    "small",
                )

            def evaluate_component_candidates(components, search_ranges, nominal_radius, target_kind, max_cross_distance):
                best_candidate = None
                for comp_y, comp_x in components:
                    x_comp_full = comp_x + x_start
                    y_comp_full = comp_y + y_start
                    comp_dists = np.sqrt((x_comp_full - xc_cross) ** 2 + (y_comp_full - yc_cross) ** 2)
                    comp_mask = comp_dists < max_cross_distance
                    x_comp = x_comp_full[comp_mask]
                    y_comp = y_comp_full[comp_mask]
                    if len(x_comp) < 15:
                        continue
                    circle_fit = fit_circle_for_mode(x_comp, y_comp, search_ranges, nominal_radius, target_kind)
                    candidate = build_circle_candidate(circle_fit, x_comp, y_comp, nominal_radius, target_kind)
                    best_candidate = choose_candidate(best_candidate, candidate)
                return best_candidate

            def build_balanced_component_points(components, max_cross_distance, per_component_limit=20):
                x_parts = []
                y_parts = []
                for comp_y, comp_x in components:
                    x_comp_full = comp_x + x_start
                    y_comp_full = comp_y + y_start
                    comp_dists = np.sqrt((x_comp_full - xc_cross) ** 2 + (y_comp_full - yc_cross) ** 2)
                    comp_mask = comp_dists < max_cross_distance
                    x_comp = x_comp_full[comp_mask]
                    y_comp = y_comp_full[comp_mask]
                    if len(x_comp) < 3:
                        continue
                    if len(x_comp) > per_component_limit:
                        sample_idx = np.linspace(0, len(x_comp) - 1, per_component_limit).astype(int)
                        x_comp = x_comp[sample_idx]
                        y_comp = y_comp[sample_idx]
                    x_parts.append(x_comp)
                    y_parts.append(y_comp)

                if not x_parts:
                    return np.array([], dtype=int), np.array([], dtype=int)
                return np.concatenate(x_parts), np.concatenate(y_parts)

            best_small_candidate = None
            best_large_candidate = None
            best_opencv_small_candidate = None
            best_medium_hough_candidate = None
            best_profile_hough_candidate = None
            best_crosshair_small_candidate = None
            best_large_raw_hough_candidate = None
            prefer_large_without_profile = False

            best_profiled_small_raw_hough_candidate = detect_profiled_small_raw_hough_candidate()
            if best_profiled_small_raw_hough_candidate is None:
                if template_kind != "large":
                    best_opencv_small_candidate = detect_opencv_circle_candidate("small")
                best_large_raw_hough_candidate = detect_large_raw_hough_candidate()
                if template_kind == "medium" or (active_row != -1 and active_feeder_auto_tc is True):
                    best_medium_hough_candidate = detect_medium_hough_circle_candidate()
                best_profile_hough_candidate = detect_profile_hough_circle_candidate()
                best_crosshair_small_candidate = detect_crosshair_small_hough_candidate()

            if template_kind != "large" and small_anchor_mode is None:
                contrast_components = extract_connected_components(contrast_edge_mask, min_pixels=6)
                best_small_candidate = choose_candidate(
                    best_small_candidate,
                    evaluate_component_candidates(
                        contrast_components,
                        small_search_ranges,
                        small_nominal,
                        "small",
                        155,
                    ),
                )

                contrast_y, contrast_x = np.where(contrast_edge_mask)
                if len(contrast_x) >= 15 and not bright_edge_mode:
                    contrast_x_full = contrast_x + x_start
                    contrast_y_full = contrast_y + y_start
                    contrast_dists = np.sqrt((contrast_x_full - xc_cross) ** 2 + (contrast_y_full - yc_cross) ** 2)
                    contrast_fit_mask = contrast_dists < 155
                    contrast_fit_x = contrast_x_full[contrast_fit_mask]
                    contrast_fit_y = contrast_y_full[contrast_fit_mask]
                    if len(contrast_fit_x) >= 15:
                        contrast_circle = fit_circle_for_mode(
                            contrast_fit_x,
                            contrast_fit_y,
                            small_search_ranges,
                            small_nominal,
                            "small",
                        )
                        best_small_candidate = choose_candidate(
                            best_small_candidate,
                            build_circle_candidate(
                                contrast_circle,
                                contrast_fit_x,
                                contrast_fit_y,
                                small_nominal,
                                "small",
                            ),
                        )

                    balanced_x, balanced_y = build_balanced_component_points(
                        contrast_components,
                        155,
                        per_component_limit=20,
                    )
                    if len(balanced_x) >= 15:
                        balanced_circle = fit_circle_for_mode(
                            balanced_x,
                            balanced_y,
                            small_search_ranges,
                            small_nominal,
                            "small",
                        )
                        best_small_candidate = choose_candidate(
                            best_small_candidate,
                            build_circle_candidate(
                                balanced_circle,
                                contrast_fit_x,
                                contrast_fit_y,
                                small_nominal,
                                "small",
                            ),
                        )
            for thresh in thresholds:
                is_dark = masked_mean < thresh
                circle_pixels_mask = is_gray_local & is_dark & is_not_cyan
                y_indices_all, x_indices_all = np.where(circle_pixels_mask)
                if len(x_indices_all) > 40000 or len(x_indices_all) < 100:
                    continue

                padded = np.pad(circle_pixels_mask, 1, mode='constant', constant_values=False)
                eroded = (
                    padded[1:-1, 1:-1] &
                    padded[:-2, 1:-1] &
                    padded[2:, 1:-1] &
                    padded[1:-1, :-2] &
                    padded[1:-1, 2:]
                )
                edge_mask = circle_pixels_mask & ~eroded
                y_indices, x_indices = np.where(edge_mask)
                if len(x_indices) < 50:
                    continue

                if has_large_washer_evidence(edge_mask):
                    prefer_large_without_profile = True

                x_idx_full = x_indices + x_start
                y_idx_full = y_indices + y_start

                edge_components = extract_connected_components(edge_mask, min_pixels=15)
                dists_to_cross = np.sqrt((x_idx_full - xc_cross) ** 2 + (y_idx_full - yc_cross) ** 2)
                small_mask = dists_to_cross < 155
                large_mask = dists_to_cross < 175

                if template_kind != "large":
                    small_candidate = evaluate_component_candidates(
                        edge_components,
                        small_search_ranges,
                        small_nominal,
                        "small",
                        155,
                    )
                    best_small_candidate = choose_candidate(best_small_candidate, small_candidate)

                    x_small = x_idx_full[small_mask]
                    y_small = y_idx_full[small_mask]
                    if len(x_small) >= 15 and not bright_edge_mode:
                        small_circle = fit_circle_for_mode(
                            x_small,
                            y_small,
                            small_search_ranges,
                            small_nominal,
                            "small",
                        )
                        best_small_candidate = choose_candidate(
                            best_small_candidate,
                            build_circle_candidate(small_circle, x_small, y_small, small_nominal, "small"),
                        )

                if template_kind == "large" or prefer_large_without_profile:
                    large_candidate = evaluate_component_candidates(
                        edge_components,
                        large_search_ranges,
                        large_nominal,
                        "large",
                        175,
                    )
                    best_large_candidate = choose_candidate(best_large_candidate, large_candidate)

                    x_large = x_idx_full[large_mask]
                    y_large = y_idx_full[large_mask]
                    if len(x_large) >= 15:
                        large_circle = fit_circle_for_mode(
                            x_large,
                            y_large,
                            large_search_ranges,
                            large_nominal,
                            "large",
                        )
                        best_large_candidate = choose_candidate(
                            best_large_candidate,
                            build_circle_candidate(large_circle, x_large, y_large, large_nominal, "large"),
                        )

                    bright_mask = masked_mean > 200
                    dark_mask = masked_mean < (thresh + 30)
                    padded_dark = np.pad(dark_mask, 2, mode='constant', constant_values=False)
                    dark_neighbor = (
                        padded_dark[2:-2, 2:-2] |
                        padded_dark[:-4, 2:-2] | padded_dark[4:, 2:-2] |
                        padded_dark[2:-2, :-4] | padded_dark[2:-2, 4:] |
                        padded_dark[:-4, :-4] | padded_dark[:-4, 4:] |
                        padded_dark[4:, :-4] | padded_dark[4:, 4:]
                    )
                    arc_mask = bright_mask & dark_neighbor & is_gray_local
                    y_arc, x_arc = np.where(arc_mask)
                    if len(x_arc) >= 50:
                        x_arc_full = x_arc + x_start
                        y_arc_full = y_arc + y_start
                        arc_dists = np.sqrt((x_arc_full - xc_cross) ** 2 + (y_arc_full - yc_cross) ** 2)
                        arc_valid = arc_dists < 175
                        x_arc_fit = x_arc_full[arc_valid]
                        y_arc_fit = y_arc_full[arc_valid]
                        if len(x_arc_fit) >= 50:
                            arc_circle = fit_circle_for_mode(
                                x_arc_fit,
                                y_arc_fit,
                                large_search_ranges,
                                large_nominal,
                                "large",
                            )
                            best_large_candidate = choose_candidate(
                                best_large_candidate,
                                build_circle_candidate(arc_circle, x_arc_fit, y_arc_fit, large_nominal, "large"),
                            )

            if template_kind == "large" or (
                prefer_large_without_profile and
                best_profiled_small_raw_hough_candidate is None
            ):
                best_candidate = best_large_candidate if best_large_candidate is not None else best_small_candidate
                if best_candidate is None and best_large_raw_hough_candidate is not None:
                    best_candidate = best_large_raw_hough_candidate
            else:
                if template_kind == "medium":
                    best_candidate = choose_candidate(best_small_candidate, best_opencv_small_candidate)
                    if best_candidate is None:
                        best_candidate = best_large_candidate
                    else:
                        refined_medium_candidate = refine_medium_ring_candidate(best_candidate)
                        if refined_medium_candidate is not None:
                            best_candidate = refined_medium_candidate
                        refined_medium_candidate = refine_medium_cardinal_geometry(best_candidate)
                        if refined_medium_candidate is not None:
                            best_candidate = refined_medium_candidate
                        refined_medium_candidate = refine_medium_axis_geometry(best_candidate)
                        if refined_medium_candidate is not None:
                            best_candidate = refined_medium_candidate
                        best_candidate = choose_medium_circle_candidate(
                            best_candidate,
                            best_medium_hough_candidate,
                        )
                elif best_opencv_small_candidate is not None:
                    best_candidate = choose_small_circle_candidate(
                        best_small_candidate,
                        best_opencv_small_candidate,
                    )
                else:
                    best_candidate = best_small_candidate if best_small_candidate is not None else best_large_candidate
                if should_prefer_unprofiled_medium_candidate(best_candidate, best_medium_hough_candidate):
                    best_candidate = best_medium_hough_candidate
                if best_profile_hough_candidate is not None:
                    if best_candidate is None:
                        best_candidate = best_profile_hough_candidate
                    elif template_kind == "medium":
                        best_candidate = choose_medium_circle_candidate(best_candidate, best_profile_hough_candidate)
                    elif template_kind == "small":
                        current_distance = float(np.hypot(
                            best_candidate["circle"][0] - xc_cross,
                            best_candidate["circle"][1] - yc_cross,
                        ))
                        hough_distance = float(np.hypot(
                            best_profile_hough_candidate["circle"][0] - xc_cross,
                            best_profile_hough_candidate["circle"][1] - yc_cross,
                        ))
                        if hough_distance + 18.0 < current_distance:
                            best_candidate = best_profile_hough_candidate
                if best_crosshair_small_candidate is not None:
                    current_profile_name = None if best_candidate is None else infer_circle_session_suffix(
                        template_match,
                        active_parts_name,
                        best_candidate["circle"][2],
                    )
                    if (
                        best_candidate is None or
                        (
                            current_profile_name == "t0.3*2.0" and
                            count_visible_circle_points(best_crosshair_small_candidate) >= 3
                        )
                    ):
                        best_candidate = best_crosshair_small_candidate
                if best_profiled_small_raw_hough_candidate is not None:
                    best_candidate = best_profiled_small_raw_hough_candidate

            if best_candidate is None:
                return None

            xc, yc, R = best_candidate["circle"]

            # Prefer the suffix that is consistent with the fitted circle
            # radius. Weak row-name OCR or template matches can guess the wrong
            # family, but the circle fit is already available here.
            inferred_suffix = infer_circle_session_suffix(template_match, active_parts_name, R)
            session_id = build_session_id(inferred_suffix)
            if session_id is None:
                session_id = build_session_id()

            if R > 120:
                R += 5.5

            visible_pixels = resolve_circle_pixel_visibility(
                xc,
                yc,
                R,
                best_candidate["support_x"],
                best_candidate["support_y"],
            )
            raw_pixels = {
                "top": [int(round(xc)), int(round(yc - R))],
                "bottom": [int(round(xc)), int(round(yc + R))],
                "left": [int(round(xc - R)), int(round(yc))],
                "right": [int(round(xc + R)), int(round(yc))],
            }
            if R > 120 or template_kind == "large":
                visible_candidates = {
                    name: pixel
                    for name, pixel in raw_pixels.items()
                    if 0 <= pixel[0] < arr.shape[1] and 0 <= pixel[1] < arr.shape[0]
                }
                if visible_candidates:
                    best_name = min(
                        visible_candidates,
                        key=lambda name: (
                            (visible_candidates[name][0] - xc_cross) ** 2 +
                            (visible_candidates[name][1] - yc_cross) ** 2
                        ),
                    )
                    visible_pixels = {name: None for name in ("top", "bottom", "left", "right")}
                    visible_pixels[best_name] = visible_candidates[best_name]
                else:
                    visible_pixels = {name: None for name in ("top", "bottom", "left", "right")}
            else:
                visible_count = sum(
                    1
                    for name in ("top", "bottom", "left", "right")
                    if visible_pixels[name] is not None
                )
                support_metrics = best_candidate.get("metrics", {})
                support_is_adequate = (
                    support_metrics.get("support_count", 0) >= 20 and
                    support_metrics.get("coverage_bins", 0) >= 6
                )
                if (
                    template_kind == "small" and
                    support_is_adequate and
                    (
                        visible_count >= 2 or
                        best_candidate.get("source") == "profiled_small_raw_hough"
                    )
                ):
                    for name, pixel in raw_pixels.items():
                        if visible_pixels[name] is not None:
                            continue
                        if 0 <= pixel[0] < arr.shape[1] and 0 <= pixel[1] < arr.shape[0]:
                            visible_pixels[name] = pixel

            visible_names = tuple(
                name for name in ("top", "bottom", "left", "right")
                if visible_pixels[name] is not None
            )
            if (
                best_candidate.get("source") != "profiled_small_raw_hough" and
                should_reject_partial_small_circle(shape_type, template_kind, visible_names)
            ):
                return None

            res = {
                "top": visible_pixels["top"],
                "bottom": visible_pixels["bottom"],
                "left": visible_pixels["left"],
                "right": visible_pixels["right"],
                "session_id": session_id,
            }
            return finalize_result(res)

        # 3. Mode Routing Control Flow
        if shape_type == "cylinder":
            cyl_res = detect_cylinder()
            if cyl_res is not None:
                return finalize_result(cyl_res, is_cylinder=True)
        elif shape_type == "circle":
            circle_res = detect_circle()
            if circle_res is not None:
                return circle_res
        else:
            auto_shape_type = resolve_auto_shape_type(
                active_row,
                active_feeder_auto_tc,
                template_match,
                active_parts_name=active_parts_name,
            )
            if auto_shape_type is None:
                if active_feeder_auto_tc is False:
                    cyl_res = detect_cylinder()
                    if (
                        isinstance(cyl_res, dict) and
                        cyl_res.get("left") is not None and
                        cyl_res.get("right") is not None
                    ):
                        return finalize_result(cyl_res, is_cylinder=True)
                    if cyl_res is None and template_kind == "large":
                        circle_res = detect_circle()
                        if circle_res is not None:
                            return circle_res
                if should_try_uncertain_circle_fallback(active_row, active_feeder_auto_tc, auto_shape_type, active_parts_name=active_parts_name):
                    shape_type = "circle"
                    circle_res = detect_circle()
                    if circle_res is not None:
                        return circle_res
                # No active row in the parts table — we have no template context,
                # so we cannot reliably distinguish real parts from background.
                # Return no-detection to prevent false positives.
                return finalize_result(make_no_detect_result(), is_cylinder=True)
            if auto_shape_type == "circle":
                shape_type = "circle"
                circle_res = detect_circle()
                if circle_res is not None:
                    return circle_res
                return finalize_result(make_no_detect_result(), is_cylinder=True)
            if auto_shape_type == "cylinder":
                if template_kind == "small" and template_match is not None:
                    circle_res = detect_circle()
                    if circle_res is not None:
                        return circle_res
                cyl_res = detect_cylinder()
                if has_visible_cylinder_points(cyl_res):
                    return finalize_result(cyl_res, is_cylinder=True)
                if template_kind == "small":
                    circle_res = detect_circle()
                    if circle_res is not None:
                        return circle_res
                return finalize_result(make_no_detect_result(), is_cylinder=True)

        # Fallback return when nothing succeeded
        return finalize_result(make_no_detect_result())

    except Exception as e:
        # If any preprocessing or decoding exception occurs, return it in the error response format
        return {"error": f"Failed to process image: {str(e)}"}
