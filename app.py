import sys
# Set integer string conversion limits to prevent issues with large data handling
if not hasattr(sys, 'get_int_max_str_digits'):
    sys.get_int_max_str_digits = lambda: 4300
if not hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits = lambda x: None

import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
import pipeline

app = Flask(__name__)
if hasattr(app, 'json'):
    app.json.sort_keys = False
else:
    app.config['JSON_SORT_KEYS'] = False
CORS(app)

POINT_FIELDS = ("top", "bottom", "left", "right")

# ---------------------------------------------------------------------------
# Sequential session cache — used by /detect_screen/next
# Key: session_id (str)  Value: {"edges": [...remaining labels...], "result": dict}
# ---------------------------------------------------------------------------
_seq_cache: dict = {}


def _success(data, status=200):
    """Return JSON response for success cases."""
    return jsonify({"data": data, "success": True}), status


def _error(message, status=400):
    """Return JSON response for error cases."""
    return jsonify({"data": {"message": message}, "success": False}), status


def _get_required(data, *keys):
    """Utility to retrieve required keys from the JSON request body."""
    values = []
    for key in keys:
        if key not in data:
            raise ValueError(f"Missing required field: '{key}'")
        values.append(data[key])
    return values if len(values) > 1 else values[0]


def _format_detector_response(result, requested_edge=None):
    if not isinstance(result, dict):
        return result
    if requested_edge is None:
        return result

    selected_label = None
    selected_message = None
    for label in ((requested_edge,) if requested_edge else POINT_FIELDS):
        point = result.get(label)
        if point is None:
            continue
        selected_label = label.capitalize()
        selected_message = f"x={point[0]},y={point[1]}"
        break

    return {
        "execute_label": selected_label,
        "session_id": result.get("session_id"),
        "message": selected_message,
    }


def _format_all_edges(result):
    """Return all four boundary pixels from a single inference result.

    Each present edge is returned in the same ``execute_label / message``
    shape that the individual edge endpoints use, so the calling script can
    reuse the same parsing logic for every entry.
    """
    if not isinstance(result, dict):
        return result

    edges = {}
    for label in POINT_FIELDS:
        point = result.get(label)
        edges[label] = (
            {"execute_label": label.capitalize(), "message": f"x={point[0]},y={point[1]}"}
            if point is not None
            else None
        )

    return {
        "session_id": result.get("session_id"),
        "edges": edges,
    }


def _run_image_detection(data, requested_edge=None):
    if not data:
        return _error("Request body must be JSON.")

    try:
        training_type = _get_required(data, "training_type")
    except ValueError as e:
        return _error(str(e))

    try:
        handler = pipeline.get_handler(pipeline.INFER_HANDLERS, training_type)
    except ValueError as e:
        return _error(str(e))

    image_b64 = data.get("img_base64")
    if not image_b64:
        return _error("Missing required field: img_base64.")

    try:
        if training_type == "FUJI_DETECTOR":
            result = handler(image_b64, shape_type=data.get("shape_type"))
        else:
            result = handler(image_b64)
    except TypeError:
        result = handler(image_b64)
    except Exception as e:
        return _error(f"Inference failed: {e}", status=500)

    if isinstance(result, dict) and "error" in result:
        return _error(result["error"])

    return _success(_format_detector_response(result, requested_edge=requested_edge))


def _run_screen_detection(data, requested_edge=None):
    data = dict(data or {})
    # Support callers that supply the detector in the query string.
    tt = (
        request.args.get("TRAINING_TYPE")
        or request.args.get("training_type")
        or data.get("training_type")
    )
    if tt:
        data["training_type"] = tt
    return _run_image_detection(data, requested_edge=requested_edge)


@app.route("/api/v1/process_image", methods=["POST"])
def process_image():
    """
    Main inference endpoint. Matches training_type (e.g. FUJI_DETECTOR)
    and routes base64 encoded image to the correct handler.
    """
    return _run_image_detection(request.get_json(silent=True))


@app.route("/api/v1/top", methods=["POST"])
def process_image_top():
    return _run_image_detection(request.get_json(silent=True), requested_edge="top")


@app.route("/api/v1/bottom", methods=["POST"])
def process_image_bottom():
    return _run_image_detection(request.get_json(silent=True), requested_edge="bottom")


@app.route("/api/v1/left", methods=["POST"])
def process_image_left():
    return _run_image_detection(request.get_json(silent=True), requested_edge="left")


@app.route("/api/v1/right", methods=["POST"])
def process_image_right():
    return _run_image_detection(request.get_json(silent=True), requested_edge="right")


@app.route("/api/v1/detect_screen", methods=["POST"])
def detect_screen():
    """
    Runs the shared image inference path using the posted base64 image.
    """
    return _run_screen_detection(request.get_json(silent=True))


@app.route("/api/v1/detect_screen/top", methods=["POST"])
def detect_screen_top():
    return _run_screen_detection(request.get_json(silent=True), requested_edge="top")


@app.route("/api/v1/detect_screen/bottom", methods=["POST"])
def detect_screen_bottom():
    return _run_screen_detection(request.get_json(silent=True), requested_edge="bottom")


@app.route("/api/v1/detect_screen/left", methods=["POST"])
def detect_screen_left():
    return _run_screen_detection(request.get_json(silent=True), requested_edge="left")


@app.route("/api/v1/detect_screen/right", methods=["POST"])
def detect_screen_right():
    return _run_screen_detection(request.get_json(silent=True), requested_edge="right")


@app.route("/api/v1/detect_screen/all", methods=["POST"])
def detect_screen_all():
    """Run inference once and return all four boundary pixels in a single response.

    Use this instead of calling /top, /bottom, /left, /right separately when
    a full circle is expected — it avoids 4 redundant inference runs.

    Response shape::

        {
          "success": true,
          "data": {
            "session_id": "...",
            "edges": {
              "top":    {"execute_label": "Top",    "message": "x=...,y=..."},
              "bottom": {"execute_label": "Bottom", "message": "x=...,y=..."},
              "left":   {"execute_label": "Left",   "message": "x=...,y=..."},
              "right":  {"execute_label": "Right",  "message": "x=...,y=..."}
            }
          }
        }

    Any edge that was not detected will be ``null`` in the ``edges`` dict.
    """
    data = dict(request.get_json(silent=True) or {})
    data["training_type"] = (
        request.args.get("TRAINING_TYPE")
        or request.args.get("training_type")
        or data.get("training_type")
        or "FUJI_DETECTOR"
    )
    if not data:
        return _error("Request body must be JSON.")

    try:
        training_type = _get_required(data, "training_type")
    except ValueError as e:
        return _error(str(e))

    try:
        handler = pipeline.get_handler(pipeline.INFER_HANDLERS, training_type)
    except ValueError as e:
        return _error(str(e))

    image_b64 = data.get("img_base64")
    if not image_b64:
        return _error("Missing required field: img_base64.")

    try:
        if training_type == "FUJI_DETECTOR":
            result = handler(image_b64, shape_type=data.get("shape_type"))
        else:
            result = handler(image_b64)
    except TypeError:
        result = handler(image_b64)
    except Exception as e:
        return _error(f"Inference failed: {e}", status=500)

    if isinstance(result, dict) and "error" in result:
        return _error(result["error"])

    return _success(_format_all_edges(result))


@app.route("/api/v1/detect_screen/next", methods=["POST"])
def detect_screen_next():
    """Round-robin endpoint: same URL, called 4 times, returns one edge per call.

    **Call 1** — omit ``session_id`` (or send an unknown one):
      Runs inference once, caches all four edges, returns ``Top``.
      The response includes a ``session_id`` — pass it back on every
      subsequent call so the server knows which cache slot to advance.

    **Calls 2-4** — include the ``session_id`` from call 1:
      No inference is run. Returns ``Bottom``, ``Left``, ``Right`` in order.
      After the 4th call the cache slot is automatically cleared.

    Request body::

        {
          "img_base64": "<base64>",     # required on call 1; ignored on 2-4
          "session_id":  "<session id>", # omit on call 1, required on calls 2-4
          "training_type": "FUJI_DETECTOR"  # optional, defaults to FUJI_DETECTOR
        }

    Response (same shape as the single-edge endpoints)::

        {
          "success": true,
          "data": {
            "execute_label": "Top",           # Top / Bottom / Left / Right
            "message":       "x=123,y=45",
            "session_id":    "session_2_t0.2*11.0",  # detector id, UUID fallback
            "edges_remaining": 3              # how many edges are still queued
          }
        }
    """
    data = dict(request.get_json(silent=True) or {})
    training_type = (
        request.args.get("TRAINING_TYPE")
        or request.args.get("training_type")
        or data.get("training_type")
        or "FUJI_DETECTOR"
    )

    session_id = data.get("session_id", "").strip()

    # ------------------------------------------------------------------
    # If we already have a live cache slot for this session, just pop the
    # next edge — no inference needed.
    # ------------------------------------------------------------------
    if session_id and session_id in _seq_cache:
        slot = _seq_cache[session_id]
        remaining_labels = slot["edges"]
        label = remaining_labels.pop(0)
        point = slot["result"].get(label)

        # Auto-clear after the last edge is consumed.
        if not remaining_labels:
            del _seq_cache[session_id]

        if point is None:
            return _error(f"Edge '{label}' was not detected in the cached result.")

        return _success({
            "execute_label": label.capitalize(),
            "message": f"x={point[0]},y={point[1]}",
            "session_id": session_id,
            "edges_remaining": len(remaining_labels),
        })

    # ------------------------------------------------------------------
    # No valid cache slot — run inference and seed a new slot.
    # ------------------------------------------------------------------
    image_b64 = data.get("img_base64")
    if not image_b64:
        return _error(
            "Missing required field: 'img_base64'. "
            "Provide it on the first call (no session_id) to start a new sequence."
        )

    try:
        handler = pipeline.get_handler(pipeline.INFER_HANDLERS, training_type)
    except ValueError as e:
        return _error(str(e))

    try:
        if training_type == "FUJI_DETECTOR":
            result = handler(image_b64, shape_type=data.get("shape_type"))
        else:
            result = handler(image_b64)
    except TypeError:
        result = handler(image_b64)
    except Exception as e:
        return _error(f"Inference failed: {e}", status=500)

    if isinstance(result, dict) and "error" in result:
        return _error(result["error"])

    # Build the ordered queue of edges that actually have a detected point.
    detected = [lbl for lbl in POINT_FIELDS if result.get(lbl) is not None]
    if not detected:
        return _error("No boundary points detected in this image.")

    # Pop the first edge immediately to return on this call.
    first_label = detected.pop(0)
    first_point = result[first_label]

    detector_session_id = result.get("session_id")
    if isinstance(detector_session_id, str):
        detector_session_id = detector_session_id.strip()
    if not detector_session_id:
        detector_session_id = str(uuid.uuid4())

    # Cache the remaining edges (may be empty if only 1 edge detected).
    if detected:
        _seq_cache[detector_session_id] = {"edges": detected, "result": result}

    return _success({
        "execute_label": first_label.capitalize(),
        "message": f"x={first_point[0]},y={first_point[1]}",
        "session_id": detector_session_id,
        "edges_remaining": len(detected),
    })


@app.route("/api/v1/clear_sessions", methods=["POST"])
def clear_sessions():
    """Clear all accumulated session data (including sequential cache slots)."""
    _seq_cache.clear()
    return jsonify({"success": True})


if __name__ == "__main__":
    from waitress import serve
    print("Starting waitress server on port 5000...")
    serve(app, host="0.0.0.0", port=5000)
