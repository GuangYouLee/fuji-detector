import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from FUJI_DETECTOR.inspect import run_inference as _run_fuji_inference

app = Flask(__name__)
app.json.sort_keys = False
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


def _request_data(default_training_type=None):
    data = dict(request.get_json(silent=True) or {})
    training_type = (
        request.args.get("TRAINING_TYPE")
        or request.args.get("training_type")
        or data.get("training_type")
        or default_training_type
    )
    if training_type is not None:
        data["training_type"] = training_type
    return data


def _run_detection(data):
    try:
        if not data:
            raise ValueError("Request body must be JSON.")

        training_type = data.get("training_type")
        if not training_type:
            raise ValueError("Missing required field: 'training_type'")
        if training_type != "FUJI_DETECTOR":
            raise ValueError(f"Unsupported training_type: '{training_type}'. Available: ['FUJI_DETECTOR']")

        image_b64 = data.get("img_base64")
        if not image_b64:
            raise ValueError("Missing required field: img_base64.")

        result = _run_fuji_inference(image_b64, shape_type=data.get("shape_type"))
    except ValueError as e:
        return None, _error(str(e))
    except Exception as e:
        return None, _error(f"Inference failed: {e}", status=500)

    if isinstance(result, dict) and "error" in result:
        return None, _error(result["error"])

    return result, None


def _format_detector_response(result, requested_edge=None):
    if not isinstance(result, dict):
        return result
    if requested_edge is None:
        return result

    point = result.get(requested_edge)

    return {
        "execute_label": requested_edge.capitalize() if point is not None else None,
        "session_id": result.get("session_id"),
        "message": f"x={point[0]},y={point[1]}" if point is not None else None,
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
    result, error = _run_detection(data)
    if error:
        return error

    if requested_edge is not None and isinstance(result, dict) and result.get(requested_edge) is None:
        return _error(f"Edge '{requested_edge}' was not detected in this image.")

    return _success(_format_detector_response(result, requested_edge=requested_edge))


@app.route("/api/v1/process_image", methods=["POST"])
def process_image():
    """
    Main inference endpoint. Matches training_type (e.g. FUJI_DETECTOR)
    and routes base64 encoded image to the correct handler.
    """
    return _run_image_detection(_request_data())


@app.route("/api/v1/top", methods=["POST"])
def process_image_top():
    return _run_image_detection(_request_data(), requested_edge="top")


@app.route("/api/v1/bottom", methods=["POST"])
def process_image_bottom():
    return _run_image_detection(_request_data(), requested_edge="bottom")


@app.route("/api/v1/left", methods=["POST"])
def process_image_left():
    return _run_image_detection(_request_data(), requested_edge="left")


@app.route("/api/v1/right", methods=["POST"])
def process_image_right():
    return _run_image_detection(_request_data(), requested_edge="right")


@app.route("/api/v1/detect_screen", methods=["POST"])
def detect_screen():
    """
    Runs the shared image inference path using the posted base64 image.
    """
    return _run_image_detection(_request_data())


@app.route("/api/v1/detect_screen/top", methods=["POST"])
def detect_screen_top():
    return _run_image_detection(_request_data(), requested_edge="top")


@app.route("/api/v1/detect_screen/bottom", methods=["POST"])
def detect_screen_bottom():
    return _run_image_detection(_request_data(), requested_edge="bottom")


@app.route("/api/v1/detect_screen/left", methods=["POST"])
def detect_screen_left():
    return _run_image_detection(_request_data(), requested_edge="left")


@app.route("/api/v1/detect_screen/right", methods=["POST"])
def detect_screen_right():
    return _run_image_detection(_request_data(), requested_edge="right")


@app.route("/api/v1/detect_screen/all", methods=["POST"])
def detect_screen_all():
    """Run inference once and return all detected edges."""
    result, error = _run_detection(_request_data(default_training_type="FUJI_DETECTOR"))
    if error:
        return error

    return _success(_format_all_edges(result))


@app.route("/api/v1/detect_screen/next", methods=["POST"])
def detect_screen_next():
    """Return one edge per call, keyed by the transient sequence session_id."""
    data = _request_data(default_training_type="FUJI_DETECTOR")

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

    result, error = _run_detection(data)
    if error:
        return error

    # Build the ordered queue of edges that actually have a detected point.
    detected = [lbl for lbl in POINT_FIELDS if result.get(lbl) is not None]
    if not detected:
        return _error("No boundary points detected in this image.")

    # Pop the first edge immediately to return on this call.
    first_label = detected.pop(0)
    first_point = result[first_label]

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
