import os
import json
import uuid
import redis
from flask import Flask, request, jsonify
from flask_cors import CORS
from FUJI_DETECTOR.inspect import check_teach_button as _check_teach_button
from FUJI_DETECTOR.inspect import is_next_parts_row_empty as _is_next_parts_row_empty
from FUJI_DETECTOR.inspect import KNOWN_CYLINDER_PART_NAME_SUFFIXES
from FUJI_DETECTOR.inspect import run_inference as _run_fuji_inference


app = Flask(__name__)
app.json.sort_keys = False
CORS(app)

POINT_FIELDS = ("top", "bottom", "left", "right")
BIG_CIRCLE_DETECTIONS = 4
BIG_CIRCLE_EDGE_ORDER = ("bottom", "left", "top", "right")
NEXT_EDGE_CACHE_PREFIX = "next_edge:"
CIRCLE_CLICK_BOUNDS = (35, 290, 330, 561)

# ---------------------------------------------------------------------------
# Sequential session cache — used by /detect_screen/next
# ---------------------------------------------------------------------------
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

try:
    _redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=0,
        decode_responses=True,
        socket_connect_timeout=1.0,
        socket_timeout=1.0
    )
    _redis_client.ping()
    _use_redis = True
    print(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    _use_redis = False
    _seq_cache: dict = {}
    print(f"Redis not available ({e}). Falling back to in-memory session cache.")


def _cache_get(key):
    if _use_redis:
        val = _redis_client.get(key)
        return json.loads(val) if val else None
    return _seq_cache.get(key)


def _cache_set(key, val, ex=3600):
    if _use_redis:
        _redis_client.set(key, json.dumps(val), ex=ex)
    else:
        _seq_cache[key] = val


def _cache_delete(key):
    if _use_redis:
        _redis_client.delete(key)
    elif key in _seq_cache:
        del _seq_cache[key]


def _cache_clear():
    if _use_redis:
        _redis_client.flushdb()
    else:
        _seq_cache.clear()


def _success(data, status=200):
    """Return JSON response for success cases."""
    return jsonify({"status": "success", "data": data}), status


def _error(message, status=400):
    """Return JSON response for error cases."""
    return jsonify({
        "status": "failed",
        "data": {
            "message": message,
            "execute": "none",
            "execute_label": "none"
        }
    }), status


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

    shape_type = (
        request.args.get("SHAPE_TYPE")
        or request.args.get("shape_type")
        or data.get("shape_type")
    )
    if shape_type is not None:
        data["shape_type"] = shape_type

    session_id = (
        request.args.get("SESSION_ID")
        or request.args.get("session_id")
        or data.get("session_id")
    )
    if session_id is not None:
        data["session_id"] = session_id

    return data


def _run_detection(data, tolerate_unsupported_resolution=False):
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
        return None, (str(e), 400)
    except Exception as e:
        return None, (f"Inference failed: {e}", 500)

    if isinstance(result, dict) and "error" in result:
        if "Unsupported image resolution" in result["error"]:
            if tolerate_unsupported_resolution:
                return {}, None
            return None, (result["error"], 400)
        return None, (result["error"], 400)

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
        "message": f"x={point[0]},y={point[1]}" if point is not None else "x=null,y=null",
    }


def _resume_cached_session(session_id):
    slot = _cache_get(session_id)
    if not slot or "edges" not in slot or "result" not in slot:
        return None

    remaining_labels = slot["edges"]
    result = slot["result"]

    if not remaining_labels:
        _cache_delete(session_id)
        return result, None, [], "pass"

    popped_label = remaining_labels.pop(0)
    remaining_after_pop = list(remaining_labels)

    if remaining_labels:
        _cache_set(session_id, slot)
        remaining_status = "continue"
    else:
        _cache_delete(session_id)
        remaining_status = "pass"

    return result, popped_label, remaining_after_pop, remaining_status


def _start_cached_session(result):
    detected = [label for label in POINT_FIELDS if result.get(label) is not None]
    if not detected:
        return None

    model_session_id = result.get("session_id")
    detector_session_id = model_session_id or str(uuid.uuid4())
    popped_label = detected.pop(0)

    if detected:
        _cache_set(detector_session_id, {"edges": detected, "result": result})

    return detector_session_id, popped_label, detected


def _format_option_b_response(result, session_id, active_labels, execute_status):
    formatted = {}
    if isinstance(result, dict):
        for k, v in result.items():
            if k not in POINT_FIELDS:
                formatted[k] = v
        for lbl in POINT_FIELDS:
            formatted[lbl] = result.get(lbl) if lbl in active_labels else None
    else:
        formatted = result

    formatted["session_id"] = session_id
    formatted["execute"] = execute_status
    formatted["execute_label"] = execute_status
    return formatted


def _no_detection_status(result):
    return "idle" if isinstance(result, dict) and result.get("screen_state") == "idle" else "none"


def _pattern_label(session_id):
    if any(session_id.endswith(f"_{suffix}") for suffix in KNOWN_CYLINDER_PART_NAME_SUFFIXES):
        return "cylinder"
    return "big_circle" if session_id.endswith("_t0.2*11.0") else "normal_circle"


def _advance_big_circle_session(session_id):
    slot = _cache_get(session_id) or {}
    detection_count = slot.get("big_circle_detections", 0) + 1
    if detection_count < BIG_CIRCLE_DETECTIONS:
        _cache_set(session_id, {"big_circle_detections": detection_count})
        return "continue"
    _cache_delete(session_id)
    return "pass"


def _cache_big_circle_next_edges(session_id, result, detected_label):
    predicted_edges = result.get("predicted_edges")
    cache_key = f"{NEXT_EDGE_CACHE_PREFIX}{session_id}"
    if (
        not isinstance(predicted_edges, dict)
        or any(edge not in predicted_edges for edge in BIG_CIRCLE_EDGE_ORDER)
        or detected_label not in BIG_CIRCLE_EDGE_ORDER
    ):
        return

    current_index = BIG_CIRCLE_EDGE_ORDER.index(detected_label)
    remaining_edges = BIG_CIRCLE_EDGE_ORDER[current_index + 1:] + BIG_CIRCLE_EDGE_ORDER[:current_index]
    _cache_set(cache_key, {
        "edges": [
            {"edge": edge, "point": predicted_edges[edge]}
            for edge in remaining_edges
        ],
    })


def _clickable_circle_point(point):
    left, top, right, bottom = CIRCLE_CLICK_BOUNDS
    return [min(max(point[0], left), right), min(max(point[1], top), bottom)]


def _run_detect_screen_with_session(data):
    result, error = _run_detection(data, tolerate_unsupported_resolution=True)
    if error:
        return _error(error[0], status=error[1])

    detected_session_id = result.get("session_id")
    if detected_session_id:
        _cache_delete(detected_session_id)

    detected = [label for label in POINT_FIELDS if result.get(label) is not None]
    execute_status = _no_detection_status(result) if not detected else "pass"

    return _success(
        _format_option_b_response(
            result,
            detected_session_id,
            detected,
            execute_status,
        )
    )


def _format_detect_screen_response(session_id, popped_label, result, remaining_status):
    point = result.get(popped_label) if popped_label else None
    data = {
        "execute_label": remaining_status,
        "session_id": session_id,
        "message": f"x={point[0]},y={point[1]}" if point is not None else "x=null,y=null",
        "edge": popped_label.capitalize() if popped_label else None
    }
    return jsonify({"success": True, "data": data})


def _detect_screen_error(message, status=400):
    return jsonify({
        "success": False,
        "data": {
            "message": message,
            "execute_label": "none"
        }
    }), status


def _run_detect_screen_1by1(data):
    session_id = (data.get("session_id") or "").strip()
    if session_id and not (_pattern_label(session_id) == "big_circle" and data.get("img_base64")):
        resumed = _resume_cached_session(session_id)
        if not resumed:
            return _detect_screen_error(f"Session '{session_id}' not found or expired.")

        result, popped_label, _, remaining_status = resumed
        return _format_detect_screen_response(session_id, popped_label, result, remaining_status)

    result, error = _run_detection(data, tolerate_unsupported_resolution=True)
    if error:
        return _detect_screen_error(error[0], status=error[1])

    detected_session_id = result.get("session_id")
    if detected_session_id and _pattern_label(detected_session_id) == "big_circle":
        detected_label = next((label for label in POINT_FIELDS if result.get(label) is not None), None)
        _cache_big_circle_next_edges(detected_session_id, result, detected_label)
        return _format_detect_screen_response(
            detected_session_id,
            detected_label,
            result,
            _advance_big_circle_session(detected_session_id),
        )

    if detected_session_id:
        resumed = _resume_cached_session(detected_session_id)
        if resumed:
            result, popped_label, _, remaining_status = resumed
            return _format_detect_screen_response(detected_session_id, popped_label, result, remaining_status)

    started = _start_cached_session(result)
    if not started:
        return _format_detect_screen_response(None, None, result, _no_detection_status(result))

    detector_session_id, popped_label, remaining_labels = started
    execute_status = "continue" if remaining_labels else "pass"
    return _format_detect_screen_response(
        detector_session_id,
        popped_label,
        result,
        execute_status,
    )


def _run_image_detection(data, requested_edge=None):
    if requested_edge is None:
        return _run_detect_screen_1by1(data)

    result, error = _run_detection(data, tolerate_unsupported_resolution=True)
    if error:
        return _error(error[0], status=error[1])

    if requested_edge is not None and isinstance(result, dict) and result.get(requested_edge) is None:
        return _error(f"Edge '{requested_edge}' was not detected in this image.")

    return _success(_format_detector_response(result, requested_edge=requested_edge))

@app.route("/api/v1/process_image", methods=["POST"])
def process_image():
    """
    Main inference endpoint. Matches training_type (e.g. FUJI_DETECTOR)
    and routes base64 encoded image to the correct handler.
    """
    result, error = _run_detection(_request_data())
    if error:
        return _error(error[0], status=error[1])
    return _success(result)


@app.route("/api/v1/check_teach_button", methods=["POST"])
def check_teach_button():
    data = _request_data()
    image_b64 = data.get("img_base64")
    if not image_b64:
        return jsonify({
            "success": False,
            "data": {"message": "Missing required field: img_base64."},
        }), 400

    return jsonify({
        "success": True,
        "data": {"execute_label": "found" if _check_teach_button(image_b64) else "idle"},
    })


@app.route("/api/v1/check_next", methods=["POST"])
def check_next():
    image_b64 = _request_data().get("img_base64")
    if not image_b64:
        return jsonify({
            "success": False,
            "data": {"message": "Missing required field: img_base64."},
        }), 400

    try:
        is_empty = _is_next_parts_row_empty(image_b64)
    except ValueError as error:
        return jsonify({"success": False, "data": {"message": str(error)}}), 400

    return jsonify({"success": True, "data": {"execute_label": str(not is_empty).lower()}})


@app.route("/api/v1/detect_pattern", methods=["POST"])
def detect_pattern():
    result, error = _run_detection(_request_data(default_training_type="FUJI_DETECTOR"))
    if error:
        return jsonify({"success": False, "data": {"message": error[0]}}), error[1]

    session_id = result.get("session_id")
    if not session_id:
        return jsonify({"success": False, "data": {"message": "No detectable pattern found."}}), 422

    return jsonify({
        "success": True,
        "data": {"execute_label": _pattern_label(session_id)},
    })


@app.route("/api/v1/detect_screen", methods=["POST"])
def detect_screen():
    return _run_detect_screen_1by1(_request_data())


@app.route("/api/v1/next_edge", methods=["POST"])
def next_edge():
    session_id = (_request_data().get("session_id") or "").strip()
    if _pattern_label(session_id) != "big_circle":
        return _detect_screen_error("Missing or unsupported big_circle session_id.")

    cache_key = f"{NEXT_EDGE_CACHE_PREFIX}{session_id}"
    slot = _cache_get(cache_key)
    edges = slot.get("edges") if isinstance(slot, dict) else None
    if not isinstance(edges, list) or not edges:
        return _detect_screen_error(f"No predicted edge found for session '{session_id}'.", status=404)

    next_prediction = edges[0]
    if not isinstance(next_prediction, dict) or not isinstance(next_prediction.get("point"), (list, tuple)):
        _cache_delete(cache_key)
        return _detect_screen_error(f"No predicted edge found for session '{session_id}'.", status=404)

    try:
        x, y = _clickable_circle_point(next_prediction["point"])
    except (IndexError, TypeError, ValueError):
        _cache_delete(cache_key)
        return _detect_screen_error(f"No predicted edge found for session '{session_id}'.", status=404)

    if [x, y] == next_prediction["point"]:
        edges.pop(0)
        if edges:
            _cache_set(cache_key, slot)
        else:
            _cache_delete(cache_key)

    return jsonify({
        "success": True,
        "data": {
            "execute_label": "next_edge",
            "message": f"x={x},y={y}",
            "edge": str(next_prediction.get("edge", "")).capitalize() or None,
        },
    })


@app.route("/api/v1/<any(top, bottom, left, right):edge>", methods=["POST"])
@app.route("/api/v1/detect_screen/<any(top, bottom, left, right):edge>", methods=["POST"])
def process_image_edge(edge):
    return _run_image_detection(_request_data(), requested_edge=edge)



@app.route("/api/v1/detect_screen/all", methods=["POST"])
def detect_screen_all():
    """Run inference once and return all detected edges with session support."""
    return _run_detect_screen_with_session(_request_data(default_training_type="FUJI_DETECTOR"))


@app.route("/api/v1/detect_screen/next", methods=["POST"])
def detect_screen_next():
    """Return one edge per call, keyed by the transient sequence session_id."""
    data = _request_data(default_training_type="FUJI_DETECTOR")

    session_id = (data.get("session_id") or "").strip()

    # ------------------------------------------------------------------
    # If we already have a live cache slot for this session, just pop the
    # next edge — no inference needed.
    # ------------------------------------------------------------------
    resumed = _resume_cached_session(session_id) if session_id else None
    if resumed:
        result, label, remaining_labels, _ = resumed
        point = result.get(label) if label is not None else None
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

    result, error = _run_detection(data, tolerate_unsupported_resolution=True)
    if error:
        return _error(error[0], status=error[1])

    detected_session_id = result.get("session_id")
    if detected_session_id:
        resumed = _resume_cached_session(detected_session_id)
        if resumed:
            result, label, remaining_labels, _ = resumed
            point = result.get(label) if label is not None else None
            if point is None:
                return _error(f"Edge '{label}' was not detected in the cached result.")

            return _success({
                "execute_label": label.capitalize(),
                "message": f"x={point[0]},y={point[1]}",
                "session_id": detected_session_id,
                "edges_remaining": len(remaining_labels),
            })

    # Build the ordered queue of edges that actually have a detected point.
    started = _start_cached_session(result)
    if not started:
        return _success({
            "execute_label": _no_detection_status(result),
            "message": "x=null,y=null",
            "session_id": None,
            "edges_remaining": 0,
        })

    detector_session_id, first_label, remaining_labels = started
    first_point = result[first_label]

    return _success({
        "execute_label": first_label.capitalize(),
        "message": f"x={first_point[0]},y={first_point[1]}",
        "session_id": detector_session_id,
        "edges_remaining": len(remaining_labels),
    })


@app.route("/api/v1/clear_sessions", methods=["GET", "POST"])
def clear_sessions():
    """Clear all accumulated session data (including sequential cache slots)."""
    _cache_clear()
    return _success({"message": "All sessions cleared"})


@app.route("/api/v1/clear_session", methods=["GET", "POST"])
def clear_session():
    """Clear a specific active session by ID."""
    data = _request_data()
    session_id = (data.get("session_id") or "").strip()
    if not session_id:
        return _error("Missing required parameter: 'session_id'")

    _cache_delete(session_id)
    _cache_delete(f"{NEXT_EDGE_CACHE_PREFIX}{session_id}")
    return _success({"message": f"Session '{session_id}' cleared"})


if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting waitress server on port {port} (threads=1)...")
    serve(app, host="0.0.0.0", port=port, threads=1)
