import sys
# Set integer string conversion limits to prevent issues with large data handling
if not hasattr(sys, 'get_int_max_str_digits'):
    sys.get_int_max_str_digits = lambda: 4300
if not hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits = lambda x: None

from flask import Flask, request, jsonify
from flask_cors import CORS
import pipeline

app = Flask(__name__)
if hasattr(app, 'json'):
    app.json.sort_keys = False
else:
    app.config['JSON_SORT_KEYS'] = False
CORS(app)

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

@app.route("/api/v1/process_image", methods=["POST"])
def process_image():
    """
    Main inference endpoint. Matches training_type (e.g. FUJI_DETECTOR)
    and routes base64 encoded image to the correct handler.
    """
    data = request.get_json(silent=True)
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

    # Optional shape hint: 'cylinder' forces cylinder detection mode;
    # None / 'circle' uses auto-detect (default).
    shape_type = data.get("shape_type", None)

    try:
        result = handler(image_b64, shape_type=shape_type)
    except TypeError:
        # Handler doesn't accept shape_type – call without it
        result = handler(image_b64)
    except Exception as e:
        return _error(f"Inference failed: {e}", status=500)

    if isinstance(result, dict) and "error" in result:
        return _error(result["error"])

    return _success(result)

@app.route("/api/v1/clear_sessions", methods=["POST"])
def clear_sessions():
    """Clear all accumulated session data."""
    return jsonify({"success": True})

if __name__ == "__main__":
    from waitress import serve
    print("Starting waitress server on port 5000...")
    serve(app, host="0.0.0.0", port=5000)
