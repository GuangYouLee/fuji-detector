"""
pipeline.py — Training type routing
Registers handler functions per training_type for each pipeline stage.
"""

from FUJI_DETECTOR import inspect as _fuji_inspect

# Inference handlers — used by process_image endpoint.
INFER_HANDLERS = {
    "FUJI_DETECTOR": _fuji_inspect.run_inference,
}

def get_handler(registry, training_type):
    """
    Resolve a handler function from a registry by training_type.
    Raises ValueError if the training_type is not registered.
    """
    handler = registry.get(training_type)
    if handler is None:
        raise ValueError(
            f"Unsupported training_type: '{training_type}'. "
            f"Available: {list(registry.keys())}"
        )
    return handler
