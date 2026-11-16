"""Process-local compatibility for the existing Python 3.11 / NumPy 1.26 env.

No upstream files, packages, policy weights or simulation physics are changed.
"""

import numpy as np
import onnxruntime as ort

_CHANGES = None


def setup():
    global _CHANGES
    if _CHANGES is not None:
        return list(_CHANGES)
    changes = []
    if not hasattr(np.linalg, "vector_norm"):
        # ATHLETE calls this with a single vector axis, never a matrix norm.
        def vector_norm(x, axis=None, keepdims=False, ord=2):
            if axis is None:
                x = np.asarray(x).reshape(-1)
            return np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

        np.linalg.vector_norm = vector_norm
        changes.append("NumPy vector_norm equivalent for vector-axis calls")
    original = ort.InferenceSession

    def cpu_session(path, sess_options=None, **kwargs):
        if sess_options is None:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2
            sess_options.inter_op_num_threads = 1
        kwargs["providers"] = ["CPUExecutionProvider"]
        return original(path, sess_options=sess_options, **kwargs)

    ort.InferenceSession = cpu_session
    changes.append("ONNX CPUExecutionProvider; intra/inter threads 2/1")
    _CHANGES = changes
    return list(changes)


if __name__ == "__main__":
    import runpy
    import sys

    print("[LOCAL COMPAT]", setup(), flush=True)
    entrypoint = sys.argv.pop(1)
    sys.argv[0] = entrypoint
    runpy.run_path(entrypoint, run_name="__main__")
