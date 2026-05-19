"""ONNX-served Isolation Forest anomaly detector.

The model is trained offline by `scripts/train_anomaly_model.py` and exported
to ONNX via `skl2onnx`. At serving time, this class wraps `onnxruntime` and
exposes a single `score()` method. Same artifact can be dropped into Seldon
Core or TorchServe.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort

from sentinel.core.logging import get_logger
from sentinel.core.metrics import inference_latency_seconds, inference_total
from sentinel.ml.features import FEATURE_ORDER, FeatureVector

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


class AnomalyDetector:
    """Thread-safe ONNX-runtime wrapper around a trained Isolation Forest.

    Args:
        model_path: Path to the .onnx artifact.
        threshold: decision_function score below which a sample is flagged.
                   More negative = more anomalous. Tuned in training.
        providers: ORT execution providers. Default CPU; override to e.g.
                   ['CUDAExecutionProvider', 'CPUExecutionProvider'] for GPU.
    """

    def __init__(
        self,
        model_path: Path | str,
        threshold: float = -0.15,
        providers: list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {self.model_path}. "
                "Run `python scripts/train_anomaly_model.py` first."
            )

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2

        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers or ["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        log.info(
            "anomaly_detector.loaded",
            path=str(self.model_path),
            threshold=threshold,
            input_name=self._input_name,
            providers=self._session.get_providers(),
        )

    def score(self, features: FeatureVector) -> tuple[float, bool]:
        """Score a single feature vector.

        Returns:
            (score, is_anomaly). Score < threshold ⇒ flagged.
        """
        return self.score_batch(features.values.reshape(1, -1))[0]

    def score_batch(self, batch: np.ndarray) -> list[tuple[float, bool]]:
        """Score a batch of N samples (shape (N, F))."""
        if batch.ndim != 2 or batch.shape[1] != len(FEATURE_ORDER):
            raise ValueError(
                f"expected shape (N, {len(FEATURE_ORDER)}), got {batch.shape}"
            )
        start = time.perf_counter()
        try:
            # IsolationForest in skl2onnx outputs both labels and scores. We
            # take the decision-function score (more negative = more anomalous).
            outputs = self._session.run(None, {self._input_name: batch.astype(np.float32)})
            scores = outputs[1].flatten()  # shape (N,)
        except Exception:
            inference_total.labels(outcome="error").inc()
            log.exception("anomaly_detector.inference_failed")
            raise
        finally:
            inference_latency_seconds.observe(time.perf_counter() - start)

        results: list[tuple[float, bool]] = []
        for s in scores:
            is_anom = bool(s < self.threshold)
            inference_total.labels(outcome="anomaly" if is_anom else "normal").inc()
            results.append((float(s), is_anom))
        return results
