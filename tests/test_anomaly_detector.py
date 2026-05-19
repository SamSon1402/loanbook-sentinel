"""Unit tests for AnomalyDetector + features."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sentinel.db.models import EventType
from sentinel.ml.anomaly_detector import AnomalyDetector
from sentinel.ml.features import FEATURE_ORDER, FeatureVector, build_features


def _ev(type_: EventType, amount: float, ts: datetime) -> SimpleNamespace:
    return SimpleNamespace(event_type=type_, amount_aed=Decimal(str(amount)), ts=ts)


def test_feature_vector_shape() -> None:
    vec = np.zeros(len(FEATURE_ORDER), dtype=np.float32)
    fv = FeatureVector(values=vec)
    assert fv.values.shape == (len(FEATURE_ORDER),)
    assert set(fv.as_dict()) == set(FEATURE_ORDER)


def test_feature_vector_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        FeatureVector(values=np.zeros(5, dtype=np.float32))


def test_build_features_empty_events_safe() -> None:
    fv = build_features([], as_of=datetime.now(UTC), principal=10_000)
    assert fv.values.shape == (len(FEATURE_ORDER),)
    # No events → days_since_last_inflow defaults to 999
    assert fv.as_dict()["days_since_last_inflow"] == 999


def test_build_features_counts_nsf_window() -> None:
    now = datetime.now(UTC)
    events = [
        _ev(EventType.NSF, -100, now - timedelta(days=10)),
        _ev(EventType.NSF, -100, now - timedelta(days=30)),
        _ev(EventType.NSF, -100, now - timedelta(days=120)),  # outside 90d window
    ]
    fv = build_features(events, as_of=now, principal=10_000)
    assert fv.as_dict()["nsf_event_count_90d"] == 2


def test_build_features_principal_must_be_positive() -> None:
    with pytest.raises(ValueError, match="principal"):
        build_features([], as_of=datetime.now(UTC), principal=0)


def test_anomaly_detector_loads_and_scores(tiny_onnx_model: Path) -> None:
    det = AnomalyDetector(tiny_onnx_model, threshold=-0.05)
    fv = FeatureVector(values=np.zeros(len(FEATURE_ORDER), dtype=np.float32))
    score, is_anom = det.score(fv)
    assert isinstance(score, float)
    assert isinstance(is_anom, bool)


def test_anomaly_detector_raises_on_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        AnomalyDetector(tmp_path / "nope.onnx")


def test_anomaly_detector_batch_shape_validation(tiny_onnx_model: Path) -> None:
    det = AnomalyDetector(tiny_onnx_model)
    with pytest.raises(ValueError, match="expected shape"):
        det.score_batch(np.zeros((3, 5), dtype=np.float32))
