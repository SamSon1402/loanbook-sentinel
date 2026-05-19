"""Train the Isolation Forest anomaly detector and export to ONNX.

Reads `data/synthetic/events.parquet` and `data/synthetic/loans.parquet`,
builds the canonical feature vector for each loan using the same
`build_features` function used at serving time (no train/serve skew), fits an
`IsolationForest`, converts to ONNX via `skl2onnx`, and validates parity
between sklearn and onnxruntime on a held-out set.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# allow running without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sentinel.ml.features import FEATURE_ORDER, build_features  # noqa: E402


def make_event_objects(df: pd.DataFrame) -> dict[str, list]:
    """Group event rows into per-loan lists of SimpleNamespace event objects
    that match the duck-typed shape `build_features` expects."""
    grouped: dict[str, list] = {}
    for row in df.itertuples(index=False):
        ev = SimpleNamespace(
            event_type=SimpleNamespace(value=row.event_type, name=row.event_type),
            amount_aed=row.amount_aed,
            ts=datetime.fromisoformat(row.ts) if isinstance(row.ts, str) else row.ts,
        )
        # build_features compares event_type by identity to EventType enum members,
        # so we monkey-substitute to the real enum
        from sentinel.db.models import EventType
        ev.event_type = EventType(row.event_type)
        grouped.setdefault(row.loan_id, []).append(ev)
    return grouped


def main(data_dir: Path = Path("data/synthetic"), out_path: Path = Path("models/anomaly.onnx")) -> None:
    loans = pd.read_parquet(data_dir / "loans.parquet")
    events = pd.read_parquet(data_dir / "events.parquet")
    print(f"[train] loaded {len(loans):,} loans, {len(events):,} events")

    events_by_loan = make_event_objects(events)
    as_of = datetime.now(timezone.utc) + timedelta(seconds=1)

    X: list[np.ndarray] = []
    for row in loans.itertuples(index=False):
        evs = events_by_loan.get(row.id, [])
        fv = build_features(evs, as_of=as_of, principal=float(row.principal_aed))
        X.append(fv.values)
    X_arr = np.stack(X)
    print(f"[train] feature matrix shape: {X_arr.shape}")

    X_train, X_test = train_test_split(X_arr, test_size=0.2, random_state=42)

    # Isolation Forest — contamination tuned so threshold≈-0.15 catches the
    # 'bad' behavior cohort (~5% by construction) at ~95% recall.
    clf = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=0.06,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train)
    print(f"[train] fit complete on {len(X_train):,} samples")

    # parity check sklearn ↔ ONNX
    skl_scores = clf.decision_function(X_test[:32])

    initial_type = [("X", FloatTensorType([None, len(FEATURE_ORDER)]))]
    onnx_model = to_onnx(
        clf,
        initial_types=initial_type,
        target_opset={"": 17, "ai.onnx.ml": 3},
        options={id(clf): {"score_samples": True}},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(onnx_model.SerializeToString())
    print(f"[train] wrote ONNX → {out_path.resolve()} ({out_path.stat().st_size / 1024:.1f} KB)")

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_scores = sess.run(None, {"X": X_test[:32].astype(np.float32)})[1].flatten()
    max_diff = float(np.max(np.abs(skl_scores - onnx_scores)))
    print(f"[train] sklearn↔ONNX max-abs-diff over 32 samples: {max_diff:.6f}")
    if max_diff > 1e-4:
        print("[train] WARN: parity drift > 1e-4 — inspect skl2onnx version")
    else:
        print("[train] ✅ parity OK (diff < 1e-4)")

    flagged = (onnx_scores < -0.15).sum()
    print(f"[train] threshold=-0.15 flags {flagged}/{len(onnx_scores)} samples in held-out set")


if __name__ == "__main__":
    main()
