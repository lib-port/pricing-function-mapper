"""Transactional SQLite run journal and exclusive run locking."""

from __future__ import annotations

import fcntl
import importlib.metadata
import json
import math
import os
import platform
import re
import sqlite3
import threading
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

import numpy as np

from pricing_mapper.advisor import AdvisorDecisionRecord, allowed_diagnostic_bin_ids
from pricing_mapper.domain import FIELD_ORDER, CarQuoteInput, DomainSpec, row_key
from pricing_mapper.exceptions import PersistenceError
from pricing_mapper.provider import ProviderAttempt

RUN_SCHEMA_VERSION = 1
SplitName = Literal["mapping", "validation", "calibration", "audit"]
_SPLITS = {"mapping", "validation", "calibration", "audit"}
_STATE_TABLES = {
    "metadata",
    "batches",
    "samples",
    "quote_cache",
    "provider_attempts",
    "rng_states",
}
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_RESUME_DEPENDENCIES = (
    "numpy",
    "pandas",
    "pydantic",
    "scikit-learn",
    "threadpoolctl",
    "pricing-function-mapper",
)


def validate_run_id(run_id: str) -> str:
    """Reject path-like, empty, and unbounded run identifiers."""

    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must be 1-64 characters using letters, digits, '.', '_', or '-' "
            "and must begin with a letter or digit"
        )
    return run_id


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in _RESUME_DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name == "pricing-function-mapper":
                versions[name] = "1.0.0"
            else:
                raise PersistenceError(
                    f"required runtime dependency {name!r} is not installed"
                ) from None
    return versions


class RunLock:
    """Advisory exclusive lock held for the complete run/export critical section."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle: Any = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.close()
            self._handle = None
            raise PersistenceError(f"run is already locked: {self.path}") from exc
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(f"pid={os.getpid()}\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


@dataclass(frozen=True)
class SampleRecord:
    id: int
    ordinal: int
    split: SplitName
    batch_id: int | None
    position: int
    row_hash: str
    row: dict[str, Any]
    premium: float | None
    source: str
    status: str

    def validated_input(self, domain: DomainSpec) -> CarQuoteInput:
        return CarQuoteInput.from_mapping(self.row, domain=domain)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")


def _json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def _json_loads(value: str, context: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise PersistenceError(f"{context} JSON is corrupt: {exc}") from exc


class RunStore:
    """Durable state for samples, quote cache, RNG, batches, and telemetry."""

    def __init__(self, path: Path, connection: sqlite3.Connection) -> None:
        self.path = path
        self._connection = connection
        self._mutex = threading.RLock()
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._connection.execute("PRAGMA busy_timeout = 30000")

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        config_fingerprint: str,
        config_toml: str,
        domain: DomainSpec,
        provider_identity: str,
    ) -> RunStore:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"run state already exists: {target}")
        connection = sqlite3.connect(target, isolation_level=None, check_same_thread=False)
        store = cls(target, connection)
        try:
            store._connection.execute("PRAGMA journal_mode = WAL")
            store._create_schema()
            with store.transaction():
                store._set_metadata_unlocked("schema_version", RUN_SCHEMA_VERSION)
                store._set_metadata_unlocked("config_fingerprint", config_fingerprint)
                store._set_metadata_unlocked("config_toml", config_toml)
                store._set_metadata_unlocked("domain", domain.to_dict())
                store._set_metadata_unlocked("provider_identity", provider_identity)
                store._set_metadata_unlocked("runtime_versions", _runtime_versions())
                store._set_metadata_unlocked("early_stopped", False)
                store._set_metadata_unlocked("stop_reason", None)
        except Exception:
            store.close()
            raise
        return store

    @classmethod
    def open(cls, path: str | Path) -> RunStore:
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"cannot resume: SQLite run state does not exist: {target}")
        uri = f"{target.resolve().as_uri()}?mode=rw"
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                uri,
                uri=True,
                isolation_level=None,
                check_same_thread=False,
            )
            store = cls(target, connection)
            version_row = store._connection.execute("PRAGMA user_version").fetchone()
            version = int(version_row[0]) if version_row is not None else -1
            if version != RUN_SCHEMA_VERSION:
                store.close()
                raise PersistenceError(
                    f"unsupported SQLite run schema version {version}; expected "
                    f"{RUN_SCHEMA_VERSION}. v0 JSON state cannot be resumed in v1"
                )
            checks = store._connection.execute("PRAGMA quick_check(1)").fetchall()
            if not checks or any(str(row[0]) != "ok" for row in checks):
                store.close()
                raise PersistenceError(f"SQLite run state failed integrity check: {target}")
            table_rows = store._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
            tables = {str(row[0]) for row in table_rows}
            missing = _STATE_TABLES - tables
            if missing:
                store.close()
                raise PersistenceError(
                    f"SQLite run state is missing required tables: {sorted(missing)}"
                )
            batch_columns = {
                str(row[1])
                for row in store._connection.execute("PRAGMA table_info(batches)").fetchall()
            }
            required_batch_columns = {
                "batch_id",
                "status",
                "metrics_json",
                "advisor_json",
            }
            missing_batch_columns = required_batch_columns - batch_columns
            if missing_batch_columns:
                store.close()
                raise PersistenceError(
                    "SQLite run state uses an incompatible batches schema; missing columns: "
                    f"{sorted(missing_batch_columns)}"
                )
            return store
        except sqlite3.DatabaseError as exc:
            if connection is not None:
                connection.close()
            raise PersistenceError(f"cannot open SQLite run state {target}: {exc}") from exc

    def _create_schema(self) -> None:
        schema = """
        BEGIN IMMEDIATE;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        CREATE TABLE batches (
            batch_id INTEGER PRIMARY KEY,
            status TEXT NOT NULL CHECK (status IN ('generated', 'quoted', 'evaluated')),
            metrics_json TEXT,
            advisor_json TEXT,
            UNIQUE(batch_id)
        );
        CREATE TABLE samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ordinal INTEGER NOT NULL UNIQUE,
            split TEXT NOT NULL CHECK (
                split IN ('mapping', 'validation', 'calibration', 'audit')
            ),
            batch_id INTEGER REFERENCES batches(batch_id),
            position INTEGER NOT NULL,
            row_hash TEXT NOT NULL UNIQUE,
            row_json TEXT NOT NULL,
            premium REAL,
            source TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('pending', 'complete')),
            UNIQUE(split, batch_id, position),
            CHECK (
                (status = 'pending' AND premium IS NULL)
                OR (status = 'complete' AND premium IS NOT NULL AND premium >= 0)
            )
        );
        CREATE INDEX samples_split_status ON samples(split, status, ordinal);
        CREATE TABLE quote_cache (
            provider_identity TEXT NOT NULL,
            row_hash TEXT NOT NULL,
            premium REAL NOT NULL CHECK (premium >= 0),
            PRIMARY KEY(provider_identity, row_hash)
        );
        CREATE TABLE provider_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider_identity TEXT NOT NULL,
            row_hash TEXT NOT NULL,
            attempt_number INTEGER NOT NULL CHECK (attempt_number >= 1),
            outcome TEXT NOT NULL CHECK (
                outcome IN ('success', 'retryable_failure', 'permanent_failure')
            ),
            latency_ms REAL NOT NULL CHECK (latency_ms >= 0),
            error_type TEXT,
            occurred_at_utc TEXT NOT NULL
        );
        CREATE TABLE rng_states (
            name TEXT PRIMARY KEY,
            state_json TEXT NOT NULL
        );
        PRAGMA user_version = 1;
        COMMIT;
        """
        try:
            self._connection.executescript(schema)
        except sqlite3.DatabaseError as exc:
            raise PersistenceError(f"cannot initialize run database: {exc}") from exc

    def transaction(self) -> Any:
        """Return a lock-aware SQLite transaction context manager."""

        return _Transaction(self)

    def close(self) -> None:
        with self._mutex:
            with suppress(sqlite3.DatabaseError):
                self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _set_metadata_unlocked(self, key: str, value: Any) -> None:
        self._connection.execute(
            """
            INSERT INTO metadata(key, value_json) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
            """,
            (key, _json_dumps(value)),
        )

    def set_metadata(self, key: str, value: Any) -> None:
        with self.transaction():
            self._set_metadata_unlocked(key, value)

    def metadata(self, key: str, default: Any = None) -> Any:
        with self._mutex:
            row = self._connection.execute(
                "SELECT value_json FROM metadata WHERE key = ?",
                (key,),
            ).fetchone()
        return default if row is None else _json_loads(str(row["value_json"]), f"metadata {key!r}")

    def save_rng_state(self, name: str, state: Mapping[str, Any]) -> None:
        with self.transaction():
            self._save_rng_state_unlocked(name, state)

    def _save_rng_state_unlocked(self, name: str, state: Mapping[str, Any]) -> None:
        self._connection.execute(
            """
            INSERT INTO rng_states(name, state_json) VALUES (?, ?)
            ON CONFLICT(name) DO UPDATE SET state_json = excluded.state_json
            """,
            (name, _json_dumps(dict(state))),
        )

    def rng_state(self, name: str) -> dict[str, Any]:
        with self._mutex:
            row = self._connection.execute(
                "SELECT state_json FROM rng_states WHERE name = ?",
                (name,),
            ).fetchone()
        if row is None:
            raise PersistenceError(f"run state is missing RNG stream {name!r}")
        raw = _json_loads(str(row["state_json"]), f"RNG stream {name!r}")
        if not isinstance(raw, dict):
            raise PersistenceError(f"RNG stream {name!r} is corrupt")
        return raw

    def register_samples(
        self,
        split: SplitName,
        rows: Iterable[Mapping[str, Any] | CarQuoteInput],
        *,
        batch_id: int | None,
        source: str,
        mapping_rng_state: Mapping[str, Any] | None = None,
        advisor_decision: Mapping[str, Any] | None = None,
    ) -> list[SampleRecord]:
        if split not in _SPLITS:
            raise ValueError(f"unknown sample split {split!r}")
        validated_advisor: dict[str, Any] | None = None
        if advisor_decision is not None:
            try:
                parsed_advisor = AdvisorDecisionRecord.model_validate(
                    dict(advisor_decision),
                    strict=True,
                )
            except (TypeError, ValueError) as exc:
                raise PersistenceError("advisor decision cannot be registered") from exc
            if parsed_advisor.batch_id != batch_id or parsed_advisor.runtime.model_dump(
                mode="json"
            ) != self.metadata("advisor_runtime"):
                raise PersistenceError("advisor decision cannot be registered")
            validated_advisor = parsed_advisor.model_dump(mode="json")
        materialized = [
            item.as_dict() if isinstance(item, CarQuoteInput) else dict(item) for item in rows
        ]
        if not materialized:
            return []
        with self.transaction():
            if split == "mapping":
                if batch_id is None or batch_id < 0:
                    raise ValueError("mapping samples require a non-negative batch_id")
                self._connection.execute(
                    """
                    INSERT INTO batches(batch_id, status, advisor_json)
                    VALUES (?, 'generated', ?)
                    """,
                    (
                        batch_id,
                        None if validated_advisor is None else _json_dumps(validated_advisor),
                    ),
                )
            elif batch_id is not None:
                raise ValueError("evaluation samples cannot have a mapping batch_id")
            elif advisor_decision is not None:
                raise ValueError("evaluation samples cannot have an advisor decision")

            row = self._connection.execute(
                "SELECT COALESCE(MAX(ordinal), -1) + 1 AS next_ordinal FROM samples"
            ).fetchone()
            next_ordinal = int(row["next_ordinal"])
            ids: list[int] = []
            for position, values in enumerate(materialized):
                canonical_json = _json_dumps({name: values[name] for name in FIELD_ORDER})
                hashed = row_key(values)
                try:
                    cursor = self._connection.execute(
                        """
                        INSERT INTO samples(
                            ordinal, split, batch_id, position, row_hash, row_json,
                            premium, source, status
                        ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, 'pending')
                        """,
                        (
                            next_ordinal + position,
                            split,
                            batch_id,
                            position,
                            hashed,
                            canonical_json,
                            source,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise PersistenceError(
                        f"sample registration collided with an existing row hash {hashed[:12]}"
                    ) from exc
                inserted_id = cursor.lastrowid
                if inserted_id is None:
                    raise PersistenceError("SQLite did not return an inserted sample id")
                ids.append(int(inserted_id))
            if mapping_rng_state is not None:
                self._save_rng_state_unlocked("mapping", mapping_rng_state)
        return [self.sample_by_id(sample_id) for sample_id in ids]

    def register_evaluation_splits(
        self,
        splits: Mapping[SplitName, Sequence[CarQuoteInput]],
    ) -> None:
        """Register every pre-generated holdout in one crash-safe transaction."""

        expected: set[SplitName] = {"validation", "calibration", "audit"}
        if set(splits) != expected:
            raise ValueError("evaluation registration requires validation/calibration/audit")
        if any(not splits[name] for name in expected):
            raise ValueError("every evaluation split must contain at least one row")
        with self.transaction():
            next_ordinal = 0
            for split in ("validation", "calibration", "audit"):
                for position, quote in enumerate(splits[split]):
                    values = quote.as_dict()
                    hashed = row_key(quote)
                    try:
                        self._connection.execute(
                            """
                            INSERT INTO samples(
                                ordinal, split, batch_id, position, row_hash, row_json,
                                premium, source, status
                            ) VALUES (?, ?, NULL, ?, ?, ?, NULL, 'evaluation_lhs', 'pending')
                            """,
                            (
                                next_ordinal,
                                split,
                                position,
                                hashed,
                                _json_dumps(values),
                            ),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise PersistenceError(
                            "pre-generated evaluation splits contain duplicate rows"
                        ) from exc
                    next_ordinal += 1
            self._set_metadata_unlocked("evaluation_generated", True)

    def register_seed_samples(
        self,
        rows: Iterable[tuple[CarQuoteInput, float]],
        *,
        batch_id: int = 0,
    ) -> list[SampleRecord]:
        materialized = list(rows)
        if not materialized:
            return []
        with self.transaction():
            self._connection.execute(
                "INSERT INTO batches(batch_id, status) VALUES (?, 'quoted')",
                (batch_id,),
            )
            start_row = self._connection.execute(
                "SELECT COALESCE(MAX(ordinal), -1) + 1 AS next_ordinal FROM samples"
            ).fetchone()
            next_ordinal = int(start_row["next_ordinal"])
            ids: list[int] = []
            for position, (quote, premium) in enumerate(materialized):
                if not math.isfinite(premium) or premium < 0:
                    raise PersistenceError("seed premiums must be finite and non-negative")
                values = quote.as_dict()
                hashed = row_key(quote)
                try:
                    cursor = self._connection.execute(
                        """
                        INSERT INTO samples(
                            ordinal, split, batch_id, position, row_hash, row_json,
                            premium, source, status
                        ) VALUES (?, 'mapping', ?, ?, ?, ?, ?, 'seed_data', 'complete')
                        """,
                        (
                            next_ordinal + position,
                            batch_id,
                            position,
                            hashed,
                            _json_dumps(values),
                            premium,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise PersistenceError(
                        "seed data contains duplicates or overlaps an evaluation holdout "
                        f"(row hash {hashed[:12]})"
                    ) from exc
                inserted_id = cursor.lastrowid
                if inserted_id is None:
                    raise PersistenceError("SQLite did not return an inserted seed id")
                ids.append(int(inserted_id))
        return [self.sample_by_id(sample_id) for sample_id in ids]

    def sample_by_id(self, sample_id: int) -> SampleRecord:
        with self._mutex:
            row = self._connection.execute(
                "SELECT * FROM samples WHERE id = ?",
                (sample_id,),
            ).fetchone()
        if row is None:
            raise PersistenceError(f"sample id {sample_id} does not exist")
        return self._sample_from_row(row)

    @staticmethod
    def _sample_from_row(row: sqlite3.Row) -> SampleRecord:
        raw = _json_loads(str(row["row_json"]), f"sample {row['id']}")
        if not isinstance(raw, dict):
            raise PersistenceError("sample row JSON is corrupt")
        split = str(row["split"])
        if split not in _SPLITS:
            raise PersistenceError(f"sample has invalid split {split!r}")
        premium_raw = row["premium"]
        return SampleRecord(
            id=int(row["id"]),
            ordinal=int(row["ordinal"]),
            split=split,  # type: ignore[arg-type]
            batch_id=None if row["batch_id"] is None else int(row["batch_id"]),
            position=int(row["position"]),
            row_hash=str(row["row_hash"]),
            row=raw,
            premium=None if premium_raw is None else float(premium_raw),
            source=str(row["source"]),
            status=str(row["status"]),
        )

    def samples(
        self,
        split: SplitName | None = None,
        *,
        completed_only: bool = False,
        batch_id: int | None = None,
    ) -> list[SampleRecord]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if split is not None:
            clauses.append("split = ?")
            parameters.append(split)
        if completed_only:
            clauses.append("status = 'complete'")
        if batch_id is not None:
            clauses.append("batch_id = ?")
            parameters.append(batch_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._mutex:
            rows = self._connection.execute(
                f"SELECT * FROM samples{where} ORDER BY ordinal",  # noqa: S608
                parameters,
            ).fetchall()
        return [self._sample_from_row(row) for row in rows]

    def pending_samples(self, splits: Iterable[SplitName] | None = None) -> list[SampleRecord]:
        parameters: list[Any] = []
        clause = "status = 'pending'"
        if splits is not None:
            selected = list(splits)
            if not selected:
                return []
            if any(split not in _SPLITS for split in selected):
                raise ValueError("pending sample query contains an invalid split")
            placeholders = ",".join("?" for _ in selected)
            clause += f" AND split IN ({placeholders})"
            parameters.extend(selected)
        with self._mutex:
            rows = self._connection.execute(
                f"SELECT * FROM samples WHERE {clause} ORDER BY ordinal",  # noqa: S608
                parameters,
            ).fetchall()
        return [self._sample_from_row(row) for row in rows]

    def cached_quote(self, provider_identity: str, hashed_row: str) -> float | None:
        with self._mutex:
            row = self._connection.execute(
                """
                SELECT premium FROM quote_cache
                WHERE provider_identity = ? AND row_hash = ?
                """,
                (provider_identity, hashed_row),
            ).fetchone()
        return None if row is None else float(row["premium"])

    def complete_quote(
        self,
        sample_id: int,
        hashed_row: str,
        premium: float,
        provider_identity: str,
    ) -> None:
        """Persist one completed quote and its cache entry in one transaction."""

        if not math.isfinite(premium) or premium < 0:
            raise PersistenceError("completed quote must be finite and non-negative")
        with self.transaction():
            row = self._connection.execute(
                "SELECT row_hash, status, batch_id FROM samples WHERE id = ?",
                (sample_id,),
            ).fetchone()
            if row is None:
                raise PersistenceError(f"sample id {sample_id} does not exist")
            if str(row["row_hash"]) != hashed_row:
                raise PersistenceError("completed quote row hash does not match the sample")
            if str(row["status"]) == "complete":
                existing = self._connection.execute(
                    "SELECT premium FROM samples WHERE id = ?",
                    (sample_id,),
                ).fetchone()
                if existing is None or not math.isclose(
                    float(existing["premium"]),
                    premium,
                    rel_tol=0.0,
                    abs_tol=0.0,
                ):
                    raise PersistenceError("completed sample was assigned a different premium")
                return
            self._connection.execute(
                "UPDATE samples SET premium = ?, status = 'complete' WHERE id = ?",
                (premium, sample_id),
            )
            self._connection.execute(
                """
                INSERT INTO quote_cache(provider_identity, row_hash, premium)
                VALUES (?, ?, ?)
                ON CONFLICT(provider_identity, row_hash)
                DO UPDATE SET premium = excluded.premium
                """,
                (provider_identity, hashed_row, premium),
            )
            batch_id = row["batch_id"]
            if batch_id is not None:
                pending = self._connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM samples
                    WHERE batch_id = ? AND status = 'pending'
                    """,
                    (int(batch_id),),
                ).fetchone()
                if pending is not None and int(pending["count"]) == 0:
                    self._connection.execute(
                        "UPDATE batches SET status = 'quoted' WHERE batch_id = ?",
                        (int(batch_id),),
                    )

    def record_attempt(self, attempt: ProviderAttempt) -> None:
        with self.transaction():
            self._connection.execute(
                """
                INSERT INTO provider_attempts(
                    provider_identity, row_hash, attempt_number, outcome,
                    latency_ms, error_type, occurred_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt.provider_identity,
                    attempt.row_hash,
                    attempt.attempt_number,
                    attempt.outcome,
                    attempt.latency_ms,
                    attempt.error_type,
                    attempt.occurred_at_utc,
                ),
            )

    def provider_summary(self) -> dict[str, Any]:
        with self._mutex:
            rows = self._connection.execute("""
                SELECT outcome, COUNT(*) AS count, COALESCE(SUM(latency_ms), 0) AS latency
                FROM provider_attempts GROUP BY outcome
                """).fetchall()
        counts = {str(row["outcome"]): int(row["count"]) for row in rows}
        total_latency = sum(float(row["latency"]) for row in rows)
        return {
            "identity": self.metadata("provider_identity"),
            "attempts": sum(counts.values()),
            "successes": counts.get("success", 0),
            "retryable_failures": counts.get("retryable_failure", 0),
            "permanent_failures": counts.get("permanent_failure", 0),
            "total_latency_ms": total_latency,
        }

    def unevaluated_batches(self) -> list[int]:
        with self._mutex:
            rows = self._connection.execute(
                "SELECT batch_id FROM batches WHERE status = 'quoted' ORDER BY batch_id"
            ).fetchall()
        return [int(row["batch_id"]) for row in rows]

    def mark_batch_evaluated(self, batch_id: int, metrics: Mapping[str, Any]) -> None:
        with self.transaction():
            cursor = self._connection.execute(
                """
                UPDATE batches SET status = 'evaluated', metrics_json = ?
                WHERE batch_id = ? AND status = 'quoted'
                """,
                (_json_dumps(dict(metrics)), batch_id),
            )
            if cursor.rowcount != 1:
                raise PersistenceError(
                    f"batch {batch_id} cannot be evaluated from its current state"
                )

    def batch_history(self) -> list[dict[str, Any]]:
        with self._mutex:
            rows = self._connection.execute("""
                SELECT batch_id, status, metrics_json, advisor_json
                FROM batches ORDER BY batch_id
                """).fetchall()
        history: list[dict[str, Any]] = []
        for row in rows:
            raw_metrics = row["metrics_json"]
            history.append(
                {
                    "batch_id": int(row["batch_id"]),
                    "status": str(row["status"]),
                    "advisor": (
                        None
                        if row["advisor_json"] is None
                        else _json_loads(
                            str(row["advisor_json"]),
                            f"batch {row['batch_id']} advisor decision",
                        )
                    ),
                    "metrics": (
                        None
                        if raw_metrics is None
                        else _json_loads(
                            str(raw_metrics),
                            f"batch {row['batch_id']} metrics",
                        )
                    ),
                }
            )
        return history

    def advisor_decisions(self) -> list[dict[str, Any]]:
        decisions: list[dict[str, Any]] = []
        for batch in self.batch_history():
            raw = batch["advisor"]
            if raw is None:
                continue
            if not isinstance(raw, dict):
                raise PersistenceError(
                    f"batch {batch['batch_id']} advisor decision must be an object"
                )
            decisions.append(raw)
        return decisions

    def next_batch_id(self) -> int:
        with self._mutex:
            row = self._connection.execute(
                "SELECT COALESCE(MAX(batch_id), -1) + 1 AS next_id FROM batches"
            ).fetchone()
        return int(row["next_id"])

    def all_row_hashes(self) -> set[str]:
        with self._mutex:
            rows = self._connection.execute("SELECT row_hash FROM samples").fetchall()
        return {str(row["row_hash"]) for row in rows}

    def validate_integrity(
        self,
        *,
        config_fingerprint: str,
        domain: DomainSpec,
        provider_identity: str,
    ) -> None:
        if self.metadata("schema_version") != RUN_SCHEMA_VERSION:
            raise PersistenceError("run metadata schema version does not match SQLite schema")
        if self.metadata("config_fingerprint") != config_fingerprint:
            raise PersistenceError(
                "resume configuration fingerprint does not match the original v1 run"
            )
        if self.metadata("provider_identity") != provider_identity:
            raise PersistenceError("resume provider identity does not match the original run")
        if self.metadata("runtime_versions") != _runtime_versions():
            raise PersistenceError(
                "resume Python or dependency versions do not match the original run"
            )
        if self.metadata("domain") != domain.to_dict():
            raise PersistenceError("resume domain snapshot does not match the original run")

        seen: set[str] = set()
        for sample in self.samples():
            quote = sample.validated_input(domain)
            expected_hash = row_key(quote)
            if expected_hash != sample.row_hash:
                raise PersistenceError(f"sample {sample.id} has a corrupted row hash")
            if sample.row_hash in seen:
                raise PersistenceError("run state contains duplicate sample hashes")
            seen.add(sample.row_hash)
            if sample.status == "complete":
                if (
                    sample.premium is None
                    or not math.isfinite(sample.premium)
                    or sample.premium < 0
                ):
                    raise PersistenceError(f"sample {sample.id} has an invalid premium")
            elif sample.premium is not None:
                raise PersistenceError(f"pending sample {sample.id} unexpectedly has a premium")

        for batch in self.batch_history():
            advisor = batch["advisor"]
            if advisor is None:
                continue
            try:
                parsed_advisor = AdvisorDecisionRecord.model_validate(advisor, strict=True)
            except (TypeError, ValueError) as exc:
                raise PersistenceError(
                    f"batch {batch['batch_id']} has a corrupt advisor decision"
                ) from exc
            stored_runtime = self.metadata("advisor_runtime")
            if (
                parsed_advisor.batch_id != batch["batch_id"]
                or parsed_advisor.runtime.model_dump(mode="json") != stored_runtime
                or any(
                    boost.bin_id not in allowed_diagnostic_bin_ids(domain)
                    for boost in parsed_advisor.response.bin_boosts
                )
            ):
                raise PersistenceError(f"batch {batch['batch_id']} has a corrupt advisor decision")

        holdout_splits: tuple[SplitName, ...] = ("validation", "calibration", "audit")
        for split in holdout_splits:
            if not self.samples(split):
                raise PersistenceError(f"run state is missing the {split} holdout split")


class _Transaction:
    def __init__(self, store: RunStore) -> None:
        self.store = store

    def __enter__(self) -> _Transaction:
        self.store._mutex.acquire()
        try:
            self.store._connection.execute("BEGIN IMMEDIATE")
        except sqlite3.DatabaseError as exc:
            self.store._mutex.release()
            raise PersistenceError(f"cannot begin SQLite transaction: {exc}") from exc
        except BaseException:
            self.store._mutex.release()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            try:
                if exc_type is None:
                    self.store._connection.execute("COMMIT")
                else:
                    self.store._connection.execute("ROLLBACK")
            except sqlite3.DatabaseError as exc:
                if exc_type is None:
                    raise PersistenceError(f"cannot commit SQLite transaction: {exc}") from exc
        finally:
            self.store._mutex.release()
