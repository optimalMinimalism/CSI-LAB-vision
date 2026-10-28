"""Persistenza transazionale SQLite, JPEG canonici nel database."""

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from framegrabber.protocol import corrected_time, experiment_id, integer, verify_frame


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(
            self.root / "experiments.sqlite", check_same_thread=False
        )
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
          PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;
          CREATE TABLE IF NOT EXISTS experiments(id TEXT PRIMARY KEY, body TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS frames(experiment TEXT REFERENCES experiments(id),
            sequence INTEGER, timestamp INTEGER, body TEXT NOT NULL, jpeg BLOB NOT NULL,
            PRIMARY KEY(experiment, sequence));
          CREATE TABLE IF NOT EXISTS artifacts(experiment TEXT REFERENCES experiments(id),
            kind TEXT, body TEXT NOT NULL, PRIMARY KEY(experiment, kind));
          CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, experiment TEXT, body TEXT NOT NULL);
        """)
        with self.lock, self.db:
            for row in self.db.execute("SELECT id,body FROM experiments").fetchall():
                item = json.loads(row["body"])
                if item["state"] in {"STARTING", "RECORDING", "STOPPING"}:
                    item.update(
                        state="INTERRUPTED",
                        end_ns=time.time_ns(),
                        stop_error="Processo terminato durante la registrazione",
                    )
                    self.db.execute(
                        "UPDATE experiments SET body=? WHERE id=?",
                        (dumps(item), row["id"]),
                    )

    def create(self, camera, title):
        item = dict(
            id=str(uuid.uuid4()),
            title=str(title)[:200],
            start_ns=time.time_ns(),
            end_ns=None,
            state="STARTING",
            camera_enabled=bool(camera),
            camera_ack=False,
            camera_telemetry={},
            created_by="csi-video-ground-truth",
        )
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO experiments VALUES(?,?)", (item["id"], dumps(item))
            )
        return item

    def get(self, identifier):
        experiment_id(identifier)
        with self.lock:
            row = self.db.execute(
                "SELECT body FROM experiments WHERE id=?", (identifier,)
            ).fetchone()
        if row is None:
            raise KeyError("Esperimento inesistente")
        return json.loads(row[0])

    def update(self, identifier, **changes):
        with self.lock, self.db:
            item = self.get(identifier)
            item.update(changes)
            self.db.execute(
                "UPDATE experiments SET body=? WHERE id=?", (dumps(item), identifier)
            )
        return item

    def list(self):
        with self.lock:
            return [
                json.loads(r[0])
                for r in self.db.execute(
                    "SELECT body FROM experiments ORDER BY rowid DESC"
                )
            ]

    def frame(self, identifier, sequence):
        self.get(identifier)
        with self.lock:
            row = self.db.execute(
                "SELECT body,jpeg FROM frames WHERE experiment=? AND sequence=?",
                (identifier, sequence),
            ).fetchone()
        if not row:
            raise KeyError("Frame inesistente")
        return json.loads(row[0]), row[1]

    def frames(self, identifier):
        self.get(identifier)
        with self.lock:
            return [
                json.loads(r[0])
                for r in self.db.execute(
                    "SELECT body FROM frames WHERE experiment=? ORDER BY timestamp,sequence",
                    (identifier,),
                )
            ]

    def receive(self, identifier, metadata, payload, server_id):
        verify_frame(metadata, payload)
        timestamp, uncertainty, fresh = corrected_time(metadata, server_id)
        with self.lock, self.db:
            item = self.get(identifier)
            old = self.db.execute(
                "SELECT body FROM frames WHERE experiment=? AND sequence=?",
                (identifier, metadata["frame_id"]),
            ).fetchone()
            if old:
                original = json.loads(old[0])
                # Retry must be an exact retry, not only identical pixels.
                if any(original.get(k) != v for k, v in metadata.items()):
                    raise ValueError("frame_id già utilizzato con metadati differenti")
                return dict(duplicate=True)
            if not item["camera_enabled"] or item["state"] not in {
                "RECORDING",
                "STOPPING",
            }:
                raise ValueError("Registrazione video non attiva")
            if timestamp < item["start_ns"] or (
                item["end_ns"] and timestamp > item["end_ns"]
            ):
                raise ValueError("Frame fuori intervallo di acquisizione")
            if timestamp > time.time_ns() + 2_000_000_000:
                raise ValueError("Frame nel futuro")
            previous = self.db.execute(
                "SELECT body FROM frames WHERE experiment=? ORDER BY sequence DESC LIMIT 1",
                (identifier,),
            ).fetchone()
            record = dict(
                metadata,
                aligned_timestamp_ns=timestamp,
                sync_uncertainty_ns=uncertainty,
                clock_fresh=fresh,
                received_ns=time.time_ns(),
                clock_step=False,
            )
            if previous:
                prev = json.loads(previous[0])
                if metadata["frame_id"] <= prev["frame_id"]:
                    raise ValueError("Sequenza fuori ordine")
                if metadata["monotonic_ns"] <= prev["monotonic_ns"]:
                    raise ValueError("Clock monotono non crescente")
                delta = (metadata["timestamp_ns"] - prev["timestamp_ns"]) - (
                    metadata["monotonic_ns"] - prev["monotonic_ns"]
                )
                record["clock_step"] = abs(delta) > 5_000_000
                if timestamp <= prev["aligned_timestamp_ns"]:
                    record["clock_step"] = True
            self.db.execute(
                "INSERT INTO frames VALUES(?,?,?,?,?)",
                (identifier, metadata["frame_id"], timestamp, dumps(record), payload),
            )
        return dict(duplicate=False)


    def artifact(self, identifier, kind, value=None):
        self.get(identifier)
        with self.lock, self.db:
            if value is not None:
                self.db.execute(
                    "INSERT OR REPLACE INTO artifacts VALUES(?,?,?)",
                    (identifier, kind, dumps(value)),
                )
            row = self.db.execute(
                "SELECT body FROM artifacts WHERE experiment=? AND kind=?",
                (identifier, kind),
            ).fetchone()
            return json.loads(row[0]) if row else []

    def audit(self, identifier, entry=None):
        with self.lock, self.db:
            self.get(identifier)
            if entry is not None:
                self.db.execute(
                    "INSERT INTO audit(experiment,body) VALUES(?,?)",
                    (identifier, dumps(entry)),
                )
            return [
                json.loads(r[0])
                for r in self.db.execute(
                    "SELECT body FROM audit WHERE experiment=? ORDER BY id",
                    (identifier,),
                )
            ]

    def close(self):
        with self.lock:
            self.db.close()
