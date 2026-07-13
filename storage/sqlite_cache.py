from __future__ import annotations

import hashlib
import os
import random
import sqlite3  # type: ignore
import threading
import time
from pathlib import Path
from typing import Final

import zstandard as zstd
from loguru import logger


class SqliteZstdCache:
    """
    Stable, multi-process/thread friendly cache backed by SQLite (WAL) + zstd.

    Design goals:
    - prioritize correctness / crash-resilience over peak throughput
    - tolerate multi-process concurrent access (WAL + busy_timeout + retry)
    - thread-safe by using per-thread connections (no cross-thread connection sharing)
    - quarantine corrupted DB and keep the app running
    """

    _CREATE_SQL: Final[str] = """
    CREATE TABLE IF NOT EXISTS kv (
      namespace TEXT NOT NULL,
      k         TEXT NOT NULL,
      v         BLOB NOT NULL,
      updated_at INTEGER NOT NULL,
      PRIMARY KEY (namespace, k)
    ) WITHOUT ROWID;
    """

    def __init__(
        self,
        *,
        name: str,
        db_path: Path | None = None,
        namespace: str | None = None,
        busy_timeout_ms: int = 8000,
        max_retries: int = 8,
    ):
        n = (name or "").strip()
        if not n:
            raise ValueError("name is empty")
        self.name = n
        self.namespace = (namespace or n).strip() or n

        base = Path(".cache").resolve()
        # Like gpt_judge, put each cache in its own folder: .cache/{name}/cache.db
        self.db_path = (db_path or (base / n / "cache.db")).resolve()

        # Migrate old location if exists: .cache/{n}.sqlite3 -> .cache/{n}/cache.db
        if db_path is None:
            old_path = (base / f"{n}.sqlite3").resolve()
            if not self.db_path.exists() and old_path.exists():
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    old_path.rename(self.db_path)
                    logger.info(
                        "migrated_cache name={} old={} new={}",
                        n,
                        str(old_path),
                        str(self.db_path),
                    )
                    # Also try to migrate -wal and -shm files if they exist
                    for suffix in ["-wal", "-shm"]:
                        old_ext = old_path.with_name(old_path.name + suffix)
                        if old_ext.exists():
                            new_ext = self.db_path.with_name(self.db_path.name + suffix)
                            old_ext.rename(new_ext)
                except Exception as e:
                    logger.warning("failed_to_migrate_cache name={} err={}", n, str(e))

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.busy_timeout_ms = int(busy_timeout_ms)
        self.max_retries = int(max_retries)

        self._local = threading.local()
        self._init_lock = threading.RLock()
        # Serialize write transactions for stability (read remains concurrent).
        self._write_lock = threading.RLock()
        self._disabled = False
        self._disabled_reason = ""

        self._init_db()

    @staticmethod
    def require_non_empty(value: str, *, name: str) -> str:
        v = (value or "").strip()
        if not v:
            raise ValueError(f"{name} is empty")
        return v

    @staticmethod
    def sha256_key(*parts: str) -> str:
        if not parts:
            raise ValueError("parts is empty")
        s = "||".join(parts)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    @staticmethod
    def _is_corruption_error(e: BaseException) -> bool:
        if isinstance(e, sqlite3.DatabaseError):
            msg = (str(e) or "").lower()
            return any(
                s in msg
                for s in (
                    "database disk image is malformed",
                    "file is not a database",
                    "malformed database schema",
                    "database is locked and cannot be accessed",
                    "disk i/o error",
                    "not an error",  # observed with some sqlite builds during corruption
                )
            )
        return False

    def _disable_and_quarantine(self, *, err: BaseException) -> None:
        if self._disabled:
            return
        self._disabled = True
        self._disabled_reason = f"{type(err).__name__}: {err}"

        ts = time.strftime("%Y%m%d-%H%M%S")
        src = self.db_path
        dst = src.with_name(f"{src.name}.corrupted.{ts}")
        try:
            if src.exists():
                src.rename(dst)
        except Exception as rename_err:
            logger.error(
                "cache_quarantine_failed name={} src={} dst={} err={}",
                self.name,
                str(src),
                str(dst),
                str(rename_err),
            )
        logger.error(
            "cache_disabled_due_to_corruption name={} db_path={} reason={} quarantined_to={}",
            self.name,
            str(src),
            self._disabled_reason,
            str(dst),
        )

        try:
            # Re-enable with a new empty DB
            self._disabled = False
            self._disabled_reason = ""
            self._close_thread_local_conn()
            self._init_db()
        except Exception as e:
            self._disabled = True
            self._disabled_reason = f"reinit_failed: {type(e).__name__}: {e}"

    def _close_thread_local_conn(self) -> None:
        conn = getattr(self._local, "conn", None)
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass
        try:
            self._local.conn = None
            self._local.pid = None
        except Exception:
            pass

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=max(1.0, self.busy_timeout_ms / 1000.0),
            isolation_level=None,  # autocommit; explicit BEGIN when needed
            check_same_thread=True,  # per-thread connections
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=FULL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute(f"PRAGMA busy_timeout={int(self.busy_timeout_ms)};")
        conn.execute("PRAGMA wal_autocheckpoint=2000;")
        conn.execute("PRAGMA mmap_size=268435456;")  # 256MB best-effort
        conn.execute("PRAGMA cache_size=-32768;")  # ~32MB page cache
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        if self._disabled:
            raise RuntimeError(f"cache disabled: {self._disabled_reason}")
        pid = os.getpid()
        conn = getattr(self._local, "conn", None)
        local_pid = getattr(self._local, "pid", None)
        if conn is None or local_pid != pid:
            conn = self._connect()
            self._local.conn = conn
            self._local.pid = pid
        return conn

    def _get_zstd_decompressor(self) -> zstd.ZstdDecompressor:
        # zstandard objects are C-backed; keep them thread-local for stability.
        zd = getattr(self._local, "zstd_decompressor", None)
        if zd is None:
            zd = zstd.ZstdDecompressor()
            self._local.zstd_decompressor = zd
        return zd

    def _init_db(self) -> None:
        with self._init_lock:
            conn = None
            try:
                conn = self._connect()
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(self._CREATE_SQL)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS kv_updated_at_idx ON kv(updated_at);"
                )
                conn.execute("COMMIT;")
            except BaseException as e:
                try:
                    if conn is not None:
                        conn.execute("ROLLBACK;")
                except Exception:
                    pass
                if self._is_corruption_error(e):
                    self._disable_and_quarantine(err=e)
                    return
                raise
            finally:
                try:
                    if conn is not None:
                        conn.close()
                except Exception:
                    pass

    def _retry_sleep_s(self, attempt: int) -> float:
        base = min(0.25, 0.01 * (2**attempt))
        return base + random.random() * 0.01

    def _with_retry(self, fn, *, op: str):
        last_err: BaseException | None = None
        for attempt in range(max(1, self.max_retries)):
            try:
                return fn()
            except sqlite3.OperationalError as e:
                msg = (str(e) or "").lower()
                if "database is locked" in msg or "database is busy" in msg:
                    last_err = e
                    time.sleep(self._retry_sleep_s(attempt))
                    continue
                raise
            except BaseException as e:
                if self._is_corruption_error(e):
                    self._disable_and_quarantine(err=e)
                    return None
                last_err = e
                break
        if last_err is not None:
            raise last_err
        raise RuntimeError(f"{op} failed with unknown error")

    def decompress_bytes(self, raw: bytes) -> bytes:
        return self._get_zstd_decompressor().decompress(raw)

    def read_raw(self, key: str) -> bytes | None:
        if self._disabled:
            return None

        def _op():
            conn = self._get_conn()
            cur = conn.execute(
                "SELECT v FROM kv WHERE namespace=? AND k=?;",
                (self.namespace, key),
            )
            row = cur.fetchone()
            return row[0] if row is not None else None

        try:
            return self._with_retry(_op, op="read_raw")
        except BaseException as e:
            if self._is_corruption_error(e):
                self._disable_and_quarantine(err=e)
                return None
            raise

    def write_raw(self, key: str, raw: bytes) -> None:
        if self._disabled:
            return

        now = int(time.time())

        def _op():
            conn = self._get_conn()
            with self._write_lock:
                conn.execute("BEGIN IMMEDIATE;")
                try:
                    conn.execute(
                        """
                        INSERT INTO kv(namespace, k, v, updated_at)
                        VALUES(?, ?, ?, ?)
                        ON CONFLICT(namespace, k) DO UPDATE SET
                          v=excluded.v,
                          updated_at=excluded.updated_at
                        ;
                        """,
                        (self.namespace, key, sqlite3.Binary(raw), now),
                    )
                    conn.execute("COMMIT;")
                except BaseException:
                    try:
                        conn.execute("ROLLBACK;")
                    except Exception:
                        pass
                    raise
                return True

        try:
            self._with_retry(_op, op="write_raw")
        except BaseException as e:
            if self._is_corruption_error(e):
                self._disable_and_quarantine(err=e)
                return
            raise

    def delete_key(self, key: str) -> bool:
        if self._disabled:
            return False

        def _op():
            conn = self._get_conn()
            with self._write_lock:
                conn.execute("BEGIN IMMEDIATE;")
                try:
                    cur = conn.execute(
                        "DELETE FROM kv WHERE namespace=? AND k=?;",
                        (self.namespace, key),
                    )
                    conn.execute("COMMIT;")
                except BaseException:
                    try:
                        conn.execute("ROLLBACK;")
                    except Exception:
                        pass
                    raise
                return bool(cur.rowcount and cur.rowcount > 0)

        try:
            v = self._with_retry(_op, op="delete_key")
            return bool(v)
        except BaseException as e:
            if self._is_corruption_error(e):
                self._disable_and_quarantine(err=e)
                return False
            raise
