from __future__ import annotations

import pickle
import struct
from pathlib import Path

import numpy as np
from loguru import logger

from storage.sqlite_cache import SqliteZstdCache


class SearchVectorCache(SqliteZstdCache):
    def __init__(self, *, db_path: Path | None = None):
        super().__init__(name="vec", db_path=db_path)

    def _key(self, text: str, backend: str) -> str:
        t = self.require_non_empty(text, name="text")
        b = self.require_non_empty(backend, name="backend")
        return self.sha256_key(b, t)

    @staticmethod
    def _normalize_vec(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        return np.ascontiguousarray(arr)

    def get(self, text: str, backend: str) -> np.ndarray | None:
        key = self._key(text, backend)
        if not self.db_path.exists():
            return None
        try:
            raw = self.read_raw(key)
            if raw is None:
                return None
            # Fast path (new format): raw float32 bytes, no pickle/zstd.
            if raw.startswith(b"VEC1") and len(raw) >= 8:
                dim = int(struct.unpack_from("<I", raw, 4)[0])
                data = raw[8:]
                if dim <= 0 or dim > 65536:
                    raise ValueError(f"bad dim {dim}")
                if len(data) != dim * 4:
                    raise ValueError("bad vec payload length")
                vec = np.frombuffer(data, dtype=np.float32).copy()
            else:
                # Backward compatibility: zstd(pickle({...}))
                payload = self.decompress_bytes(raw)
                obj = pickle.loads(payload)
                shape = tuple(obj["shape"])
                dtype = str(obj["dtype"])
                data = obj["data"]
                if dtype != "float32":
                    raise ValueError(f"unexpected dtype {dtype}")
                vec = np.frombuffer(data, dtype=np.float32).reshape(shape).copy()
            logger.debug(
                "vec_cache_hit backend={} text={} dim={} bytes={}",
                backend,
                text[:80],
                int(vec.size),
                int(vec.size * 4),
            )
            return vec
        except Exception as e:
            logger.warning(
                "vec_cache_read_error backend={} text={} err={}",
                backend,
                text[:80],
                str(e),
            )
            try:
                self.delete(text, backend)
            except Exception:
                pass
            return None

    def set(self, text: str, backend: str, vec: np.ndarray) -> None:
        key = self._key(text, backend)
        try:
            v = self._normalize_vec(vec)
            # Fast format: store raw float32 bytes (no pickle/zstd for speed).
            payload = v.tobytes()
            raw = b"VEC1" + struct.pack("<I", int(v.size)) + payload
            self.write_raw(key, raw)
            logger.debug(
                "vec_cache_set backend={} text={} dim={} bytes={} compressed={}",
                backend,
                text[:80],
                int(v.size),
                int(len(payload)),
                int(len(raw)),
            )
        except Exception as e:
            logger.error(
                "vec_cache_write_error backend={} text={} err={}",
                backend,
                text[:80],
                str(e),
            )

    def delete(self, text: str, backend: str) -> bool:
        key = self._key(text, backend)
        try:
            return self.delete_key(key)
        except Exception:
            return False


_VEC_CACHE: SearchVectorCache | None = None


def get_search_vector_cache() -> SearchVectorCache:
    global _VEC_CACHE
    if _VEC_CACHE is None:
        _VEC_CACHE = SearchVectorCache()
    return _VEC_CACHE
