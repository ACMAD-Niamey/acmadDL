"""Paced HTTP range reader for remote rasters (COG / GeoTIFF).

GDAL's own ``/vsicurl/`` driver is the wrong client for a rate-limited server.
Opening one CHIRPS-sized COG through it sends about 60 requests within a
fraction of a second: a directory listing and ~54 ``HEAD`` probes for sidecar
files that do not exist (``.aux.xml``, ``.ovr``, ``.msk``, ...), plus the
actual reads. None of that can be paced from Python, and the UCSB Climate
Hazards Center server blocks an address that keeps exceeding about two
requests per second (see ``catalog.yaml``, CHIRPS section).

:class:`RangeStore` hands GDAL read-only file objects it reads *through*, so
every byte comes from a request this module sends itself:

* one connection, one request at a time;
* a caller-supplied limiter is waited on before **every** request, so
  ``request_interval`` is a true floor on the gap between HTTP requests;
* reads are served from a chunk cache and adjacent missing chunks are fetched
  in one request, so a regional window costs a handful of requests per file
  (the header, then roughly one per row of tiles the window touches);
* no ``HEAD``, no directory listing, no sidecar probes;
* a refusal raises :class:`ServerBlockedError`, which is never retried and
  stops the whole pull: at once for HTTP 403 or a 429 that reports a block,
  and after three retries for a plain 429. Each plain 429 is waited out
  (15, 30, 60 s) and halves the pace of the rest of the pull, down to one
  request per 10 s.
"""
import io
import re
import time
import urllib.error
import urllib.request

from ._robust import ServerUnreachableError, refusal_wait, user_agent

_CHUNK = 1024 * 1024          # bytes per cached chunk and minimum request size
_TIMEOUT = 60.0               # seconds; a blocked address can also just hang
_REFUSAL_STATUSES = (403, 429)


class RangeStore:
    """The bytes of one remote file, fetched on demand by paced range requests.

    ``limiter`` is anything with a ``wait()`` method (``_RateLimiter``); it is
    called before each request. ``n_requests`` and ``n_bytes`` count what was
    actually sent and received. GDAL opens a file more than once while reading
    it; every :meth:`open` view shares this one cache, so no byte is requested
    twice.
    """

    def __init__(self, url, limiter=None, chunk_size=_CHUNK, timeout=_TIMEOUT):
        self.url = url
        self._limiter = limiter
        self._chunk = int(chunk_size)
        self._timeout = timeout
        self._chunks = {}
        self._size = None
        self.n_requests = 0
        self.n_bytes = 0
        self.n_refused = 0         # 429 answers that were waited out
        # First failed request. GDAL swallows exceptions raised inside its read
        # callbacks, so the caller checks this; once set, nothing more is sent.
        self.error = None

    def connect(self):
        """Fetch the first chunk (file header and total size). Raises on failure.

        Call this before handing the store to GDAL: an exception raised inside
        one of GDAL's read callbacks cannot propagate (it can abort the
        process), so the request most likely to fail — the first — is made
        here, in ordinary Python.
        """
        self._fetch(0, 1)
        return self

    def open(self):
        return PacedRangeFile(self)

    def opener(self, path, mode="rb", **kwargs):
        """``rasterio.open(url, opener=store.opener)``: only ``url`` itself exists.

        Anything else GDAL asks for (a sidecar or sibling file) is answered
        "not found" locally, without a request.
        """
        if str(path) != self.url or "r" not in mode:
            raise FileNotFoundError(path)
        return self.open()

    @property
    def size(self):
        if self._size is None:
            self._fetch(0, 1)          # the first chunk's Content-Range carries the size
        return self._size

    def read(self, start, n):
        """Bytes ``[start, start + n)``, clipped to the file."""
        stop = min(start + n, self.size)
        if stop <= start:
            return b""
        first, last = start // self._chunk, (stop - 1) // self._chunk
        missing = [c for c in range(first, last + 1) if c not in self._chunks]
        if missing:
            # One request per contiguous run of missing chunks.
            run_start = prev = missing[0]
            for c in missing[1:] + [None]:
                if c is not None and c == prev + 1:
                    prev = c
                    continue
                self._fetch(run_start, prev + 1)
                run_start = prev = c
        data = b"".join(self._chunks[c] for c in range(first, last + 1))
        offset = start - first * self._chunk
        return data[offset:offset + (stop - start)]

    # -- network ------------------------------------------------------------
    def _fetch(self, first_chunk, stop_chunk):
        """Request chunks ``[first_chunk, stop_chunk)`` in a single range GET."""
        if self.error is not None:
            raise self.error
        try:
            self._request(first_chunk, stop_chunk)
        except Exception as exc:
            self.error = exc
            raise

    def _request(self, first_chunk, stop_chunk):
        lo = first_chunk * self._chunk
        hi = stop_chunk * self._chunk - 1
        if self._size is not None:
            hi = min(hi, self._size - 1)
        request = urllib.request.Request(self.url, headers={
            "Range": f"bytes={lo}-{hi}",
            "User-Agent": user_agent(),
            "Accept-Encoding": "identity",
        })
        attempt = 0
        while True:
            if self._limiter is not None:
                self._limiter.wait()
            self.n_requests += 1
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    status = response.status
                    content_range = response.headers.get("Content-Range", "")
                    body = response.read()
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in _REFUSAL_STATUSES:
                    raise
                # Raises ServerBlockedError unless this is a brief, retryable 429.
                wait = refusal_wait(self.url, exc, attempt)
                pace = self._limiter.slow_down() if hasattr(self._limiter, "slow_down") else None
                self.n_refused += 1
                print(f"[acmaddl] HTTP 429 from {self.url}: waiting {wait:.0f}s"
                      + (f", then continuing at one request per {pace:.0f}s" if pace else ""))
                time.sleep(wait)
                attempt += 1
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                raise ServerUnreachableError(f"no answer from {self.url}: {exc}") from exc
        self.n_bytes += len(body)

        if status == 200:
            # Server ignored the Range header and sent the whole file.
            self._size = len(body)
            self._chunks = {i: body[o:o + self._chunk]
                            for i, o in enumerate(range(0, len(body), self._chunk))}
            return
        match = re.match(r"bytes (\d+)-(\d+)/(\d+)", content_range)
        if not match or int(match.group(1)) != lo:
            raise OSError(f"unexpected Content-Range {content_range!r} from {self.url} "
                          f"for bytes={lo}-{hi}")
        self._size = int(match.group(3))
        for i, o in enumerate(range(0, len(body), self._chunk)):
            self._chunks[first_chunk + i] = body[o:o + self._chunk]


class PacedRangeFile(io.RawIOBase):
    """Seekable read-only file view over a :class:`RangeStore`."""

    def __init__(self, store):
        super().__init__()
        self._store = store
        self.name = store.url
        self._pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return False

    def tell(self):
        return self._pos

    @property
    def size(self):
        try:
            return self._store.size
        except Exception:
            return 0

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        return self._pos

    def read(self, n=-1):
        # Called from inside GDAL: never raise. A failed request is recorded on
        # the store (``store.error``) and reported to GDAL as a short read.
        try:
            if n is None or n < 0:
                n = max(self._store.size - self._pos, 0)
            data = self._store.read(self._pos, n)
        except Exception:
            return b""
        self._pos += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)
