from abc import ABC, abstractmethod


class AdapterBase(ABC):
    @abstractmethod
    def fetch_data(self, product_config, variable, date_range=None, region=None):
        ...

    def _resolve_streams(self, product_config, date_range):
        """Resolve a request into [(stream, sub_range)] segments.

        Mirrors today's start-year stream selection. When the product sets
        ``append_streams`` and the requested range spans the end of
        ``grid.hindcast_range``, return both a hindcast and a forecast segment so
        the adapter can fetch and concatenate them on time.

        The forecast segment's start is ``grid.forecast_range[0]`` when the
        product declares one, else ``hindcast_range[1] + 1`` (legacy default for
        products that don't set forecast_range). For most products these two
        streams are disjoint (forecast starts the year after the hindcast ends),
        so this is a no-op vs. the old ``hr[1] + 1`` logic. But some datasets
        (e.g. IRI NMME CFSv2) split their two OPeNDAP streams by MONTH, not by
        year -- the hindcast stream runs through ~March of its final year and the
        forecast stream starts ~April of that SAME year. For those, the catalog
        sets ``hindcast_range`` and ``forecast_range`` to OVERLAP at that split
        year (e.g. hindcast_range=[1982, 2011], forecast_range=[2011, null]), so
        the year appears in BOTH segments below; each stream's own init-time (S)
        filter then naturally contributes only the inits it actually has, and the
        adapter unions the two fetches. Backward-compatible: when the ranges do
        NOT overlap (forecast_range[0] > hindcast_range[1], or forecast_range is
        absent), the segments are exactly as before -- no shared year.
        """
        if not product_config.get("split_streams"):
            return [("hindcast", date_range)]
        grid = product_config.get("grid") or {}
        hr = grid.get("hindcast_range")
        fr = grid.get("forecast_range")
        fstart = fr[0] if fr else (hr[1] + 1 if hr else None)
        if (product_config.get("append_streams") and hr and fstart is not None and date_range
                and date_range[0] <= hr[1] and fstart <= date_range[1]):
            segs = []
            if date_range[0] <= hr[1]:
                segs.append(("hindcast", (date_range[0], min(date_range[1], hr[1]))))
            if date_range[1] >= fstart:
                segs.append(("forecast", (max(date_range[0], fstart), date_range[1])))
            return segs
        is_forecast = bool(date_range and hr and date_range[0] > hr[1])
        return [("forecast" if is_forecast else "hindcast", date_range)]

    def health_check(self, product_config, probe_remote=False):
        """Return adapter health metadata.

        Subclasses should override this to provide adapter-specific checks.
        """
        return {
            "healthy": True,
            "kind": "config",
            "message": "No adapter-specific health check implemented.",
            "probe_remote": bool(probe_remote),
        }
