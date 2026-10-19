"""Sample-based packet-loss intervals shared by recording import and export."""

from __future__ import annotations

import operator
from typing import Iterable, Optional


# Reserved for app provenance; normal event annotations remain separate.
ANNOTATION_DESCRIPTION = "BAD_ganglion_packet_loss"


def loss_spans(samples: Iterable[int], lengths: Optional[Iterable[int]] = None,
               n_samples: Optional[int] = None) -> list[tuple[int, int]]:
    """Return sorted, unique (start, length) gaps, clipped to the recording.

    Legacy packet-loss logs store only the start of a two-sample held packet.
    Explicit lengths preserve a partial packet when recordings are cropped.
    """
    sizes = list(lengths) if lengths is not None else []
    spans = set()
    for i, value in enumerate(samples):
        try:
            start = operator.index(value)
            length = operator.index(sizes[i]) if i < len(sizes) else 2
        except (TypeError, ValueError):
            continue
        if length <= 0:
            continue
        end = start + length
        start = max(0, start)
        if n_samples is not None:
            end = min(n_samples, end)
        if end > start:
            spans.add((start, end - start))
    return sorted(spans)


def crop_loss_spans(samples: Iterable[int], lengths: Optional[Iterable[int]],
                    start: int, end: int) -> list[tuple[int, int]]:
    """Intersect loss intervals with [start, end), shifting into crop samples."""
    out = []
    for sample, length in loss_spans(samples, lengths):
        lo, hi = max(start, sample), min(end, sample + length)
        if lo < hi:
            out.append((lo - start, hi - lo))
    return out
