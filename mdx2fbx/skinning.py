"""Warcraft 3 matrix groups -> per-vertex bone influences."""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

from mdx_parser import Geoset

log = logging.getLogger("mdx2fbx.skinning")

Influence = Tuple[int, float]  # (bone_object_id, weight)


def matrix_group_offsets(sizes: Sequence[int]) -> List[int]:
    offs = [0]
    acc = 0
    for s in sizes:
        acc += s
        offs.append(acc)
    return offs


def geoset_influences(
    geoset: Geoset,
    bone_index_to_id: Sequence[int],
    max_influences: int = 4,
) -> Tuple[List[List[Influence]], int]:
    """Return per-vertex influence lists and the number of clamped vertices."""
    offsets = matrix_group_offsets(geoset.matrix_group_sizes)
    n_groups = len(geoset.matrix_group_sizes)
    n_mats = len(geoset.matrix_indices)
    clamped = 0
    out: List[List[Influence]] = []

    for vi, g in enumerate(geoset.vertex_groups):
        if g < 0 or g >= n_groups:
            raise ValueError(
                "GEOS[%d] vertex %d references matrix group %d (have %d)"
                % (geoset.index, vi, g, n_groups)
            )
        count = geoset.matrix_group_sizes[g]
        start = offsets[g]
        end = start + count
        if end > n_mats:
            raise ValueError(
                "GEOS[%d] matrix group %d overruns MATS" % (geoset.index, g)
            )
        raw = geoset.matrix_indices[start:end]
        if count <= 0:
            out.append([])
            continue
        if count > max_influences:
            raw = raw[:max_influences]
            count = max_influences
            clamped += 1
        weight = 1.0 / float(count)
        inf: List[Influence] = []
        for bone_index in raw:
            if bone_index < 0 or bone_index >= len(bone_index_to_id):
                raise ValueError(
                    "GEOS[%d] vertex %d references bone index %d (have %d bones)"
                    % (geoset.index, vi, bone_index, len(bone_index_to_id))
                )
            inf.append((bone_index_to_id[bone_index], weight))
        out.append(inf)

    if clamped:
        log.warning(
            "GEOS[%d]: clamped %d vertices from >%d influences (WC3 matrix groups)",
            geoset.index,
            clamped,
            max_influences,
        )
    return out, clamped


def max_raw_influences(geoset: Geoset) -> int:
    return max(geoset.matrix_group_sizes) if geoset.matrix_group_sizes else 0
