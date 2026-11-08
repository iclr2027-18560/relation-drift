"""P-C: KRAS pair-local leftover vs a pre-registered cryptic-pocket interface.

GDP-bound G12C (pocket not occupied by a Switch-II covalent inhibitor) vs
AMG 510 / sotorasib (6OIM). Leftover is each contact vs its own past in the
sender's local frame. Not a typed shared R_r. Not 'hinge must be highest'.
"""

from __future__ import annotations

from typing import Set

# Pre-registered from Canon et al. Nature 2019 / AMG 510 literature
# (Sci Rep 2020: Switch I 25–40, Switch II 58–72; cryptic subpocket H95/Y96/Q99).
SWITCH_I = range(25, 41)
SWITCH_II = range(58, 73)
CRYPTIC_SUBPOCKET: Set[int] = {95, 96, 99}
# SII-P forming residues used in AMG 510 binding descriptions
POCKET: Set[int] = {12, 58, 59, 60, 61, 64, 68, 69, 72, 92, 95, 96, 99}
P_LOOP = range(10, 18)


def kras_region(resseq: int) -> str:
    if resseq in POCKET:
        return "pocket"
    if resseq in SWITCH_I:
        return "switch_i"
    if resseq in SWITCH_II:
        return "switch_ii"
    if resseq in P_LOOP:
        return "p_loop"
    return "core"


def is_si_sii_pair(ri: int, rj: int) -> bool:
    a, b = sorted((ri, rj))
    return a in SWITCH_I and b in SWITCH_II


def is_pocket_pair(ri: int, rj: int) -> bool:
    return ri in POCKET or rj in POCKET


def is_core_pair(ri: int, rj: int) -> bool:
    return kras_region(ri) == "core" and kras_region(rj) == "core"
