"""ICEWS / CAMEO sanctions-date experiment (observational, no planted ρ).

Triple selectivity: time (t*), type (affected vs unaffected), dyad
(sanctioned vs west/control). Metrics:

  1. Type-resolved structural break  Δ_r(t) = ||ρ̂_r(t) - ρ̂_r(t-1)||_F
  2. Frozen-pre coboundary AUC on sanctioned vs control dyads
  3. Pre/post cosine drift ratio (affected / unaffected types)

ICEWS Coded Event Data is preferred (drop yearly TSVs in data/icews/).
Harvard Dataverse is often WAF-blocked; --source gdelt downloads a
CAMEO-coded GDELT 1.0 sample with the same event taxonomy.

Default episode: Russia / Crimea, t* = 2014-03.

Maps are polar(S_r) of the typed pair cross-moment on frozen pre-t*
TransE embeddings, estimated separately on each dyad stratum so a
RUS–West shift is not averaged into USA–GBR traffic.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import zipfile
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
from urllib.request import urlopen

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
_SRC = _ROOT / "src"
_FIX = _SRC / "run"
for _p in (_SRC, _FIX, _EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_evolvegcn_baseline import (  # noqa: E402
    DEVICE,
    EvolveGCNModel,
)
from run_procrustes_baseline import orthogonal_procrustes, polar_so  # noqa: E402

# ---------------------------------------------------------------------------
# Episode and CAMEO mapping
# ---------------------------------------------------------------------------

RELATION_TYPES: List[str] = [
    "cooperate",
    "trade",
    "threaten",
    "sanction",
    "protest",
]
AFFECTED_TYPES: Set[str] = {"cooperate", "trade"}
UNAFFECTED_TYPES: Set[str] = {"threaten", "protest"}
STRATA: List[str] = ["sanctioned", "west", "control"]

# CAMEO root codes (2-digit).
CAMEO_ROOT_TO_TYPE: Dict[str, str] = {
    "03": "cooperate",
    "04": "cooperate",
    "05": "cooperate",
    "07": "cooperate",
    "06": "trade",
    "13": "threaten",
    "16": "sanction",
    "14": "protest",
}

SANCTIONERS: Set[str] = {
    "USA", "GBR", "FRA", "DEU", "ITA", "ESP", "POL", "NLD", "BEL",
    "CAN", "AUS", "JPN", "UKR",
}
TARGETS: Set[str] = {"RUS"}
CONTROL_COUNTRIES: Set[str] = {
    "BRA", "IND", "ZAF", "ARG", "CHL", "MEX", "IDN", "THA", "NGA", "EGY",
}

COUNTRY_ALIASES: Dict[str, str] = {
    "US": "USA", "UNITED STATES": "USA",
    "UK": "GBR", "UNITED KINGDOM": "GBR",
    "FRANCE": "FRA", "GERMANY": "DEU", "GER": "DEU",
    "ITALY": "ITA", "SPAIN": "ESP", "POLAND": "POL",
    "NETHERLANDS": "NLD", "BELGIUM": "BEL",
    "CANADA": "CAN", "AUSTRALIA": "AUS", "JAPAN": "JPN",
    "UKRAINE": "UKR", "RUSSIA": "RUS",
    "BRAZIL": "BRA", "INDIA": "IND",
    "SOUTH AFRICA": "ZAF", "ARGENTINA": "ARG", "CHILE": "CHL",
    "MEXICO": "MEX", "INDONESIA": "IDN", "THAILAND": "THA",
    "NIGERIA": "NGA", "EGYPT": "EGY",
}
for _code in SANCTIONERS | TARGETS | CONTROL_COUNTRIES:
    COUNTRY_ALIASES.setdefault(_code, _code)

WATCH_COUNTRIES: Set[str] = set(SANCTIONERS | TARGETS | CONTROL_COUNTRIES)

DyadFn = Callable[[str, str], bool]


def _canon(name: str) -> str:
    key = (name or "").strip().upper()
    return COUNTRY_ALIASES.get(key, key)


def _is_sanctioned_dyad(a: str, b: str) -> bool:
    pair = {_canon(a), _canon(b)}
    return bool(pair & TARGETS) and bool(pair & SANCTIONERS)


def _is_west_dyad(a: str, b: str) -> bool:
    return _canon(a) in SANCTIONERS and _canon(b) in SANCTIONERS


def _is_control_dyad(a: str, b: str) -> bool:
    return _canon(a) in CONTROL_COUNTRIES and _canon(b) in CONTROL_COUNTRIES


STRATUM_FN: Dict[str, DyadFn] = {
    "sanctioned": _is_sanctioned_dyad,
    "west": _is_west_dyad,
    "control": _is_control_dyad,
}


def cameo_to_type(code: str) -> Optional[str]:
    digits = "".join(ch for ch in str(code) if ch.isdigit())
    if len(digits) >= 2:
        return CAMEO_ROOT_TO_TYPE.get(digits[:2])
    return None


def relation_name_to_type(name: str) -> Optional[str]:
    n = (name or "").lower()
    if any(k in n for k in ("sanction", "embargo", "boycott")):
        return "sanction"
    if "protest" in n or "demonstrate" in n:
        return "protest"
    if "threaten" in n or "threat" in n:
        return "threaten"
    if any(k in n for k in ("trade", "economic cooperation", "material cooperation")):
        return "trade"
    if any(k in n for k in ("cooperat", "consult", "aid", "diplomatic", "engage in")):
        return "cooperate"
    return cameo_to_type(name)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

Event = Dict[str, Any]


def _parse_date(raw: str) -> Optional[date]:
    s = str(raw).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s[:10], fmt).date()
        except ValueError:
            pass
    if s.isdigit() and len(s) == 8:
        try:
            return datetime.strptime(s, "%Y%m%d").date()
        except ValueError:
            return None
    return None


def _norm_actor(name: str, country: str) -> str:
    actor = (name or "").strip().upper()
    if not actor or actor in {"-", "NULL"}:
        actor = _canon(country)
    return actor


def load_icews_dir(data_dir: Path) -> List[Event]:
    """Load ICEWS Coded Event Data TSV/CSV or TKG triples from data_dir."""
    events: List[Event] = []
    files = sorted(
        list(data_dir.glob("*.tab"))
        + list(data_dir.glob("*.tsv"))
        + list(data_dir.glob("*.csv"))
        + list(data_dir.glob("*.txt"))
    )
    for path in files:
        if path.name.startswith("gdelt"):
            events.extend(load_gdelt_csv(path))
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            if "\t" in sample:
                dialect_delim = "\t"
            else:
                dialect_delim = ","
            first = sample.splitlines()[0] if sample else ""
            has_header = any(
                tok.lower() in first.lower()
                for tok in ("event date", "cameo", "source country", "sqldate")
            )
            if has_header:
                reader = csv.DictReader(handle, delimiter=dialect_delim)
                for row in reader:
                    ev = _row_icews_dict(row)
                    if ev is not None:
                        events.append(ev)
            else:
                for line in handle:
                    parts = line.rstrip("\n").split(dialect_delim)
                    ev = _row_tkg(parts)
                    if ev is not None:
                        events.append(ev)
    return events


def _row_icews_dict(row: Dict[str, str]) -> Optional[Event]:
    keys = {k.lower().strip(): k for k in row}

    def pick(*names: str) -> str:
        for name in names:
            key = keys.get(name.lower())
            if key is not None:
                return (row.get(key) or "").strip()
        return ""

    src_cc = _canon(pick("source country", "source_country", "actor1countrycode"))
    dst_cc = _canon(pick("target country", "target_country", "actor2countrycode"))
    src = _norm_actor(pick("source", "source_name", "actor1name", "actor1"), src_cc)
    dst = _norm_actor(pick("target", "target_name", "actor2name", "actor2"), dst_cc)
    code = pick("cameo code", "cameo_code", "event code", "eventcode")
    rel = relation_name_to_type(code) or cameo_to_type(code)
    when = _parse_date(pick("event date", "event_date", "sqldate", "date"))
    if rel is None or when is None or not src or not dst or src == dst:
        return None
    return {
        "date": when, "src": src, "dst": dst,
        "src_cc": src_cc, "dst_cc": dst_cc, "rel": rel,
    }


def _row_tkg(parts: Sequence[str]) -> Optional[Event]:
    if len(parts) < 4:
        return None
    if _parse_date(parts[0]) is not None or (parts[0].isdigit() and len(parts[0]) >= 8):
        when = _parse_date(parts[0])
        src, rel_raw, dst = parts[1], parts[2], parts[3]
    else:
        src, rel_raw, dst = parts[0], parts[1], parts[2]
        when = _parse_date(parts[3])
    if when is None:
        return None
    rel = relation_name_to_type(rel_raw)
    src_c, dst_c = _canon(src), _canon(dst)
    if rel is None or not src_c or not dst_c or src_c == dst_c:
        return None
    return {
        "date": when, "src": src_c, "dst": dst_c,
        "src_cc": src_c, "dst_cc": dst_c, "rel": rel,
    }


def load_gdelt_csv(path: Path) -> List[Event]:
    events: List[Event] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            when = _parse_date(row.get("date") or row.get("SQLDATE") or "")
            src_cc = _canon(row.get("src_cc") or row.get("src") or row.get("Actor1CountryCode") or "")
            dst_cc = _canon(row.get("dst_cc") or row.get("dst") or row.get("Actor2CountryCode") or "")
            src = _norm_actor(row.get("src") or row.get("Actor1Name") or "", src_cc)
            dst = _norm_actor(row.get("dst") or row.get("Actor2Name") or "", dst_cc)
            rel = row.get("rel") or cameo_to_type(row.get("EventRootCode") or "")
            if when is None or rel not in RELATION_TYPES:
                continue
            if not src or not dst or src == dst:
                continue
            events.append({
                "date": when, "src": src, "dst": dst,
                "src_cc": src_cc, "dst_cc": dst_cc, "rel": rel,
            })
    return events


def download_gdelt_filtered(
    out_csv: Path,
    start: date,
    end: date,
    days_of_month: Sequence[int],
) -> Path:
    """Download GDELT 1.0 daily zips, keep watch-country CAMEO events."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_out: List[Tuple[str, str, str, str, str, str]] = []
    cur_year, cur_month = start.year, start.month
    while date(cur_year, cur_month, 1) <= end:
        for day in days_of_month:
            try:
                stamp = date(cur_year, cur_month, day)
            except ValueError:
                continue
            if stamp < start or stamp > end:
                continue
            ymd = stamp.strftime("%Y%m%d")
            url = f"http://data.gdeltproject.org/events/{ymd}.export.CSV.zip"
            try:
                with urlopen(url, timeout=60) as resp:
                    blob = resp.read()
            except Exception as exc:  # noqa: BLE001
                print(f"    skip {ymd}: {exc}")
                continue
            try:
                with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                    name = zf.namelist()[0]
                    raw = zf.read(name).decode("utf-8", errors="replace")
            except zipfile.BadZipFile:
                print(f"    skip {ymd}: bad zip")
                continue
            kept = 0
            for line in raw.splitlines():
                parts = line.split("\t")
                if len(parts) < 29:
                    continue
                src_cc = _canon(parts[7])
                dst_cc = _canon(parts[17])
                if src_cc not in WATCH_COUNTRIES or dst_cc not in WATCH_COUNTRIES:
                    continue
                if src_cc == dst_cc:
                    continue
                rel = cameo_to_type(parts[28])
                if rel is None:
                    continue
                src = _norm_actor(parts[6], src_cc)
                dst = _norm_actor(parts[16], dst_cc)
                if not src or not dst or src == dst:
                    continue
                rows_out.append((stamp.isoformat(), src, dst, src_cc, dst_cc, rel))
                kept += 1
            print(f"    {ymd}: kept {kept} watch-country events")
        if cur_month == 12:
            cur_year += 1
            cur_month = 1
        else:
            cur_month += 1

    out_csv.write_text(
        "date,src,dst,src_cc,dst_cc,rel\n"
        + "\n".join(",".join(_csv_escape(x) for x in r) for r in rows_out)
        + ("\n" if rows_out else ""),
        encoding="utf-8",
    )
    print(f"  wrote {len(rows_out)} events -> {out_csv}")
    return out_csv


def _csv_escape(value: str) -> str:
    if any(ch in value for ch in ",\"\n"):
        return '"' + value.replace('"', '""') + '"'
    return value


# ---------------------------------------------------------------------------
# Embeddings and maps
# ---------------------------------------------------------------------------


def month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def filter_events(
    events: List[Event],
    start: date,
    end: date,
) -> List[Event]:
    keep = []
    for ev in events:
        if not (start <= ev["date"] <= end):
            continue
        if ev["rel"] not in RELATION_TYPES:
            continue
        if ev["src_cc"] not in WATCH_COUNTRIES or ev["dst_cc"] not in WATCH_COUNTRIES:
            continue
        keep.append(ev)
    return keep


class TransE(nn.Module):
    def __init__(self, n_ent: int, n_rel: int, dim: int = 32):
        super().__init__()
        self.ent = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)

    def score(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        return -(self.ent(h) + self.rel(r) - self.ent(t)).norm(dim=-1)


def train_transe(
    triples: List[Tuple[int, int, int]],
    n_ent: int,
    n_rel: int,
    dim: int,
    epochs: int,
    seed: int,
) -> Tensor:
    """Return frozen entity embeddings [n_ent, dim]."""
    random.seed(seed)
    torch.manual_seed(seed)
    if not triples:
        return torch.zeros(n_ent, dim)
    model = TransE(n_ent, n_rel, dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    idx = list(range(len(triples)))
    batch = min(512, len(triples))
    for _ in range(epochs):
        random.shuffle(idx)
        for start in range(0, len(idx), batch):
            sl = idx[start : start + batch]
            h = torch.tensor([triples[i][0] for i in sl], dtype=torch.long)
            r = torch.tensor([triples[i][1] for i in sl], dtype=torch.long)
            t = torch.tensor([triples[i][2] for i in sl], dtype=torch.long)
            t_neg = torch.randint(0, n_ent, (len(sl),))
            pos = model.score(h, r, t)
            neg = model.score(h, r, t_neg)
            loss = F.relu(1.0 + neg - pos).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        emb = model.ent.weight.detach().clone()
        emb = F.normalize(emb, dim=-1)
    return emb


def collect_pair_stacks(
    events: List[Event],
    rel: str,
    emb: Tensor,
    ent_index: Dict[str, int],
    dyad_fn: Optional[DyadFn],
    unique: bool,
) -> Optional[Tuple[Tensor, Tensor]]:
    src_rows: List[Tensor] = []
    dst_rows: List[Tensor] = []
    seen: Set[Tuple[str, str]] = set()
    for ev in events:
        if ev["rel"] != rel:
            continue
        if dyad_fn is not None and not dyad_fn(ev["src_cc"], ev["dst_cc"]):
            continue
        if ev["src"] not in ent_index or ev["dst"] not in ent_index:
            continue
        key = (ev["src"], ev["dst"])
        if unique and key in seen:
            continue
        seen.add(key)
        src_rows.append(emb[ent_index[ev["src"]]])
        dst_rows.append(emb[ent_index[ev["dst"]]])
    if len(src_rows) < 2:
        return None
    return torch.stack(src_rows), torch.stack(dst_rows)


def fit_polar_map(
    stacks: Optional[Tuple[Tensor, Tensor]],
    dim: int,
    ridge: float = 1e-3,
) -> Tensor:
    """Paper estimator: polar of the pooled typed cross-moment S = E[x_v x_u^T]."""
    eye = torch.eye(dim)
    if stacks is None:
        return eye
    src, dst = stacks
    cov = (dst.T @ src) / float(src.shape[0])
    return polar_so(cov + ridge * eye)


def fro_delta(a: Tensor, b: Tensor) -> float:
    return float((a - b).square().sum().sqrt().item())


def vec_cos_drift(pre: Tensor, post: Tensor) -> float:
    u = pre.reshape(-1)
    v = post.reshape(-1)
    denom = float(u.norm() * v.norm()) + 1e-8
    return 1.0 - float(torch.dot(u, v) / denom)


# ---------------------------------------------------------------------------
# EvolveGCN global weight path
# ---------------------------------------------------------------------------


def evolvegcn_weight_series(
    months: List[str],
    monthly_events: Dict[str, List[Event]],
    emb: Tensor,
    ent_index: Dict[str, int],
    t_star: str,
    epochs: int,
    seed: int,
) -> List[float]:
    """Train EvolveGCN-H on pre-t* months; unroll W_t from the last pre state.

    Eval does *not* reinitialize: a reinit+unroll GRU with frozen x produces
    a monotone contraction that is not a graph-change statistic.
    """
    torch.manual_seed(seed)
    n_ent = emb.shape[0]
    dim = emb.shape[1]
    snaps: List[Dict[str, Tensor]] = []
    for month in months:
        evs = monthly_events[month]
        pairs = [
            (ent_index[e["src"]], ent_index[e["dst"]])
            for e in evs
            if e["src"] in ent_index and e["dst"] in ent_index
        ]
        if not pairs:
            ei = torch.zeros(2, 0, dtype=torch.long)
        else:
            ei = torch.tensor(pairs, dtype=torch.long).t().contiguous()
            ei = torch.cat([ei, ei.flip(0)], dim=1)
        snaps.append({"x": emb, "struct_edge_index": ei, "edge_index": ei})

    train_end = max(1, months.index(t_star) if t_star in months else len(months) // 2)
    model = EvolveGCNModel(
        num_nodes=n_ent,
        in_channels=dim,
        hidden_dim=min(32, dim),
        num_layers=1,
        variant="H",
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def _neg_edges(n_neg: int) -> Tensor:
        src = torch.randint(0, n_ent, (n_neg,))
        dst = torch.randint(0, n_ent, (n_neg,))
        return torch.stack([src, dst], dim=0)

    model.train()
    for _ in range(epochs):
        model.reinitialize_weights()
        for t in range(train_end):
            x = snaps[t]["x"].to(DEVICE)
            struct = snaps[t]["struct_edge_index"].to(DEVICE)
            pos = snaps[t]["edge_index"]
            if pos.shape[1] == 0:
                continue
            neg = _neg_edges(min(256, max(pos.shape[1], 8)))
            query = torch.cat([pos, neg], dim=1).to(DEVICE)
            y = torch.cat(
                [
                    torch.ones(pos.shape[1], device=DEVICE),
                    torch.zeros(neg.shape[1], device=DEVICE),
                ]
            )
            opt.zero_grad()
            logits = model.forward_timestep(x, struct, query)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            opt.step()
            model.detach_weights()

    model.eval()
    model.reinitialize_weights()
    weights: List[Tensor] = []
    with torch.no_grad():
        for t in range(len(months)):
            x = snaps[t]["x"].to(DEVICE)
            struct = snaps[t]["struct_edge_index"].to(DEVICE)
            dummy = struct if struct.shape[1] else torch.zeros(2, 1, dtype=torch.long, device=DEVICE)
            if dummy.shape[1] == 0:
                dummy = torch.zeros(2, 1, dtype=torch.long, device=DEVICE)
            _ = model.forward_timestep(x, struct if struct.shape[1] else dummy, dummy)
            w = model.layers[0].weight
            if w is None:
                weights.append(torch.zeros(1))
            else:
                weights.append(w.detach().cpu().squeeze(0).clone())
            # Continue the GRU state through post-t* snapshots (no reinit).
    deltas = [0.0]
    for t in range(1, len(weights)):
        if weights[t].numel() != weights[t - 1].numel():
            deltas.append(0.0)
        else:
            deltas.append(fro_delta(weights[t], weights[t - 1]))
    return deltas


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def mw_pvalue(break_vals: List[float], rest_vals: List[float]) -> float:
    if len(break_vals) < 1 or len(rest_vals) < 2:
        return float("nan")
    try:
        stat = mannwhitneyu(break_vals, rest_vals, alternative="greater")
        return float(stat.pvalue)
    except ValueError:
        return float("nan")


def ks_pvalue(break_vals: List[float], rest_vals: List[float]) -> float:
    if len(break_vals) < 1 or len(rest_vals) < 2:
        return float("nan")
    try:
        stat = ks_2samp(break_vals, rest_vals, alternative="less")
        return float(stat.pvalue)
    except ValueError:
        return float("nan")


def rank_pvalue(delta_at: float, rest_vals: List[float]) -> float:
    if not rest_vals:
        return float("nan")
    more = sum(1 for v in rest_vals if v >= delta_at)
    return (1.0 + more) / (1.0 + len(rest_vals))


def break_test(series: List[float], break_window: List[int], break_idx: Optional[int]) -> Dict[str, Any]:
    brk = [series[i] for i in break_window]
    rest = [series[i] for i in range(len(series)) if i not in break_window and i > 0]
    at = series[break_idx] if break_idx is not None else float("nan")
    return {
        "delta_at_tstar": at,
        "mean_rest": float(np.mean(rest)) if rest else float("nan"),
        "z_vs_rest": (
            (at - float(np.mean(rest))) / (float(np.std(rest)) + 1e-8)
            if break_idx is not None and rest else float("nan")
        ),
        "mannwhitney_p": mw_pvalue(brk, rest),
        "ks_p": ks_pvalue(brk, rest),
        "rank_p": rank_pvalue(at, rest) if break_idx is not None else float("nan"),
        "break_mean": float(np.mean(brk)) if brk else float("nan"),
    }


def dyad_auc(
    events: List[Event],
    rho: Tensor,
    emb: Tensor,
    ent_index: Dict[str, int],
    rel: str,
    positive_fn: DyadFn,
    negative_fn: DyadFn,
) -> float:
    scores: List[float] = []
    labels: List[int] = []
    for ev in events:
        if ev["rel"] != rel:
            continue
        if ev["src"] not in ent_index or ev["dst"] not in ent_index:
            continue
        xu = emb[ent_index[ev["src"]]]
        xv = emb[ent_index[ev["dst"]]]
        resid = float((xv - rho @ xu).norm().item())
        if positive_fn(ev["src_cc"], ev["dst_cc"]):
            scores.append(resid)
            labels.append(1)
        elif negative_fn(ev["src_cc"], ev["dst_cc"]):
            scores.append(resid)
            labels.append(0)
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def write_delta_plot(
    path: Path,
    months: List[str],
    delta_sanc: Dict[str, List[float]],
    delta_west_coop: List[float],
    delta_global: List[float],
    delta_egcn: List[float],
    t_star: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def zscore(vals: List[float]) -> List[float]:
        arr = np.array(vals, dtype=float)
        mu = float(np.mean(arr[1:])) if len(arr) > 2 else 0.0
        sd = float(np.std(arr[1:])) if len(arr) > 2 else 1.0
        sd = sd if sd > 1e-8 else 1.0
        return [float((v - mu) / sd) for v in vals]

    xs = list(range(len(months)))
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    colors = {
        "cooperate": "#1d4ed8",
        "trade": "#b45309",
        "threaten": "#6b7280",
        "protest": "#9ca3af",
        "sanction": "#7c3aed",
    }
    for rel, series in delta_sanc.items():
        lw = 2.2 if rel in AFFECTED_TYPES else 1.2
        ax.plot(xs, zscore(series), label=f"{rel} (RUS–West)", color=colors.get(rel, "#111"), linewidth=lw)
    ax.plot(xs, zscore(delta_west_coop), label="cooperate (West–West)", color="#0f766e", linestyle="-.", linewidth=1.3)
    ax.plot(xs, zscore(delta_global), label="mixed Procrustes", color="#a16207", linestyle=":", linewidth=1.3)
    ax.plot(xs, zscore(delta_egcn), label="EvolveGCN ||ΔW||", color="#111827", linestyle="--", linewidth=1.4)
    if t_star in months:
        ax.axvline(months.index(t_star), color="#b91c1c", linestyle=":", linewidth=1.2, label=f"t*={t_star}")
    ax.set_xticks(xs[::3])
    ax.set_xticklabels([months[i] for i in xs[::3]], rotation=45, ha="right")
    ax.set_xlabel("Month")
    ax.set_ylabel(r"z-scored $\Delta_r(t)=\|A_r(t)-A_r(t-1)\|_F$")
    ax.set_title("Typed country-adjacency jump (RUS–West) vs West–West and mixed")
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> Dict[str, Any]:
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = _ROOT / data_dir
    cache = data_dir / "gdelt_actors.csv"
    t_star = datetime.strptime(args.t_star, "%Y-%m").date().replace(day=1)
    start = datetime.strptime(args.start, "%Y-%m").date().replace(day=1)
    end = datetime.strptime(args.end, "%Y-%m").date().replace(day=1)
    t_star_key = month_key(t_star)

    events: List[Event] = []
    source_used = "none"
    if args.source in {"icews", "auto"}:
        if data_dir.exists():
            events = load_icews_dir(data_dir)
            if events:
                source_used = "icews"
                print(f"  loaded {len(events)} ICEWS-style events from {data_dir}")
    if not events and args.source in {"gdelt", "auto"}:
        if args.download or not cache.exists():
            print("  downloading GDELT 1.0 CAMEO events (watch countries + actor names)")
            download_gdelt_filtered(
                cache,
                start,
                end,
                days_of_month=tuple(args.gdelt_days),
            )
        events = load_gdelt_csv(cache)
        source_used = "gdelt"
        print(f"  loaded {len(events)} GDELT events from {cache}")
    if not events:
        raise FileNotFoundError(
            "No events loaded. Put ICEWS TSVs in data/icews/ or pass --source gdelt --download."
        )

    events = filter_events(events, start, end)
    print(f"  {len(events)} events in window {start}..{end} source={source_used}")
    if len(events) < 50:
        raise RuntimeError(f"too few events ({len(events)}) to estimate typed maps")

    months = sorted({month_key(ev["date"]) for ev in events})
    monthly: Dict[str, List[Event]] = {m: [] for m in months}
    for ev in events:
        monthly[month_key(ev["date"])].append(ev)

    pre_events = [ev for ev in events if month_key(ev["date"]) < t_star_key]
    counts: Dict[str, int] = defaultdict(int)
    for ev in pre_events:
        counts[ev["src"]] += 1
        counts[ev["dst"]] += 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    if args.max_actors > 0:
        ranked = ranked[: args.max_actors]
    entities = [name for name, n in ranked if n >= args.min_actor_events]
    ent_index = {name: i for i, name in enumerate(entities)}
    rel_index = {name: i for i, name in enumerate(RELATION_TYPES)}
    print(f"  {len(entities)} frozen pre-t* actors, {len(months)} months")

    triples = [
        (ent_index[ev["src"]], rel_index[ev["rel"]], ent_index[ev["dst"]])
        for ev in pre_events
        if ev["src"] in ent_index and ev["dst"] in ent_index
    ]
    print(f"  training TransE on {len(triples)} pre-t* triples")
    emb = train_transe(
        triples,
        n_ent=len(entities),
        n_rel=len(RELATION_TYPES),
        dim=args.dim,
        epochs=args.transe_epochs,
        seed=args.seed,
    )

    maps: Dict[str, Dict[str, Dict[str, Tensor]]] = {
        stratum: {m: {} for m in months} for stratum in STRATA
    }
    pair_n: Dict[str, Dict[str, Dict[str, int]]] = {
        stratum: {m: {} for m in months} for stratum in STRATA
    }
    for stratum, fn in STRATUM_FN.items():
        for month in months:
            for rel in RELATION_TYPES:
                stacks = collect_pair_stacks(
                    monthly[month], rel, emb, ent_index, fn, unique=args.unique
                )
                pair_n[stratum][month][rel] = 0 if stacks is None else int(stacks[0].shape[0])
                maps[stratum][month][rel] = fit_polar_map(stacks, args.dim)

    delta: Dict[str, Dict[str, List[float]]] = {
        stratum: {rel: [0.0] for rel in RELATION_TYPES} for stratum in STRATA
    }
    mixed_delta = [0.0]
    for i in range(1, len(months)):
        prev, cur = months[i - 1], months[i]
        mixed_prev = orthogonal_procrustes(*_stack_all(monthly[prev], emb, ent_index, args.unique))
        mixed_cur = orthogonal_procrustes(*_stack_all(monthly[cur], emb, ent_index, args.unique))
        mixed_delta.append(fro_delta(mixed_cur, mixed_prev))
        for stratum in STRATA:
            for rel in RELATION_TYPES:
                delta[stratum][rel].append(
                    fro_delta(maps[stratum][cur][rel], maps[stratum][prev][rel])
                )

    print("  fitting EvolveGCN global W_t")
    egcn_delta = evolvegcn_weight_series(
        months,
        monthly,
        emb,
        ent_index,
        t_star_key,
        epochs=args.egcn_epochs,
        seed=args.seed,
    )

    break_idx = months.index(t_star_key) if t_star_key in months else None
    break_window: List[int] = []
    if break_idx is not None:
        break_window = [break_idx]
        if break_idx + 1 < len(months):
            break_window.append(break_idx + 1)

    tests: Dict[str, Any] = {}
    for stratum in STRATA:
        tests[stratum] = {
            rel: break_test(delta[stratum][rel], break_window, break_idx)
            for rel in RELATION_TYPES
        }
    tests["mixed_procrustes"] = break_test(mixed_delta, break_window, break_idx)
    tests["evolvegcn"] = break_test(egcn_delta, break_window, break_idx)

    pre_months = [m for m in months if m < t_star_key]
    post_months = [m for m in months if m >= t_star_key]
    drift: Dict[str, Dict[str, float]] = {}
    for stratum in STRATA:
        drift[stratum] = {}
        for rel in RELATION_TYPES:
            pre_m = _mean_maps([maps[stratum][m][rel] for m in pre_months])
            post_m = _mean_maps([maps[stratum][m][rel] for m in post_months])
            drift[stratum][rel] = vec_cos_drift(pre_m, post_m)
    aff_drift = float(np.mean([drift["sanctioned"][r] for r in AFFECTED_TYPES]))
    una_drift = float(np.mean([drift["sanctioned"][r] for r in UNAFFECTED_TYPES]))
    west_coop_drift = drift["west"]["cooperate"]
    drift_ratio = aff_drift / max(una_drift, 1e-8)
    drift_ratio_vs_west = aff_drift / max(west_coop_drift, 1e-8)

    # Frozen pre-map on sanctioned dyads of type r (the constraint that should
    # have held before t*). Scoring post events with this map; fitting on the
    # scored month would collapse residuals.
    pre_sanc_maps = {
        rel: _mean_maps([maps["sanctioned"][m][rel] for m in pre_months])
        for rel in RELATION_TYPES
    }
    post_events = [ev for ev in events if month_key(ev["date"]) >= t_star_key]
    aucs: Dict[str, Any] = {}
    for rel in RELATION_TYPES:
        aucs[rel] = {
            "sanctioned_vs_west": dyad_auc(
                post_events, pre_sanc_maps[rel], emb, ent_index, rel,
                _is_sanctioned_dyad, _is_west_dyad,
            ),
            "sanctioned_vs_control": dyad_auc(
                post_events, pre_sanc_maps[rel], emb, ent_index, rel,
                _is_sanctioned_dyad, _is_control_dyad,
            ),
        }

    monthly_counts = {
        m: {
            "all": {rel: sum(1 for e in monthly[m] if e["rel"] == rel) for rel in RELATION_TYPES},
            "sanctioned_unique": pair_n["sanctioned"][m],
            "west_unique": pair_n["west"][m],
            "control_unique": pair_n["control"][m],
        }
        for m in months
    }

    # Fixed-support country adjacency (one-hot S_r). Polar/TransE on a
    # rank-deficient 16-D slice saturates at random-rotation scale; this is
    # the identifying pair-moment statistic on a frozen country basis.
    country_nodes = sorted(WATCH_COUNTRIES)
    adj_delta, adj_mixed = _country_adjacency_deltas(
        months, monthly, country_nodes, unique=True
    )
    adj_tests: Dict[str, Any] = {}
    for stratum in STRATA:
        adj_tests[stratum] = {
            rel: break_test(adj_delta[stratum][rel], break_window, break_idx)
            for rel in RELATION_TYPES
        }
    adj_tests["mixed_sanctioned"] = break_test(adj_mixed["sanctioned"], break_window, break_idx)
    adj_tests["mixed_west"] = break_test(adj_mixed["west"], break_window, break_idx)

    plot_path = _ROOT / "results" / "icews_sanctions_delta.png"
    write_delta_plot(
        plot_path,
        months,
        adj_delta["sanctioned"],
        adj_delta["west"]["cooperate"],
        adj_mixed["sanctioned"],
        egcn_delta,
        t_star_key,
    )

    adj_sanc = adj_tests["sanctioned"]
    aff_z = float(np.mean([adj_sanc[r]["z_vs_rest"] for r in AFFECTED_TYPES]))
    una_z = float(np.mean([adj_sanc[r]["z_vs_rest"] for r in UNAFFECTED_TYPES]))
    west_z = float(adj_tests["west"]["cooperate"]["z_vs_rest"])
    sanc_z = float(adj_sanc["sanction"]["z_vs_rest"])
    affected_p = [
        adj_sanc[r]["mannwhitney_p"] for r in AFFECTED_TYPES
        if adj_sanc[r]["mannwhitney_p"] == adj_sanc[r]["mannwhitney_p"]
    ]
    payload = {
        "protocol": (
            "Observational CAMEO-coded events. No planted ρ. "
            "TransE embeddings frozen on pre-t* actors only. "
            "Typed maps are polar(S_r) of unique frozen pairs, fit per dyad stratum. "
            "Metric 2 uses the frozen pre-map residual (fitting ρ on the "
            "scored month would collapse residuals). "
            "EvolveGCN W_t is unrolled from the last pre-t* GRU state."
        ),
        "source": source_used,
        "t_star": t_star_key,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "n_events": len(events),
        "n_pre_triples": len(triples),
        "n_actors": len(entities),
        "unique_pairs": bool(args.unique),
        "months": months,
        "delta_typed": delta,
        "delta_mixed_procrustes": mixed_delta,
        "delta_evolvegcn": egcn_delta,
        "delta_country_adjacency": adj_delta,
        "delta_country_adjacency_mixed": adj_mixed,
        "break_tests": tests,
        "break_tests_country_adjacency": adj_tests,
        "drift": drift,
        "drift_ratio_affected_over_unaffected": drift_ratio,
        "drift_ratio_affected_over_west_cooperate": drift_ratio_vs_west,
        "dyad_auc_frozen_pre_map": aucs,
        "monthly_counts": monthly_counts,
        "plot": str(plot_path),
        "success": {
            "affected_break_p_lt_0.05": any(p < 0.05 for p in affected_p),
            "affected_z_gt_unaffected_z": aff_z > una_z,
            "affected_z_gt_west_cooperate_z": aff_z > west_z,
            "drift_ratio_gt_2": drift_ratio > 2.0,
            "adjacency_sanction_z": sanc_z,
            "adjacency_cooperate_z": float(adj_sanc["cooperate"]["z_vs_rest"]),
            "confirmatory_cooperate_trade": bool(
                any(p < 0.05 for p in affected_p) and aff_z > una_z and aff_z > west_z
            ),
        },
    }
    return payload


def _stack_all(
    events: List[Event],
    emb: Tensor,
    ent_index: Dict[str, int],
    unique: bool,
) -> Tuple[Tensor, Tensor]:
    src_rows, dst_rows = [], []
    seen: Set[Tuple[str, str]] = set()
    for ev in events:
        if ev["src"] not in ent_index or ev["dst"] not in ent_index:
            continue
        key = (ev["src"], ev["dst"])
        if unique and key in seen:
            continue
        seen.add(key)
        src_rows.append(emb[ent_index[ev["src"]]])
        dst_rows.append(emb[ent_index[ev["dst"]]])
    if len(src_rows) < 2:
        eye = torch.eye(emb.shape[1])
        return eye, eye
    return torch.stack(src_rows), torch.stack(dst_rows)


def _country_adjacency_deltas(
    months: List[str],
    monthly: Dict[str, List[Event]],
    nodes: List[str],
    unique: bool,
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, List[float]]]:
    idx = {name: i for i, name in enumerate(nodes)}
    n = len(nodes)

    def adj_matrix(events: List[Event], rel: Optional[str], dyad_fn: DyadFn) -> Tensor:
        mat = torch.zeros(n, n)
        seen: Set[Tuple[int, int]] = set()
        for ev in events:
            if rel is not None and ev["rel"] != rel:
                continue
            if ev["rel"] not in RELATION_TYPES:
                continue
            if not dyad_fn(ev["src_cc"], ev["dst_cc"]):
                continue
            i = idx.get(ev["src_cc"])
            j = idx.get(ev["dst_cc"])
            if i is None or j is None:
                continue
            key = (i, j)
            if unique and key in seen:
                continue
            seen.add(key)
            mat[j, i] = 1.0
        return mat

    typed: Dict[str, Dict[str, List[float]]] = {
        stratum: {rel: [0.0] for rel in RELATION_TYPES} for stratum in STRATA
    }
    mixed: Dict[str, List[float]] = {stratum: [0.0] for stratum in STRATA}
    prev_typed = {
        stratum: {
            rel: adj_matrix(monthly[months[0]], rel, fn) for rel in RELATION_TYPES
        }
        for stratum, fn in STRATUM_FN.items()
    }
    prev_mixed = {
        stratum: adj_matrix(monthly[months[0]], None, fn)
        for stratum, fn in STRATUM_FN.items()
    }
    for month in months[1:]:
        for stratum, fn in STRATUM_FN.items():
            mixed_cur = adj_matrix(monthly[month], None, fn)
            mixed[stratum].append(fro_delta(mixed_cur, prev_mixed[stratum]))
            prev_mixed[stratum] = mixed_cur
            for rel in RELATION_TYPES:
                cur = adj_matrix(monthly[month], rel, fn)
                typed[stratum][rel].append(fro_delta(cur, prev_typed[stratum][rel]))
                prev_typed[stratum][rel] = cur
    return typed, mixed


def _mean_maps(maps: List[Tensor]) -> Tensor:
    if not maps:
        raise ValueError("empty map list")
    stacked = torch.stack(maps)
    return stacked.mean(dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["auto", "icews", "gdelt"], default="auto")
    parser.add_argument("--data-dir", type=str, default="data/icews")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--start", type=str, default="2013-03")
    parser.add_argument("--end", type=str, default="2015-03")
    parser.add_argument("--t-star", type=str, default="2014-03")
    parser.add_argument("--gdelt-days", type=int, nargs="*", default=[1, 10, 20])
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--transe-epochs", type=int, default=40)
    parser.add_argument("--egcn-epochs", type=int, default=15)
    parser.add_argument("--min-pairs", type=int, default=6)
    parser.add_argument("--max-actors", type=int, default=400)
    parser.add_argument("--min-actor-events", type=int, default=4)
    parser.add_argument("--unique", action="store_true", default=True)
    parser.add_argument("--weighted", action="store_true", help="Count-weight pairs instead of unique dyads")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results/icews_sanctions.json")
    args = parser.parse_args()
    args.unique = not args.weighted

    payload = run(args)
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {out}")
    print(json.dumps({
        "source": payload["source"],
        "n_events": payload["n_events"],
        "n_actors": payload["n_actors"],
        "break_tests_sanctioned": payload["break_tests"]["sanctioned"],
        "break_tests_west_cooperate": payload["break_tests"]["west"]["cooperate"],
        "break_tests_adjacency_sanc": payload["break_tests_country_adjacency"]["sanctioned"],
        "break_tests_adjacency_mixed_sanc": payload["break_tests_country_adjacency"]["mixed_sanctioned"],
        "break_tests_global": {
            "mixed": payload["break_tests"]["mixed_procrustes"],
            "evolvegcn": payload["break_tests"]["evolvegcn"],
        },
        "drift_ratio": payload["drift_ratio_affected_over_unaffected"],
        "dyad_auc": payload["dyad_auc_frozen_pre_map"],
        "success": payload["success"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
