"""Gate 0: does a native agent→agent style pair law exist on public chats?

The leftover object is a shared linear map x_v ≈ A x_u on an established
link. A2A is a candidate only if that map exists on native traces, not
because a planted map jump can be scored (P-A) and not because a single
agent looks odd.

Pre-registered native object: CAMEL AI Society two-agent chats
(camel-ai/ai_society). Pair law is assistant-turn style ≈ A · user-turn
style. Holdout is by conversation, not by turn.

Template prefixes Instruction:/Solution: are stripped before featurizing.
Those tokens are role labels, not a behavioral map.
"""

from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from robotics_inject import linear_pair_law_grouped

REPO_ID = "camel-ai/ai_society"
ARCHIVE_NAME = "ai_society_chat.tar.gz"

FEATURE_NAMES: Tuple[str, ...] = (
    "log1p_chars",
    "words_per_sentence",
    "digit_rate",
    "punct_rate",
    "type_token",
    "question_rate",
    "list_rate",
    "mean_word_len",
)
CLAIM_CHANNEL = "user_to_assistant"
CONTROL_CHANNEL = "assistant_to_user"
CLAIM_STALKS = ("style", "style_affine")
N_ROLE_PAIRS = 12
MIN_ROLE_PAIR_CONVOS = 8
N_SPLITS = 5
SEED = 42

_PREFIX = re.compile(
    r"^\s*(Instruction|Solution|Understood\.?\s*Let's begin with the first instruction:)\s*:?\s*",
    re.IGNORECASE,
)
_DONE = re.compile(r"<CAMEL_TASK_DONE>", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z0-9']+")
_SENT = re.compile(r"[.!?]+")
_LIST = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+", re.MULTILINE)


def fetch_camel_archive(cache_dir: Path | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    cache_dir = Path(cache_dir) if cache_dir is not None else Path("data/camel")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=ARCHIVE_NAME,
            local_dir=str(cache_dir),
        )
    )


def _strip_template(text: str) -> str:
    text = _DONE.sub(" ", text or "")
    text = _PREFIX.sub("", text, count=1)
    return text.strip()


def style_vector(text: str) -> np.ndarray:
    """8-d style profile. Named before any score is computed."""
    body = _strip_template(text)
    n_chars = max(len(body), 1)
    words = _WORD.findall(body)
    n_words = max(len(words), 1)
    n_sent = max(len(_SENT.findall(body)), 1)
    if body and not _SENT.search(body):
        n_sent = 1
    digits = sum(ch.isdigit() for ch in body)
    punct = sum(ch in ",.;:!?()[]{}\"'`-" for ch in body)
    uniq = len({w.lower() for w in words})
    questions = body.count("?")
    lists = len(_LIST.findall(body))
    mean_w = float(np.mean([len(w) for w in words])) if words else 0.0
    return np.array(
        [
            np.log1p(len(body)),
            n_words / n_sent,
            digits / n_chars,
            punct / n_chars,
            uniq / n_words,
            questions / n_sent,
            lists / max(body.count("\n") + 1, 1),
            mean_w,
        ],
        dtype=np.float64,
    )


def _role_pair(convo_id: str) -> str:
    parts = convo_id.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return convo_id


def _iter_messages(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    n = int(rec.get("num_messages") or 0)
    out: List[Dict[str, Any]] = []
    i = 1
    while True:
        key = f"message_{i}"
        if key not in rec:
            break
        msg = rec[key]
        if isinstance(msg, dict) and msg.get("content"):
            out.append(msg)
        i += 1
        if n and i > n + 2:
            break
    return out


def load_turn_pairs(archive: Path) -> Dict[str, Any]:
    """USER→ASSISTANT and ASSISTANT→USER consecutive turns, grouped by convo."""
    src_ua: List[np.ndarray] = []
    dst_ua: List[np.ndarray] = []
    grp_ua: List[int] = []
    pair_ua: List[str] = []
    src_au: List[np.ndarray] = []
    dst_au: List[np.ndarray] = []
    grp_au: List[int] = []
    pair_au: List[str] = []
    convo_ids: List[str] = []
    convo_index: Dict[str, int] = {}
    n_convos = 0
    n_empty = 0

    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            rec = json.loads(fh.read().decode("utf-8"))
            cid = str(rec.get("id") or member.name.replace(".json", ""))
            msgs = _iter_messages(rec)
            if len(msgs) < 2:
                n_empty += 1
                continue
            if cid not in convo_index:
                convo_index[cid] = len(convo_ids)
                convo_ids.append(cid)
            g = convo_index[cid]
            rp = _role_pair(cid)
            n_convos += 1
            for a, b in zip(msgs, msgs[1:]):
                ra = str(a.get("role_type") or a.get("role") or "").upper()
                rb = str(b.get("role_type") or b.get("role") or "").upper()
                xa = style_vector(str(a.get("content") or ""))
                xb = style_vector(str(b.get("content") or ""))
                if "USER" in ra and "ASSISTANT" in rb:
                    src_ua.append(xa)
                    dst_ua.append(xb)
                    grp_ua.append(g)
                    pair_ua.append(rp)
                elif "ASSISTANT" in ra and "USER" in rb:
                    src_au.append(xa)
                    dst_au.append(xb)
                    grp_au.append(g)
                    pair_au.append(rp)

    return {
        "user_to_assistant": {
            "src": np.stack(src_ua) if src_ua else np.zeros((0, 8)),
            "dst": np.stack(dst_ua) if dst_ua else np.zeros((0, 8)),
            "groups": np.asarray(grp_ua, dtype=np.int32),
            "role_pairs": np.asarray(pair_ua),
        },
        "assistant_to_user": {
            "src": np.stack(src_au) if src_au else np.zeros((0, 8)),
            "dst": np.stack(dst_au) if dst_au else np.zeros((0, 8)),
            "groups": np.asarray(grp_au, dtype=np.int32),
            "role_pairs": np.asarray(pair_au),
        },
        "n_conversations": int(len(convo_ids)),
        "n_skipped_short": int(n_empty),
        "convo_ids": convo_ids,
        "feature_names": list(FEATURE_NAMES),
    }


def _affine(src: np.ndarray) -> np.ndarray:
    return np.concatenate([src, np.ones((src.shape[0], 1), dtype=src.dtype)], axis=1)


def _eval_cell(
    src: np.ndarray,
    dst: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
    *,
    affine: bool,
) -> Dict[str, float]:
    x = _affine(src) if affine else src
    return linear_pair_law_grouped(x, dst, groups, rng, n_splits=N_SPLITS)


def _preregistered_role_pairs(role_pairs: np.ndarray, groups: np.ndarray) -> List[str]:
    """First N role pairs by sorted id that have enough conversations."""
    counts: Dict[str, set] = {}
    for rp, g in zip(role_pairs.tolist(), groups.tolist()):
        counts.setdefault(rp, set()).add(int(g))
    eligible = sorted(
        rp for rp, convos in counts.items() if len(convos) >= MIN_ROLE_PAIR_CONVOS
    )
    return eligible[:N_ROLE_PAIRS]


def evaluate_gate0(pairs: Dict[str, Any], *, seed: int = SEED) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    report: Dict[str, Any] = {}
    for channel in (CLAIM_CHANNEL, CONTROL_CHANNEL):
        block = pairs[channel]
        src, dst, groups = block["src"], block["dst"], block["groups"]
        role_pairs = block["role_pairs"]
        chosen = _preregistered_role_pairs(role_pairs, groups)
        report[channel] = {
            "pooled": {
                "style": _eval_cell(src, dst, groups, rng, affine=False),
                "style_affine": _eval_cell(src, dst, groups, rng, affine=True),
            },
            "role_pairs_tested": chosen,
            "role_pair": {},
        }
        for rp in chosen:
            mask = role_pairs == rp
            report[channel]["role_pair"][rp] = {
                "style": _eval_cell(src[mask], dst[mask], groups[mask], rng, affine=False),
                "style_affine": _eval_cell(
                    src[mask], dst[mask], groups[mask], rng, affine=True
                ),
            }
    return report


def gate0_verdict(report: Dict[str, Any]) -> Dict[str, Any]:
    """Pass only on the user→assistant channel.

    Pooled pass = shared map across chats.
    Typed pass = at least half of the 12 pre-registered role pairs pass
    on a claim stalk. Reverse channel cannot authorize.
    """
    claim = report[CLAIM_CHANNEL]
    pooled_pass = any(
        bool(claim["pooled"][stalk].get("usable_linear_pair_law"))
        for stalk in CLAIM_STALKS
    )
    typed_hits = 0
    typed_n = 0
    for rp, cells in claim["role_pair"].items():
        typed_n += 1
        if any(bool(cells[stalk].get("usable_linear_pair_law")) for stalk in CLAIM_STALKS):
            typed_hits += 1
    typed_pass = typed_n > 0 and typed_hits >= max(1, (typed_n + 1) // 2)
    reverse_pass = any(
        bool(report[CONTROL_CHANNEL]["pooled"][stalk].get("usable_linear_pair_law"))
        for stalk in CLAIM_STALKS
    )
    passed = bool(pooled_pass or typed_pass)
    return {
        "passed": passed,
        "pooled_pass": bool(pooled_pass),
        "typed_pass": bool(typed_pass),
        "typed_hits": typed_hits,
        "typed_n": typed_n,
        "reverse_pooled_pass": bool(reverse_pass),
        "open_plant": passed,
        "reason": (
            "native user→assistant style map exists; plant of M-only jump is allowed"
            if passed
            else "no native linear pair law; do not plant, do not claim A2A leftover"
        ),
    }
