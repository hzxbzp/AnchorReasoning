"""LLM judges for the two free-text fields: ``implication`` (per object) and ``reason``
(per frame).

Three deliberate properties of these judges:

1. **Relaxed rubric.**  A prediction matches when it refers to the SAME object / condition and the
   SAME direction of constraint.  Paraphrase, a coarser or finer wording, and dropped non-essential
   modifiers are all accepted; only a wrong referent or an opposite constraint scores 0.
2. **Axes reported separately.**  ``cause`` and ``effect`` are the headline numbers; the ``overall``
   conjunction is still emitted, but it should not be read on its own — the conjunction hides the
   difference between "right object, wrong effect" and "wrong object".
3. **API errors drop the item** (all three fields ``None``) instead of scoring 0, so a network
   hiccup is not charged to the model.  An empty prediction still scores 0 — that is a real failure.
"""
from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence

from training.core.paths import API_KEY_FILE

__all__ = ["make_implication_judge", "make_reason_judge", "evaluate_reason", "load_key",
           "JUDGE_MODEL", "IMPL_SYS", "REASON_SYS"]

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.5-2026-04-23")

_RELAX = (
    "Be LENIENT about wording: accept paraphrase, a coarser or a more detailed description, a "
    "different sentence order, and omitted non-essential modifiers. Score 0 ONLY when the "
    "prediction refers to a DIFFERENT object/condition than the reference, or asserts the "
    "OPPOSITE direction of constraint (e.g. 'releases the path' vs 'blocks the path'). "
    "Judge the two axes independently -- a wrong CAUSE does not force EFFECT to 0, and vice versa."
)

IMPL_SYS = (
    "You evaluate an autonomous-driving model's explanation of ONE key object. Every REFERENCE "
    "explanation has two parts: (A) the CAUSE = what the object is and its driving-relevant "
    "state/position; (B) the EFFECT = how it constrains or releases the ego vehicle's motion. "
    "Compare the MODEL explanation to the REFERENCE on BOTH axes. " + _RELAX +
    " Reply with EXACTLY two digits separated by a comma: cause,effect (e.g. '1,0'). No other text."
)

REASON_SYS = (
    "You evaluate an autonomous-driving model's one-sentence REASON for its driving plan. The "
    "REFERENCE reason names (A) the DOMINANT factor that shapes the ego vehicle's motion -- the "
    "object or condition that blocks, narrows, opens, guides or yields its path -- and (B) the "
    "EFFECT that factor has on the ego's motion. Score the MODEL reason on both axes: cause = it "
    "identifies the same dominant factor (pointing at the same object or the same kind of "
    "condition is enough; it need not match the level of detail); effect = the direction of the "
    "influence agrees. " + _RELAX +
    " Reply with EXACTLY two digits separated by a comma: cause,effect (e.g. '1,0'). No other text."
)

_DROPPED: Dict[str, Any] = {"cause": None, "effect": None, "overall": None, "error": True}
_ZERO: Dict[str, Any] = {"cause": 0, "effect": 0, "overall": 0}


def load_key(path: Optional[str] = None) -> str:
    """OPENAI_API_KEY, else ``path`` / OPENAI_API_KEY_FILE, else ``paths.API_KEY_FILE``
    (``<repo>/api_key.txt``, git-ignored).  The key is never printed or logged."""
    k = os.environ.get("OPENAI_API_KEY")
    if k:
        return k.strip()
    path = path or os.environ.get("OPENAI_API_KEY_FILE") or API_KEY_FILE
    if path and os.path.isfile(path):
        txt = open(path).read()
        m = re.search(r"(sk-[A-Za-z0-9_\-]{20,})", txt)
        if m:
            return m.group(1)
        for line in txt.splitlines():
            if "OPENAI_API_KEY" in line and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("OpenAI API key not found (set OPENAI_API_KEY / OPENAI_API_KEY_FILE / --key-file)")


def _parse(txt: str) -> Dict[str, Any]:
    digits = re.findall(r"[01]", txt or "")
    if len(digits) < 2:
        return dict(_DROPPED)                      # unparseable reply -> drop, do not charge the model
    c, e = int(digits[0]), int(digits[1])
    return {"cause": c, "effect": e, "overall": int(c and e)}


_STATE = {"no_credit_hits": 0, "exhausted": False}     # process-wide circuit breaker for exhausted credits
JUDGE_MAX_DROP = 0.20                                    # > 20 % judge items dropped -> headline numbers withheld


def _make_judge(system: str, user_of: Callable[[Dict[str, Any]], str], model: str,
                key_file: Optional[str], max_workers: int, client: Any = None):
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=load_key(key_file))

    def one(it: Dict[str, Any]) -> Dict[str, Any]:
        if not (it.get("reference") and it.get("prediction")):
            return dict(_ZERO)                     # missing prediction IS a failure
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_of(it)}]

        def content(**kw: Any) -> str:
            # Rate-limit 429s are transient: back off and retry instead of dropping the item (a dropped item
            # changes the denominator, a retry changes nothing).  "No credits remaining" is NOT transient:
            # after a few of those the whole run stops calling the API (every item is dropped at once,
            # the columns become '—' and the run is re-judged from its chains once credits are added).
            for attempt in range(5):
                if _STATE["exhausted"]:
                    raise RuntimeError("judge API credits exhausted (circuit open)")
                try:
                    r = client.chat.completions.create(model=model, messages=msgs, **kw)
                    return r.choices[0].message.content or ""
                except Exception as exc:
                    code = getattr(exc, "status_code", None)
                    msg = str(exc)
                    if "no credits" in msg.lower() or "insufficient_quota" in msg.lower():
                        _STATE["no_credit_hits"] += 1
                        if _STATE["no_credit_hits"] >= 3 and not _STATE["exhausted"]:
                            _STATE["exhausted"] = True
                            print("[judge] API credits exhausted -> no more judge calls this run (re-judge later "
                                  "with evaluation/rejudge.py or rescore --judge)", flush=True)
                        raise
                    if (code == 429 or "RateLimit" in type(exc).__name__) and attempt < 4:
                        time.sleep(2.0 * (2 ** attempt))
                        continue
                    raise
            raise RuntimeError("unreachable")
        try:
            try:
                txt = content(temperature=0, max_tokens=8)          # classic chat models
            except Exception:
                try:
                    txt = content(max_completion_tokens=64, reasoning_effort="none")
                except Exception:
                    txt = content(max_completion_tokens=2048)
        except Exception as exc:                                    # network / quota / refusal
            if not _STATE["exhausted"] or _STATE["no_credit_hits"] <= 3:
                print(f"[judge] dropped one item: {type(exc).__name__}: {str(exc)[:80]}", flush=True)
            return dict(_DROPPED)
        return _parse(txt)

    def judge(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(one, items))
    return judge


def make_implication_judge(model: str = JUDGE_MODEL, key_file: Optional[str] = None,
                           max_workers: int = 8, client: Any = None):
    """judge(items) -> [{cause, effect, overall}]; items: {obj_desc, reference, prediction}."""
    return _make_judge(
        IMPL_SYS,
        lambda it: (f"OBJECT: {it.get('obj_desc', '')}\nREFERENCE: {it['reference']}\n"
                    f"MODEL: {it['prediction']}\nScore cause,effect as two digits."),
        model, key_file, max_workers, client)


def make_reason_judge(model: str = JUDGE_MODEL, key_file: Optional[str] = None,
                      max_workers: int = 8, client: Any = None):
    """judge(items) -> [{cause, effect, overall}]; items: {objects, reference, prediction}.

    ``objects`` is the frame's GT object list as context only -- it is never scored against."""
    return _make_judge(
        REASON_SYS,
        lambda it: (f"OBJECTS IN THE SCENE (context only): {it.get('objects', 'n/a')}\n"
                    f"REFERENCE: {it['reference']}\nMODEL: {it['prediction']}\n"
                    f"Score cause,effect as two digits."),
        model, key_file, max_workers, client)


def reason_items(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build the judge items for ``reason`` from the eval records (frames whose GT has a reason)."""
    items = []
    for rec in records:
        pano = rec.get("pano") or {}
        ref = (pano.get("reason") or "").strip()
        if not ref:
            continue                                    # p1-p6 frames carry no reason label
        objs = ", ".join(
            f"{(a.get('attributes') or {}).get('type', '?')}"
            f"{' ' + ((a.get('attributes') or {}).get('location') or '') if (a.get('attributes') or {}).get('location') else ''}"
            for a in (pano.get("annotations") or []))
        items.append({"objects": objs or "none", "reference": ref,
                      "prediction": ((rec.get("pred") or {}).get("reason") or "").strip()})
    return items


def evaluate_reason(records: Sequence[Dict[str, Any]], judge: Any = None) -> Dict[str, Any]:
    """``reason`` metrics.  ``cause`` / ``effect`` are the headline numbers; ``overall`` is their
    conjunction.  Without a judge only the missing rate is returned (no token-overlap proxy:
    a 1-2 sentence reason is too short for one to mean anything)."""
    items = reason_items(records)
    n_gt = len(items)
    n_missing = sum(1 for it in items if not it["prediction"])
    out: Dict[str, Any] = {
        "reason_n_with_gt": n_gt,
        "reason_missing_rate": (n_missing / n_gt) if n_gt else None,
        "reason_cause": None, "reason_effect": None, "reason_overall": None,
        "reason_n_scored": 0, "reason_n_dropped": 0,
    }
    if judge is None or not items:
        return out
    scored = judge(items)
    keep = [s for s in scored if s.get("cause") is not None]
    out["reason_n_scored"] = len(keep)
    out["reason_n_dropped"] = len(scored) - len(keep)
    # Dropped items (API failures) leave the denominator; when MORE than JUDGE_MAX_DROP of the items were
    # dropped the remaining sample is biased (typically only the 'missing prediction = 0' items survive
    # an outage), so the headline numbers are withheld (None -> '—') until the run is re-judged.
    if keep and out["reason_n_dropped"] <= JUDGE_MAX_DROP * len(scored):
        out["reason_cause"] = sum(int(s["cause"]) for s in keep) / len(keep)
        out["reason_effect"] = sum(int(s["effect"]) for s in keep) / len(keep)
        out["reason_overall"] = sum(int(s["overall"]) for s in keep) / len(keep)
    elif keep:
        out["reason_judge_incomplete"] = True
    return out


if __name__ == "__main__":
    # ---- offline self-test: prompts, parsing, drop semantics, aggregation (no API call) ----
    class _FakeClient:
        """Returns the reply queued for the n-th call; an Exception instance is raised."""

        def __init__(self, replies):
            self.replies, self.seen = list(replies), []
            outer = self

            class _C:
                class completions:
                    @staticmethod
                    def create(model, messages, **kw):
                        outer.seen.append(messages[-1]["content"])
                        r = outer.replies[len(outer.seen) - 1]
                        if isinstance(r, Exception):
                            raise r

                        class _M:
                            content = r

                        class _Ch:
                            message = _M()
                        return type("R", (), {"choices": [_Ch()]})()
            self.chat = _C()

    assert _parse("1,0") == {"cause": 1, "effect": 0, "overall": 0}
    assert _parse("1,1") == {"cause": 1, "effect": 1, "overall": 1}
    assert _parse("sorry")["cause"] is None                     # unparseable -> dropped
    j = make_implication_judge(client=_FakeClient(["1,1", "0,1", RuntimeError("429")]))
    out = j([{"obj_desc": "Car front", "reference": "r1", "prediction": "p1"},
             {"obj_desc": "Traffic light", "reference": "r2", "prediction": "p2"},
             {"obj_desc": "Sign", "reference": "r3", "prediction": "p3"}])
    assert [o["cause"] for o in out] == [1, 0, None], out
    assert out[2]["effect"] is None and out[2].get("error"), out[2]
    assert make_implication_judge(client=_FakeClient([]))([{"reference": "r", "prediction": ""}]) == [_ZERO]

    recs = [{"pano": {"reason": "The lead vehicle blocks the lane.",
                      "annotations": [{"attributes": {"type": "Car", "location": "front in the same lane"}}]},
             "pred": {"reason": "the car ahead is in the way"}},
            {"pano": {"reason": "Red light.", "annotations": []}, "pred": {"reason": ""}},
            {"pano": {"annotations": []}, "pred": {"reason": "ignored, no GT reason"}}]
    items = reason_items(recs)
    assert len(items) == 2 and "Car front in the same lane" in items[0]["objects"], items
    # frame 2 has a GT reason but an EMPTY prediction -> scored 0,0 without an API call, so the
    # fake client only serves frame 1.  Both items count: cause = (1 + 0) / 2.
    m = evaluate_reason(recs, make_reason_judge(client=_FakeClient(["1,1"])))
    assert m["reason_n_with_gt"] == 2 and m["reason_missing_rate"] == 0.5
    assert m["reason_n_scored"] == 2 and m["reason_n_dropped"] == 0, m
    assert m["reason_cause"] == 0.5 and m["reason_effect"] == 0.5 and m["reason_overall"] == 0.5, m
    # an API error on the only scorable item -> dropped, not charged
    m1 = evaluate_reason(recs[:1], make_reason_judge(client=_FakeClient([RuntimeError("429")])))
    assert m1["reason_n_scored"] == 0 and m1["reason_n_dropped"] == 1 and m1["reason_cause"] is None, m1
    m0 = evaluate_reason(recs)
    assert m0["reason_cause"] is None and m0["reason_missing_rate"] == 0.5
    print("judges.py self-test OK  (relaxed rubrics, cause/effect split, drop-on-error, aggregation)")
