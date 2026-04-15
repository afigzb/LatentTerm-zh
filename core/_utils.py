import re
import math
from collections import Counter
from dataclasses import dataclass, field

_LEFT_NOISE = frozenset(
    '的了着过地得也都就又还更不没于在从对把被让们些所其有这那和与或是'
    '向以为而'
)
_RIGHT_NOISE = frozenset(
    '的了着过地得也都就却又还更再不没来出'
    '这那已是走被则推有自之终看'
    '以为而及去里边处些个时后吗呢啊吧呀'
    '在和'
)
_PHRASE_MARKERS = frozenset('的和与或是')

_RE_CHINESE = re.compile(r'[\u4e00-\u9fff]+')
_RE_PURE_CHINESE = re.compile(r'^[\u4e00-\u9fff]+$')


@dataclass
class StrategyResult:
    """四策略的统一返回类型。
    scores  — 候选词 → 原始分数
    meta    — 策略附带的额外信息（如上下文模板命中详情）
    """
    scores: dict[str, float]
    meta: dict = field(default_factory=dict)


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values())


def _normalize(scores: dict) -> dict:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _clean_boundary(word: str) -> bool:
    if word[0] in _LEFT_NOISE or word[-1] in _RIGHT_NOISE:
        return False
    for i in range(1, len(word) - 1):
        if word[i] in _PHRASE_MARKERS:
            return False
    return True
