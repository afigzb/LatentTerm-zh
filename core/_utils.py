import re
import math
from collections import Counter
from dataclasses import dataclass, field

_LEFT_NOISE = frozenset(
    '的了着过地得也都就又还更没于在从对把被让们些所其有这那和与或是'
    '向以为而'
)
_RIGHT_NOISE = frozenset(
    '的了着过地得也都就却又还更再不没来出'
    '这那已是走被则推有自之终看'
    '以为而及去里边处些个时后吗呢啊吧呀'
    '在和'
)
_PHRASE_MARKERS = frozenset('的和与或是把被让将给')

# 小说常见"名字+对白动词"尾字。用于 vocab 构建时的额外审查，
# 不加入 _RIGHT_NOISE（否则会误杀"传说""知道"等合法词）。
_DIALOGUE_TRAIL = frozenset('说道问笑叫喊')

# 这些字出现在 4+ 字串的内部（即 w[1:-1]）时，使用短词判断。
# 2 字词的 w[1:-1] 为空，"着急""得到""了解"等合法词不受影响。
_LONG_PHRASE_INTERIOR = frozenset(
    '的地得'          # 结构助词
    '着了过'          # 动态助词
    '到在从向往'      # 介词
    '把被让将给'      # 处置/被动标记
    '我你他她它们'    # 代词
    '个些位只次'        # 量词
)

_RE_CHINESE = re.compile(r'[\u4e00-\u9fff]+')
_RE_PURE_CHINESE = re.compile(r'^[\u4e00-\u9fff]+$')


@dataclass
class StrategyResult:
    """六策略的统一返回类型。
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
    n = len(scores)
    if n == 1:
        return {k: 1.0 for k in scores}
    sorted_words = sorted(scores, key=scores.get, reverse=True)
    return {w: 1.0 - i / (n - 1) for i, w in enumerate(sorted_words)}


def _clean_boundary(word: str) -> bool:
    if word[0] in _LEFT_NOISE or word[-1] in _RIGHT_NOISE:
        return False
    for i in range(1, len(word) - 1):
        if word[i] in _PHRASE_MARKERS:
            return False
    return True
