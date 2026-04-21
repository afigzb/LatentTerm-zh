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
    """按值归一化 + 尖锐度衰减：s/max × (1 - mean/max)。

    纯 s/max 会把离散 / 粗粒度策略（char_overlap、morpheme）的"塌顶高原"
    原样送进融合层——几十个候选并列在 max，每人白拿 w_char + w_morph
    ≈ 0.40 的绝对分，挤压真正有区分度的通道（cooccurrence / context /
    co_topic）对排名的影响。

    加一个尖锐度因子 sharpness = 1 - mean/max：
      - Lift / IDF 类陡分布：mean/max 很小 → sharpness ≈ 1，几乎无影响，
        保留策略内部精心设计刻度的原始坡度；
      - 塌顶高原：mean/max 接近 1 → sharpness ≈ 0，整个策略自动降权，
        "没话可说时不占用融合的绝对分数"；
      - 中间情况按比例衰减。

    这样六策略仍然共用一个归一化函数，但每条通道对融合的贡献自动按
    "本次是否真的有区分度"加权，不需要用户调 w_* 来手动补偿。

    单一候选视为稀有强证据，保留满分 1.0（没有对比对象时不做衰减）。
    """
    if not scores:
        return {}
    vals = list(scores.values())
    max_score = max(vals)
    if max_score <= 0:
        return {w: 0.0 for w in scores}
    n = len(vals)
    if n == 1:
        return {w: 1.0 for w in scores}
    mean_score = sum(vals) / n
    sharpness = 1.0 - mean_score / max_score  # ∈ [0, 1)
    return {w: (s / max_score) * sharpness for w, s in scores.items()}


def _clean_boundary(word: str) -> bool:
    if word[0] in _LEFT_NOISE or word[-1] in _RIGHT_NOISE:
        return False
    for i in range(1, len(word) - 1):
        if word[i] in _PHRASE_MARKERS:
            return False
    return True
