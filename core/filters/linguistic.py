"""
语言学过滤器：和语料无关，只看字面。L1 / L2 候选共用同一条管道。

每个过滤器都是纯函数 (word, *, min_len, max_len) -> Verdict。
Verdict.passed=False 时附带 reason（形如 'boundary.right_noise'），
便于日志、UI 展示和单测。

六类过滤器（按代价由低到高排列，管道按此序短路判定）：
  1. length          — 长度窗口
  2. boundary        — 两端噪字 + 中间强助词 + 长词内部禁字
  3. blacklist       — 伪名硬名单
  4. structural      — 量词打头 / 数字+量词 / 2 字前缀黑名单
  5. dialogue_tail   — 末字是对白动词首字（'说/道/问/笑/叫/喊'）
  6. adverbial_di    — "副词+地+动词"结构（跑 jieba.posseg，仅在含'地'时触发）

两个预设管道：
  DEFAULT_LING_PIPELINE  — L2（正则捕获）专用。六条全开。
  L1_LING_PIPELINE       — L1（n-gram 统计）专用。不含 dialogue_tail：
                           L1 侧本身对"名字+对白动词"有 context-aware 的
                           额外检查（_vocab_builder.py），无差别砍掉
                           末字为 '说/道' 的词会误伤"传说""知道"这类合法词。

原 _pattern_miner._valid_candidate 的 12 条规则完全被吃进 DEFAULT_LING_PIPELINE，
本文件即规则的唯一权威来源；_utils 只保留"原料"字表（_LEFT_NOISE 等）。
"""

from dataclasses import dataclass

from .._utils import (
    _LEFT_NOISE, _RIGHT_NOISE, _PHRASE_MARKERS,
    _DIALOGUE_TRAIL, _LONG_PHRASE_INTERIOR,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Verdict：过滤器的统一返回
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str = ''


_OK = Verdict(passed=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 硬编码字表（从 _pattern_miner 迁入，这里是唯一声明点）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 伪名黑名单（模板常会碰到的高频代词/连词/虚化名词）
BLACK_WORDS = frozenset({
    # 指示代词 / 连词
    '这个', '那个', '其中', '于是', '所以', '然后', '因此', '但是',
    '不过', '这时', '那时', '此时', '此刻', '此后', '随后', '众人',
    '大家', '我们', '他们', '她们', '它们', '你们', '自己', '别人',
    '一些', '一点', '一切', '一边', '一旁', '什么', '怎么', '如何',
    '那里', '这里', '哪里', '何处', '何方', '事情', '东西', '地方',
    '时候', '之后', '之前', '以后', '以前', '结果', '办法',
    # 类属抽象词（不是专名，是范畴词）
    '强者', '佣兵', '种族', '势力', '核心', '目光', '记名', '习会',
    '二人', '三人', '四人', '五人', '众多', '众位', '各位', '诸位',
    '此人', '此物', '吾儿', '修者', '长辈', '晚辈', '同门', '同辈',
    '前辈', '后辈', '门人', '弟子', '族人', '家人', '友人', '故人',
    '之人', '之物', '之流', '之辈', '之内', '之外', '之上', '之下',
    '美其名', '做记名', '天材地宝', '真空地带',
    # "另/其/数量 + 单位" 类泛化引用
    '另一', '另外', '一名', '两名', '三名', '一位', '两位', '三位',
    '一个', '两个', '三个', '一只', '两只', '三只',
})

# 数字字（"一二三..."）。**只**在后接量词单位字时用于拦截，
# 避免误伤"九阳神功""五岳剑派""三国演义""八荒六合"等武侠命名。
NUM_PREFIX_CHARS = frozenset(
    '一二三四五六七八九十百千万两几众多每某另'
)

# 量词单位字（"名位个只头尊" + X，X 才是专名）。
# 注意：**不含"张"**——"张"作为姓氏极常见（张无忌、张三丰），
# "三张纸"这种误用可以靠 L1 分解（桌子/纸 在 L1）后期再拦。
QUANTIFIER_UNITS = frozenset(
    '名位个只头尊条匹根片块场阵把段丝缕支杯盘串团朵人'
)

# 2 字前缀黑名单（常见副词/修饰短语起始）
BAD_PREFIXES = frozenset({
    '另外', '其他', '其中', '此外', '另一', '一个', '一种', '一名',
    '一位', '几位', '几名', '几个', '哪位', '谁家', '那位', '这位',
    '每位', '众位', '各位',
    '做记', '美其', '清楚', '忍不', '冲着', '虽然', '见到', '最后',
    '三名', '两名', '两位', '三位', '四名', '四位', '五名', '五位',
    '他清', '她清', '我清',      # "他清楚地知..."
    '这名', '那名', '每名', '众名',
})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 单个过滤器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_length(word: str, *,
                 min_len: int = 2, max_len: int = 8, **_) -> Verdict:
    if not (min_len <= len(word) <= max_len):
        return Verdict(False, 'length')
    return _OK


def check_boundary(word: str, **_) -> Verdict:
    """两端不能是噪字，中间不能含强助词，长词内部不能含助/介/代/量字。

    合并了原 _utils._clean_boundary 与 _vocab_builder 里独立的
    _LONG_PHRASE_INTERIOR 检查，变成一处。
    """
    if not word:
        return Verdict(False, 'boundary.empty')
    if word[0] in _LEFT_NOISE:
        return Verdict(False, 'boundary.left_noise')
    if word[-1] in _RIGHT_NOISE:
        return Verdict(False, 'boundary.right_noise')
    for i in range(1, len(word) - 1):
        if word[i] in _PHRASE_MARKERS:
            return Verdict(False, 'boundary.mid_phrase_marker')
    if len(word) >= 4:
        for c in word[1:-1]:
            if c in _LONG_PHRASE_INTERIOR:
                return Verdict(False, 'boundary.long_phrase_interior')
    return _OK


def check_blacklist(word: str, **_) -> Verdict:
    if word in BLACK_WORDS:
        return Verdict(False, 'blacklist')
    return _OK


def check_structural(word: str, **_) -> Verdict:
    """量词打头 / 数字+量词 / 2 字前缀黑名单。"""
    if not word:
        return _OK
    if word[0] in QUANTIFIER_UNITS:
        return Verdict(False, 'structural.quantifier_head')
    if (len(word) >= 2
            and word[0] in NUM_PREFIX_CHARS
            and word[1] in QUANTIFIER_UNITS):
        return Verdict(False, 'structural.num_unit')
    if len(word) >= 3 and word[:2] in BAD_PREFIXES:
        return Verdict(False, 'structural.bad_prefix')
    return _OK


def check_dialogue_tail(word: str, **_) -> Verdict:
    """末字是对白动词首字 → 拒绝（避免"道/说"粘入专名）。

    只对 L2 生效（DEFAULT_LING_PIPELINE 包含，L1_LING_PIPELINE 不含）。
    L2 的捕获组恰好出现在"某某说/道"前面，末字粘到对白动词的概率远超
    L1 统计通道，因此采用这条强规则；L1 有独立的 context-aware 检查
    （见 _vocab_builder.py，需要 prefix 本身也在 solid 里才拒绝）。
    """
    if word and word[-1] in _DIALOGUE_TRAIL:
        return Verdict(False, 'dialogue_tail')
    return _OK


# ── jieba.posseg 懒加载（仅 adverbial_di 使用）──

_jieba_pseg = None


def _get_pseg():
    """懒加载 jieba.posseg，避免 import 时触发 jieba 初始化。"""
    global _jieba_pseg
    if _jieba_pseg is None:
        import jieba
        import jieba.posseg as pseg
        jieba.setLogLevel(60)   # 静音 jieba 初始化日志
        _jieba_pseg = pseg
    return _jieba_pseg


def check_adverbial_di(word: str, **_) -> Verdict:
    """判断含"地"的候选是否是"副词/形容词 + 地 + 动词"结构。

    三条独立规则（原 _pattern_miner._is_adverbial_di_phrase），任一命中
    即判为副词短语：
      R1. 存在 token 为 '地'，词性标为 uv/u/ud/ug（结构助词）
          → 明确的副词标记（无奈地开口、惊骇地失声）
      R2. 首 token 词性以 z / d / ad 开头（状态词 / 副词 / 副形）
          → 专名极少以这种词性起始（淡淡地开口 → 首 z）
      R3. 存在非单字 token 以"地"开头（如 '地知'、'地轻'）
          → jieba 把 "...地 + X" 误粘成一个词（清楚地知 → 地知/n）

    对 '九幽地冥蟒' 三条都不命中 → 放行（首 m"九"、无 uv、
    "幽地"开头是"幽"）。

    首字是"地"（如"地球""地下"）直接放行，不走 jieba。
    """
    if '地' not in word[1:]:
        return _OK
    tokens = list(_get_pseg().cut(word))
    if not tokens:
        return _OK
    # R1
    for tk in tokens:
        if tk.word == '地' and tk.flag in ('uv', 'u', 'ud', 'ug'):
            return Verdict(False, 'adverbial_di.structural_particle')
    # R2
    first_flag = tokens[0].flag or ''
    if first_flag[:1] in ('z', 'd') or first_flag[:2] == 'ad':
        return Verdict(False, 'adverbial_di.head_adverb')
    # R3
    for tk in tokens:
        if len(tk.word) >= 2 and tk.word[0] == '地':
            return Verdict(False, 'adverbial_di.di_glue')
    return _OK


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 管道：按序短路执行若干过滤器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LingPipeline:
    """按序执行若干过滤器，任一拒绝即短路返回。

    filters 是 (id, fn) 列表；id 用于未来按需开关或做拒绝统计。
    fn 签名统一为 (word, *, min_len, max_len) -> Verdict。
    """

    __slots__ = ('filters',)

    def __init__(self, filters):
        self.filters = tuple(filters)

    def check(self, word: str, *,
              min_len: int = 2, max_len: int = 8) -> Verdict:
        for _id, fn in self.filters:
            v = fn(word, min_len=min_len, max_len=max_len)
            if not v.passed:
                return v
        return _OK

    def is_valid(self, word: str, *,
                 min_len: int = 2, max_len: int = 8) -> bool:
        return self.check(word, min_len=min_len, max_len=max_len).passed


# 预设管道：L2 专用（正则捕获后的校验）
DEFAULT_LING_PIPELINE = LingPipeline([
    ('length',         check_length),
    ('boundary',       check_boundary),
    ('blacklist',      check_blacklist),
    ('structural',     check_structural),
    ('dialogue_tail',  check_dialogue_tail),
    ('adverbial_di',   check_adverbial_di),
])

# L1 专用：不含 dialogue_tail
# 原因：L1 候选来自 n-gram 统计，末字为 说/道 的"传说""知道"等合法词
# 非常常见；L1 侧对"名字+对白动词"另有 context-aware 判定
# （见 _vocab_builder.py）。
L1_LING_PIPELINE = LingPipeline([
    ('length',         check_length),
    ('boundary',       check_boundary),
    ('blacklist',      check_blacklist),
    ('structural',     check_structural),
    ('adverbial_di',   check_adverbial_di),
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# trim_noise：L2 专用的预清洗工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def trim_noise(word: str) -> str:
    """去两端噪字，并按最后一个中间助词切割，返回"最长合法子串"。

    仅用于 L2（正则捕获）的预清洗：正则命中串常带一两个噪字或"的/和/与/是"
    这类连接词，trim 后再走 LingPipeline 校验。L1 候选来自 n-gram，本身
    就是"干净段"，不需要 trim。

    原 _pattern_miner._trim_noise 搬家到此，行为完全等价。
    """
    if not word:
        return ''
    lo, hi = 0, len(word)
    while lo < hi and word[lo] in _LEFT_NOISE:
        lo += 1
    while hi > lo and word[hi - 1] in _RIGHT_NOISE:
        hi -= 1
    if lo >= hi:
        return ''
    trimmed = word[lo:hi]
    cut = -1
    for i, c in enumerate(trimmed):
        if c in _PHRASE_MARKERS:
            cut = i
    if cut >= 0:
        trimmed = trimmed[cut + 1:]
    return trimmed
