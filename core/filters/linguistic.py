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
                           L2 候选来自"X说/X道"附近的正则，捕获即大概率
                           为"名字+对白动词"粘连，兜底合理；L1 候选是
                           高频 n-gram，无差别砍"末字说/道/笑/问"会
                           误伤"传说 / 微笑 / 询问 / 通道"这类通用词，
                           甚至误杀"龙王传说"之类的真书名/专名。
                           L1 侧对"名字+对白动词"另有 context-aware
                           判定（见 _vocab_builder.py，要求 prefix 自身
                           也是 solid 才拒绝），对 false-positive 更友好。

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

# 伪名黑名单（精确匹配拦截：模板常会碰到的高频代词/连词/虚化名词，本身无专名实体含义）
#
# 分工原则：
#   • "数字+量词"结构（如"一名/两位/三个/四只..."）一律交给
#     check_structural.num_unit 统一处理（reason=structural.num_unit），
#     此处不枚举，避免"一到三走 blacklist、四起走 num_unit"的报因不一致。
#   • "另一 / 另外"等"另"打头但第二字不是量词单位的词，num_unit 吃不到，
#     必须显式列在此处。
BLACK_WORDS = frozenset({
    # ── 指示代词 / 连词（高频语法粘合剂）──
    '这个', '那个', '其中', '此外',
    '于是', '所以', '然后', '因此', '但是', '不过',
    '这时', '那时', '随后',
    '前者', '后者', '总之', '反之',

    # ── 之 X 系（方位 / 时点 / 类属，语法虚化前缀为"之"）──
    '之上', '之下', '之内', '之外', '之前', '之后',
    '之间', '之中', '之时', '之际', '之初', '之处',
    '之人', '之物', '之流', '之辈',

    # ── 此 X 系（书面指示代词，用于上下文代指而非命名）──
    '此时', '此刻', '此后',
    '此人', '此物',
    '此处', '此地', '此事', '此番', '此次', '此行', '此生', '此身',

    # ── 以 X 系（时间 / 范围指代，与"之X"互补）──
    '以上', '以下', '以内', '以外', '以前', '以后', '以来', '以往',

    # ── 这 / 那 / 哪 + X（次数 / 样态 / 程度 / 方位 / 类型）──
    '这次', '那次', '每次', '每回',
    '这般', '那般', '这样', '那样', '这么', '那么',
    '这种', '那种',
    '这里', '那里', '哪里',
    '这边', '那边', '两边', '旁边',

    # ── 疑问 / 何 X 系 ──
    '什么', '怎么', '怎样', '如何', '为何', '为什么',
    '何处', '何方', '何时', '何人', '何事',

    # ── 群体 / 自指代词 ──
    '大家', '众人',
    '我们', '他们', '她们', '它们', '你们',
    '自己', '别人', '人家', '本人',
    '你我', '咱们', '我等', '我辈',

    # ── 量化修饰（泛指，无具体实体属性）──
    '一些', '一点', '一切', '一边', '一旁',

    # ── 类属抽象词（身份 / 境界 / 关系 / 范畴词，描述"是什么"而非"叫什么"）──
    '强者', '佣兵', '种族', '势力', '目光', '记名', '习会',
    '二人', '三人', '四人', '五人', '众多',
    '众位', '各位', '诸位',
    '吾儿', '修者',
    '长辈', '晚辈', '前辈', '后辈', '同门', '同辈',
    '门人', '弟子', '族人', '家人', '友人', '故人',

    # ── 通用事件 / 时空 / 结果词 ──
    '事情', '东西', '地方', '时候', '结果', '办法',

    # ── 固定熟语片段（正则 / n-gram 常捕获到的残片）──
    '美其名', '做记名', 

    # ── 另 X（非"数字+量词"结构，num_unit 吃不到，必须显式列出）──
    '另一', '另外',
})

# 数字字（"一二三..."）。
# 作用：配合 QUANTIFIER_UNITS 拦截"数字+量词+名词"结构（如"三名魂师"）。
# 策略：**只**在后接量词单位字时才拦截，避免误伤"九阳神功""五岳剑派"等合法武侠/玄幻命名。
NUM_PREFIX_CHARS = frozenset(
    '一二三四五六七八九十百千万两几众多每某另'
)

# 量词单位字（"名位个只头尊" + X，X 才是专名）。
# 作用：拦截"量词打头"的词（如"个魂环"）或配合数字字拦截。
# 注意：**不含"张"**——"张"作为姓氏极常见（张无忌、张三丰），
# "三张纸"这种误用可以靠 L1 分解（桌子/纸 在 L1）后期再拦。
QUANTIFIER_UNITS = frozenset(
    '名位个只头尊条匹根片块场阵把段丝缕支杯盘串团朵人'
)

# 2 字前缀黑名单（前缀匹配拦截：常见副词/修饰短语/动宾结构起始）
# 作用：当候选词以这些 2 字开头时直接拒绝，防止修饰语粘连到专名上。
#
# 注意：凡是"数字字 + 量词字"的前缀（如"一名/两位/三个/四只..."）都已由
# check_structural 的 num_unit 规则在更早位置拦下（num_unit 在 bad_prefix
# 之前短路），此处列它们是死代码，一律不再枚举。
BAD_PREFIXES = frozenset({
    # ── 泛指 / 指代类前缀（非"数字+量词"结构，num_unit 吃不到）──
    '另外', '其他', '其中', '此外', '另一', '一种',
    '哪位', '谁家', '那位', '这位', '每位', '众位', '各位', '诸位',
    '这名', '那名', '每名',
    '所有', '一整', '整个',

    # ── 动词 / 副词 / 连词类前缀（动作或状态粘连，如"见到(唐三)"）──
    '做记', '美其', '清楚', '忍不', '冲着', '虽然', '见到', '最后',

    # ── 主谓结构前缀（常出现在"他清楚地知道"等被错误切分的片段中）──
    '他清', '她清', '我清',
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
    L1 统计通道，因此采用这条强规则。

    L1 侧不挂：n-gram 里"传说 / 微笑 / 询问 / 通道"等通用词、以及
    "龙王传说"这类把"说/道/笑/问"作为结尾字的合法书名/专名非常常见，
    一刀切会误伤。L1 有独立的 context-aware 检查
    （见 _vocab_builder.py，要求 prefix 本身也在 solid 里才拒绝）。
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

    __slots__ = ('filters', 'enable_stats', 'stats')

    def __init__(self, filters):
        self.filters = tuple(filters)
        self.enable_stats = False
        # stats 结构: { reason: { word: count } }
        self.stats: dict[str, dict[str, int]] = {}

    def check(self, word: str, *,
              min_len: int = 2, max_len: int = 8) -> Verdict:
        for _id, fn in self.filters:
            v = fn(word, min_len=min_len, max_len=max_len)
            if not v.passed:
                if self.enable_stats:
                    reason = v.reason or _id
                    if reason not in self.stats:
                        self.stats[reason] = {}
                    self.stats[reason][word] = self.stats[reason].get(word, 0) + 1
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
# 原因：L1 候选来自 n-gram 统计，末字为 说/道/笑/问 的"传说 / 微笑 /
# 询问 / 通道"等通用词非常常见，"龙王传说"这类书名/专名也会被误伤。
# L1 侧对"名字+对白动词"另有 context-aware 判定（见 _vocab_builder.py）。
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
