"""
通道 B / C：高精度模板狙击 + 关键词自适应扩充。

通道 B（build_index 时一次性运行）
  - 对原文直接跑一组高精度正则，命中即入选
  - 完全跳过 PMI / 分词 / 自由度
  - 专攻只出现 2~10 次的孤岛专名（剧情道具/副本人物/边缘地名）

通道 C（extract 时按关键词临时运行）
  - 收集关键词左右最频繁的 2~3 字指纹
  - 在原文里用"指纹 + X"形式反向匹配 X
  - 产出临时候选，只在本次 extract 生效

类型标签：5 大类 + 1 兜底
  person     人名
  place      地点
  creature   生物（魂兽/妖兽等）
  skill      招式/功法/法宝/丹药等物·技合并
  group      宗门/组织/家族
  misc       其他（未分类）
"""

import re
from collections import Counter, defaultdict

from ._utils import _LEFT_NOISE, _RIGHT_NOISE, _PHRASE_MARKERS, _DIALOGUE_TRAIL


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 模板定义
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 说明
#   每条模板 = (id, regex, group_idx, type)
#   - regex 必须有捕获组，group_idx 指定哪一组是候选词
#   - 所有正则用 (?<![\u4e00-\u9fff]) 做前置锚点，避免"突然张三"这类吃前缀
#   - 捕获到候选后，还会走一次 _trim_noise（去两端噪字）+ 长度过滤
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_CH = r'[\u4e00-\u9fff]'
_NOT_CH_BEFORE = r'(?<![\u4e00-\u9fff])'
_NOT_CH_AFTER = r'(?![\u4e00-\u9fff])'

_DIALOG_VERBS = (
    # 2+ 字对白动词。单字"说"/"道"被拆出去单独处理（B1b），避免
    # "淡淡地开口道" 这类副词+地+动词被误识别为人名。
    '说道|笑道|问道|答道|怒道|喝道|叫道|沉声道|冷声道|淡淡道|冷笑道|'
    '轻声道|朗声道|低声道|怒喝道|厉声道|苦笑道|叹道|附和道|低喝道|'
    '嘟囔道|心道|暗道|自语道|嘀咕道|喃喃道|连忙道|急道|微笑道|傲然道|'
    '哈哈大笑道|缓缓道|开口道|回答道|解释道|反问道|补充道|疑惑道|'
    '惊讶道|惊呼道|长叹道|呵呵道|叹息道|轻笑道|冷哼道|皱眉道'
)

_NAMING_VERBS = (
    '名为|名叫|叫做|叫作|唤作|号为|号曰|称为|人称|世称|自称|又称|'
    '名曰|号称|此乃|此物名曰|此剑名曰|此功名曰|此阵名曰'
)

_CREATURE_QUANTS = '一头|一只|一尊|一条|一匹|这头|那头|这只|那只|两头|几只|众多'

_PLACE_VERBS = (
    '来到|进入|走进|踏入|闯入|飞入|抵达|返回|离开|前往|赶往|奔向|'
    '路过|飞往|踏足|进驻|镇守|潜入'
)

_SKILL_VERBS = (
    '使出|施展|祭出|释放|修炼|领悟|激发|催动|运转|练成|使用|催发|'
    '打出|施放|修习|研习|掌握|精通|习得'
)

# 地点/组织/生物后缀（构词模板）
_PLACE_TAIL = '山|岛|城|峰|谷|洞|湖|海|林|原|界|境|村|镇|州|关|府|院|宫|阁|塔'
_GROUP_TAIL = '宗|门|派|教|会|帮|盟|殿|堂|阁|族|家|楼|庄|堡|阵营|势力'

PATTERNS = [
    # ── B1  对白说话人（最高置信度，多字动词）────────────────────────
    # 右侧是 2+ 字对白动词，动词本身就是天然分界。不再允许 optional
    # 状语+地（那会把"淡淡地开口"吃进 name）。
    ('B1.dialog',
     rf'{_NOT_CH_BEFORE}({_CH}{{2,5}}?)(?:{_DIALOG_VERBS})',
     1, 'person'),

    # ── B1b  对白说话人（单字动词：说/道）──────────────────────────
    # 前置必须是句子断点，后置必须是冒号/引号/逗号。这两个锚点把
    # "淡淡地开口道：" "清楚地知道，" 这类排除掉——因为它们前面不是
    # 句子断点而是上一个字（中文或"地"）。
    ('B1b.dialog_short',
     rf'(?<=[，。；？！…"「『」』\n\r 　])({_CH}{{2,5}}?)[说道](?=[:：「『""，])',
     1, 'person'),

    # ── B2  命名套式（极高置信度）─────────────────────────────────────
    # 名词常被标点或"的/之/是"包围 → 非贪婪 + 尾部要求非中文或显式停止字
    ('B2.named',
     rf'(?:{_NAMING_VERBS})[「『"]?({_CH}{{2,6}}?)(?=[」』"]|[^\u4e00-\u9fff]|的|之|是|$)',
     1, 'misc'),

    # ── B3a  生物量词 ────────────────────────────────────────────────
    ('B3.creature',
     rf'(?:{_CREATURE_QUANTS})({_CH}{{2,5}}?)(?=[^\u4e00-\u9fff]|的|在|向|朝|往|$)',
     1, 'creature'),

    # ── B3b  地点动作动词 ────────────────────────────────────────────
    ('B3.place',
     rf'(?:{_PLACE_VERBS})({_CH}{{2,6}}?)(?=[^\u4e00-\u9fff]|的|之后|之前|时|后|$)',
     1, 'place'),

    # ── B3c  技能动作动词 ────────────────────────────────────────────
    ('B3.skill',
     rf'(?:{_SKILL_VERBS})({_CH}{{2,6}}?)(?=[^\u4e00-\u9fff]|化解|击退|将|之|的|将来|$)',
     1, 'skill'),

    # ── B4a  组织后缀（构词式，自带右锚）────────────────────────────
    ('B4.group',
     rf'{_NOT_CH_BEFORE}({_CH}{{1,5}}(?:{_GROUP_TAIL})){_NOT_CH_AFTER}',
     1, 'group'),

    # ── B4b  地点后缀（构词式，自带右锚）───────────────────────────
    ('B4.place',
     rf'{_NOT_CH_BEFORE}({_CH}{{1,4}}(?:{_PLACE_TAIL})){_NOT_CH_AFTER}',
     1, 'place'),

    # ── B5  角色身份 + 专名（XX掌门/XX教主/XX宗主/XX长老）───────────
    # 右侧有身份后缀锚点 → 非贪婪
    ('B5.role',
     rf'{_NOT_CH_BEFORE}({_CH}{{2,5}}?)(?:掌门|教主|宗主|门主|帮主|'
     rf'长老|弟子|大人|前辈|师兄|师姐|师叔|师伯|师傅|师父)',
     1, 'person'),
]

# 模板 id → 类型（建索引用）
_TEMPLATE_TYPE = {p[0]: p[3] for p in PATTERNS}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 关键词类型推断（用户输入"魂兽"→creature）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_KEYWORD_TYPE_HINTS: dict[str, tuple[str, ...]] = {
    'creature': (
        '魂兽', '妖兽', '怪兽', '灵兽', '凶兽', '神兽', '异兽',
        '蛊虫', '魔物', '妖怪',
    ),
    'skill': (
        '功法', '武功', '剑法', '刀法', '拳法', '身法', '心法',
        '剑诀', '神功', '秘籍', '法术', '招式', '绝学', '神通',
        '丹药', '灵药', '仙丹', '法宝', '宝物', '灵器', '神器',
    ),
    'place': (
        '城池', '洞府', '秘境', '禁地',
    ),
    'group': (
        '门派', '宗门', '势力', '家族', '组织', '帮会',
    ),
    'person': (
        '人物', '角色', '高手', '武者',
    ),
}


def infer_keyword_type(keyword: str) -> str | None:
    """关键词类型推断：如果关键词包含类型提示字串，返回该类型。"""
    for t, hints in _KEYWORD_TYPE_HINTS.items():
        for h in hints:
            if h in keyword:
                return t
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 候选校验
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 伪名黑名单（模板常会碰到的高频代词/连词/虚化名词）
_BLACK_WORDS = frozenset({
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


def _trim_noise(w: str) -> str:
    """去两端噪字，并切掉中间助词。返回"最长合法子串"。"""
    if not w:
        return ''
    # 两端 trim
    lo, hi = 0, len(w)
    while lo < hi and w[lo] in _LEFT_NOISE:
        lo += 1
    while hi > lo and w[hi - 1] in _RIGHT_NOISE:
        hi -= 1
    if lo >= hi:
        return ''
    # 中间助词：如果有"的/和/与/是"，按最后一个助词后切
    trimmed = w[lo:hi]
    cut = -1
    for i, c in enumerate(trimmed):
        if c in _PHRASE_MARKERS:
            cut = i
    if cut >= 0:
        trimmed = trimmed[cut + 1:]
    return trimmed


# 数字字（"一二三..."）。**只**在后接量词单位字时用于拦截，
# 避免误伤"九阳神功""五岳剑派""三国演义""八荒六合"等武侠命名。
_NUM_PREFIX_CHARS = frozenset(
    '一二三四五六七八九十百千万两几众多每某另'
)
# 量词单位字（"名位个只头尊" + X，X 才是专名）。
# 注意：**不含"张"**——"张"作为姓氏极常见（张无忌、张三丰），
# "三张纸"这种误用可以靠 L1 分解（桌子/纸 在 L1）后期再拦。
_QUANTIFIER_UNITS = frozenset(
    '名位个只头尊条匹根片块场阵把段丝缕支杯盘串团朵人'
)

# 2 字前缀黑名单（常见副词/修饰短语起始）
_BAD_PREFIXES = frozenset({
    '另外', '其他', '其中', '此外', '另一', '一个', '一种', '一名',
    '一位', '几位', '几名', '几个', '哪位', '谁家', '那位', '这位',
    '每位', '众位', '各位',
    '做记', '美其', '清楚', '忍不', '冲着', '虽然', '见到', '最后',
    '三名', '两名', '两位', '三位', '四名', '四位', '五名', '五位',
    '他清', '她清', '我清',      # "他清楚地知..."
    '这名', '那名', '每名', '众名',
})


# Jieba 词性分析（懒加载），用于识别"副词+地+动词"结构
_jieba_pseg = None


def _get_pseg():
    """懒加载 jieba.posseg，避免每次 import 触发 jieba 初始化。"""
    global _jieba_pseg
    if _jieba_pseg is None:
        import jieba
        import jieba.posseg as pseg
        jieba.setLogLevel(60)   # 静音 jieba 初始化日志
        _jieba_pseg = pseg
    return _jieba_pseg


def _is_adverbial_di_phrase(w: str) -> bool:
    """Jieba 视角下判断 w 是否是"副词/形容词 + 地 + 动词"结构。

    三条独立规则，任一命中即判为副词短语（拒绝）：
      R1. 存在 token 为 '地'，词性标为 uv/u/ud/ug（结构助词）
          → 明确的副词标记（无奈地开口、惊骇地失声）
      R2. 首 token 词性以 z/d/ad 开头（叠字状态词/副词/副形）
          → 专名极少以这种词性起始（淡淡地开口 → 首 z）
      R3. 存在非单字 token 以"地"开头（如 '地知'、'地轻'）
          → jieba 把 "...地 + X" 误粘成一个词（清楚地知 → 地知/n）

    对 '九幽地冥蟒' 三条都不命中 → 放行（首 m"九"、无 uv、
    "幽地"开头是"幽"）。
    """
    if '地' not in w:
        return False
    tokens = list(_get_pseg().cut(w))
    if not tokens:
        return False
    # R1: 结构助词"地"
    for tk in tokens:
        if tk.word == '地' and tk.flag in ('uv', 'u', 'ud', 'ug'):
            return True
    # R2: 首 token 是状态/副词性
    first_flag = tokens[0].flag or ''
    if first_flag[:1] in ('z', 'd'):
        return True
    if first_flag[:2] == 'ad':
        return True
    # R3: 非单字 token 以"地"起始（jieba 乱粘信号）
    for tk in tokens:
        if len(tk.word) >= 2 and tk.word[0] == '地':
            return True
    return False


def _valid_candidate(w: str, min_len: int = 2, max_len: int = 8) -> bool:
    if not (min_len <= len(w) <= max_len):
        return False
    if w in _BLACK_WORDS:
        return False
    if w[0] in _LEFT_NOISE or w[-1] in _RIGHT_NOISE:
        return False
    # 最后一个字是对白动词首字，大概率把"道/说"粘进来了
    if w[-1] in _DIALOGUE_TRAIL:
        return False
    # 首字是量词单位 → 拒绝（"名斗宗""位前辈"首字是单位字）
    if w[0] in _QUANTIFIER_UNITS:
        return False
    # 首字数字 + 第二字量词单位 → 拒绝（"三名XX""一头XX""两只XX"）
    # 但"九阳""五岳""三国""七窍"这类放行（第二字是名词不是量词）
    if len(w) >= 2 and w[0] in _NUM_PREFIX_CHARS and w[1] in _QUANTIFIER_UNITS:
        return False
    # 2 字前缀黑名单（适用于 3+ 字候选）
    if len(w) >= 3 and w[:2] in _BAD_PREFIXES:
        return False
    # 中间含强助词 → 拒绝
    for i in range(1, len(w) - 1):
        if w[i] in _PHRASE_MARKERS:
            return False
    # 含"地"时做 Jieba 词性判定（替换掉原本的"中间含地即拒"硬规则）
    if '地' in w[1:] and _is_adverbial_di_phrase(w):
        return False
    return True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# L1 分解检查：用 L1 词表反验 L2 候选是否是"多个 L1 词的短语拼凑"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 原理：
#   L1 vocab 是经过 PMI + 左右熵 + Trie 长词优先切分 过滤出来的"结实"
#   词集。如果某个 L2 候选 w 可以被 L1 词集完全覆盖切分（且片段 ≥2 字，
#   排除单字零碎），说明它是 L1 词 + L1 词的意外共现，不是独立词素。
#
# 为什么不会误伤真复合词（如 "九阳神功"）：
#   - 如果它真是整体常见 → 自己就过了 PMI/熵 → 进 L1，根本不走 L2
#   - Trie 长词优先切分会把子词 "九阳"/"神功" 的 clean_freq 压低，
#     它们反而可能进不了 L1 vocab → 分解查询命中 False → 保留 w

def can_decompose_by_l1(w: str, l1_words, min_token_len: int = 2) -> bool:
    """判断 w 是否能被 L1 词表完全切分成 ≥2 个 ≥min_token_len 字的片段。

    用动态规划：dp[i] = True 表示 w[:i] 可完全切分。
    """
    n = len(w)
    if n < 2 * min_token_len:
        return False
    dp = [False] * (n + 1)
    dp[0] = True
    # 最长 L1 词长度通常 ≤ 8，搜索窗口限制在 8
    for i in range(min_token_len, n + 1):
        for j in range(max(0, i - 8), i - min_token_len + 1):
            if dp[j] and w[j:i] in l1_words:
                dp[i] = True
                break
    return dp[n]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PatternMiner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PatternMiner:
    """对原文跑高精度模板，产出低频专名候选池。"""

    # 同一个词被类型投票时，多类型冲突 → misc
    def __init__(self, min_len: int = 2, max_len: int = 8,
                 max_evidence: int = 5):
        self.min_len = min_len
        self.max_len = max_len
        self.max_evidence = max_evidence
        self._compiled = [
            (pid, re.compile(rx), gi, ty)
            for pid, rx, gi, ty in PATTERNS
        ]

    def mine(self, text: str) -> dict[str, dict]:
        """返回 {word: {freq, templates, types, type, evidence, origins}}

        freq       — 正则命中次数（近似，不是全文 count）
        templates  — 命中的模板 id 集合
        types      — 类型投票 Counter
        type       — 最终裁定的类型（最高票；平票归 misc）
        evidence   — [(pos, tmpl_id), …]，截断到 max_evidence
        origins    — {'B1', 'B2', …}（模板 id 的第一段）
        """
        hits: dict[str, dict] = {}

        for pid, rx, gi, ty in self._compiled:
            origin = pid.split('.')[0]
            for m in rx.finditer(text):
                raw = m.group(gi)
                w = _trim_noise(raw)
                if not _valid_candidate(w, self.min_len, self.max_len):
                    continue
                pos = m.start(gi)

                entry = hits.get(w)
                if entry is None:
                    entry = {
                        'freq': 0,
                        'templates': set(),
                        'types': Counter(),
                        'evidence': [],
                        'origins': set(),
                    }
                    hits[w] = entry

                entry['freq'] += 1
                entry['templates'].add(pid)
                entry['origins'].add(origin)
                entry['types'][ty] += 1
                if len(entry['evidence']) < self.max_evidence:
                    entry['evidence'].append((pos, pid))

        # 类型裁定：最高票；若 misc 和另一类平票，选另一类；全 misc 则 misc
        for w, e in hits.items():
            e['type'] = _resolve_type(e['types'])

        return hits

    # ----------------------------------------------------------------- #
    # 通道 C：关键词自适应模板                                              #
    # ----------------------------------------------------------------- #

    def channel_c(self, keyword: str, text: str,
                  positions: list[int],
                  known_vocab: set[str] | None = None,
                  top_k_each_side: int = 4,
                  min_pattern_hits: int = 2) -> dict[str, dict]:
        """基于关键词上下文的自适应模板抽取。

        算法：
          1. 在 keyword 每个出现位置，取左/右各 2~3 字作为"指纹"
          2. 对左右指纹各取 top-K 高频
          3. 对每个指纹 fp，在全文搜 fp + X（或 X + fp）形式
          4. 命中的 X 作为 L2 临时候选返回（type = misc）

        说明：这个通道的任务是"补漏"——通道 B 依赖固定模板，覆盖不到
        某些领域内独特的上下文。通道 C 让关键词自己暴露它的典型搭配。
        """
        if not positions:
            return {}

        klen = len(keyword)
        text_len = len(text)

        # ── 收集左右指纹 ──
        left_fp: Counter = Counter()
        right_fp: Counter = Counter()
        for pos in positions:
            for w in (2, 3):
                if pos >= w:
                    seg = text[pos - w:pos]
                    if all('\u4e00' <= c <= '\u9fff' for c in seg):
                        left_fp[seg] += 1
                end = pos + klen
                if end + w <= text_len:
                    seg = text[end:end + w]
                    if all('\u4e00' <= c <= '\u9fff' for c in seg):
                        right_fp[seg] += 1

        # 只保留出现 ≥2 次的指纹（避免单次噪音）
        left_sig = [fp for fp, c in left_fp.most_common(top_k_each_side * 2)
                    if c >= min_pattern_hits][:top_k_each_side]
        right_sig = [fp for fp, c in right_fp.most_common(top_k_each_side * 2)
                     if c >= min_pattern_hits][:top_k_each_side]

        hits: dict[str, dict] = {}
        known_vocab = known_vocab or set()

        def _add(w: str, pos: int, tmpl: str):
            w = _trim_noise(w)
            if not w:
                return
            # 候选池对齐：如果捕获串较长，优先从中取 known_vocab 内的最长子串，
            # 消除"只听张无忌"→"张无忌"这类前缀/后缀吃入问题
            if known_vocab and len(w) >= 3:
                aligned = _align_to_vocab(w, known_vocab)
                if aligned:
                    w = aligned
            if not _valid_candidate(w, self.min_len, self.max_len):
                return
            if w == keyword:
                return
            entry = hits.get(w)
            if entry is None:
                entry = {
                    'freq': 0,
                    'templates': set(),
                    'types': Counter(),
                    'evidence': [],
                    'origins': {'C'},
                }
                hits[w] = entry
            entry['freq'] += 1
            entry['templates'].add(tmpl)
            entry['origins'].add('C')
            entry['types']['misc'] += 1
            if len(entry['evidence']) < self.max_evidence:
                entry['evidence'].append((pos, tmpl))

        # ── 用指纹反向匹配 ──
        # 左指纹："fp + X"  → X 是候选
        for fp in left_sig:
            rx = re.compile(
                re.escape(fp) + rf'({_CH}{{{self.min_len},{self.max_len}}})'
                rf'{_NOT_CH_AFTER}'
            )
            for m in rx.finditer(text):
                _add(m.group(1), m.start(1), f'C:{fp}…')

        # 右指纹："X + fp" → X 是候选
        for fp in right_sig:
            rx = re.compile(
                rf'{_NOT_CH_BEFORE}({_CH}{{{self.min_len},{self.max_len}}})'
                + re.escape(fp)
            )
            for m in rx.finditer(text):
                _add(m.group(1), m.start(1), f'C:…{fp}')

        for w, e in hits.items():
            e['type'] = _resolve_type(e['types'])

        return hits


def _align_to_vocab(w: str, vocab: set[str]) -> str | None:
    """候选池对齐：若 w 含有 vocab 内的子串，返回其中**最长且右对齐优先**的那个。

    策略：从长到短枚举，对每个长度从右往左扫（因为实体通常在词组右端），
    一命中即返回。对齐后的词比原 w 更精确（消除前缀吃入）。

    如果 w 完全匹配，直接返回 w。如果没有子串命中，返回 None（调用方决定
    是否接受原 w——对"完全新词"应当接受）。
    """
    if w in vocab:
        return w
    n = len(w)
    for length in range(min(n, 6), 1, -1):
        for start in range(n - length, -1, -1):
            sub = w[start:start + length]
            if sub in vocab:
                return sub
    return None


def _resolve_type(type_counter: Counter) -> str:
    """类型投票裁定：
    - 全部是 misc → misc
    - 某个非 misc 类占比 >= 50% → 该类
    - 多个非 misc 类打平 → misc
    """
    if not type_counter:
        return 'misc'
    total = sum(type_counter.values())
    non_misc = {t: c for t, c in type_counter.items() if t != 'misc'}
    if not non_misc:
        return 'misc'
    top_t, top_c = max(non_misc.items(), key=lambda x: x[1])
    if top_c / total >= 0.5:
        return top_t
    # 否则看绝对领先
    sorted_types = sorted(non_misc.values(), reverse=True)
    if len(sorted_types) == 1 or sorted_types[0] > sorted_types[1]:
        return top_t
    return 'misc'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 证据格式化（给 extract 输出用）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SENT_BREAKS = frozenset('。！？…\n\r「」『』""')


def format_evidence_snippets(
    text: str,
    evidence: list[tuple[int, str]],
    word: str,
    max_count: int = 3,
    radius: int = 18,
) -> list[str]:
    """从证据位置列表生成可读的原文短句。

    每条返回形如："……使出[柔骨魅兔]冲向敌人……"
    [] 标注命中词，…… 表示截断。
    """
    out = []
    wlen = len(word)
    seen = set()
    for pos, tmpl in evidence[:max_count * 2]:
        if pos in seen:
            continue
        seen.add(pos)
        # 向左扩到最近的断句
        lo = max(0, pos - radius)
        for i in range(pos - 1, lo - 1, -1):
            if text[i] in _SENT_BREAKS:
                lo = i + 1
                break
        # 向右扩到最近的断句
        hi = min(len(text), pos + wlen + radius)
        for i in range(pos + wlen, hi):
            if text[i] in _SENT_BREAKS:
                hi = i
                break
        prefix = '…' if lo > 0 and text[lo - 1] not in _SENT_BREAKS else ''
        suffix = '…' if hi < len(text) and text[hi] not in _SENT_BREAKS else ''
        seg = (prefix + text[lo:pos] + '【' + word + '】'
               + text[pos + wlen:hi] + suffix)
        seg = seg.replace('\n', ' ').replace('\r', ' ')
        out.append(seg)
        if len(out) >= max_count:
            break
    return out
