"""
三阶段级联中文术语提取器

阶段一  build_index  — 全文词汇发现（与关键词无关，只需执行一次）
                       实现见 _vocab_builder.py → VocabBuilderMixin
阶段二  _strategy_*  — 六策略并行候选提取
                       字符包含 / 上下文 / 共现 / 构词 / 互替 / 段落共主题
                       实现见 _strategies.py  → StrategiesMixin
阶段三  extract      — 多信号归一化融合 + 交叉验证加分 + 后处理（本文件）
"""

from collections import Counter, defaultdict

from ._utils import _normalize, _LEFT_NOISE, _RIGHT_NOISE
from ._vocab_builder import VocabBuilderMixin
from ._strategies import StrategiesMixin


# ── 模式预设（UI 层和算法层共用）────────────────────────────────────────
MODE_PRESETS = {
    'balanced': {
        'label': '均衡模式',
        'desc': '默认策略，适合大多数关键词',
        'min_freq': 3, 'min_freq_range': (2, 50),
        'max_freq': 0,
        'top_n': 75,
        'w_char': 0.25, 'w_context': 0.40, 'w_cooccur': 0.40,
        'w_morph': 0.15, 'w_subst': 0.30, 'w_topic': 0.25,
    },
    'high_freq': {
        'label': '高频模式（≥800次）',
        'desc': '只保留出现 ≥800 次的高频术语',
        'min_freq': 800, 'min_freq_range': (100, 2000),
        'max_freq': 0,
        'top_n': 75,
        'w_char': 0.25, 'w_context': 0.40, 'w_cooccur': 0.40,
        'w_morph': 0.15, 'w_subst': 0.30, 'w_topic': 0.25,
    },
    'low_freq': {
        'label': '低频模式（≤20次）',
        'desc': '挖掘稀有术语，两跳发散 + 排除高频词',
        'min_freq': 2, 'min_freq_range': (2, 10),
        'max_freq': 100,
        'top_n': 150,
        'w_char': 0.15, 'w_context': 0.5, 'w_cooccur': 0.4,
        'w_morph': 0.1, 'w_subst': 0.25, 'w_topic': 0.30,
    },
}


class TermExtractor(VocabBuilderMixin, StrategiesMixin):

    # ── 集中管理的阈值与参数 ──────────────────────────────────────────────

    # 词表构建（build_index 使用）
    VOCAB_COH_STRICT = 0.08
    VOCAB_FREE_STRICT = 0.1
    VOCAB_COH_RELAXED = 0.03
    VOCAB_FREE_RELAXED = 0.03

    # 伪合成词过滤（_filter_extensions 使用）
    FILTER_COH_STRICT = 0.1
    FILTER_COH_LENIENT = 0.05
    FILTER_DOM_STRICT = 8
    FILTER_DOM_LENIENT = 20

    # 评分融合（extract 使用）
    MISS_PENALTY = -0.1
    CROSS_BONUS = 0.1

    # 策略内部衰减
    CONTEXT_DECAY = 0.7
    HOP2_DECAY = 0.7

    def __init__(self, min_len: int = 2, max_len: int = 8):
        self.min_len = min_len
        self.max_len = max_len
        self._text: str = ''
        self._text_len: int = 0
        self._freq: Counter = Counter()
        self._left_nb: defaultdict = defaultdict(Counter)
        self._right_nb: defaultdict = defaultdict(Counter)
        self._vocab: dict = {}
        self._vocab_relaxed: dict = {}
        self._built: bool = False

    # ------------------------------------------------------------------ #
    # 词表选择                                                              #
    # ------------------------------------------------------------------ #

    def _active_vocab(self, mode: str) -> dict:
        """按模式返回对应词表，取代原先 extract() 中的临时替换。"""
        return self._vocab_relaxed if mode == 'low_freq' else self._vocab

    # ------------------------------------------------------------------ #
    # 阶段三：多信号融合排序                                                  #
    # ------------------------------------------------------------------ #

    def extract(self, keyword: str, top_n: int = 75, min_freq: int = 2,
                w_char: float = 0.25, w_context: float = 0.40,
                w_cooccur: float = 0.40, w_morph: float = 0.15,
                w_subst: float = 0.30, w_topic: float = 0.25,
                mode: str = 'balanced', max_freq: int = 0) -> list[dict]:
        if not self._built:
            raise RuntimeError('请先调用 build_index(text)')

        vocab = self._active_vocab(mode)
        raw_scores, pattern_info = self._run_strategies(keyword, mode, vocab)

        n_char = _normalize(raw_scores['char_overlap'])
        n_context = _normalize(raw_scores['context_pattern'])
        n_cooccur = _normalize(raw_scores['cooccurrence'])
        n_morph = _normalize(raw_scores['morpheme'])
        n_subst = _normalize(raw_scores['substitution'])
        n_topic = _normalize(raw_scores['co_topic'])
        all_words = (set(n_char) | set(n_context) | set(n_cooccur)
                     | set(n_morph) | set(n_subst) | set(n_topic))

        miss = self.MISS_PENALTY
        results = []
        for word in all_words:
            info = vocab.get(word)
            if not info or info['freq'] < min_freq:
                continue
            if max_freq > 0 and info['freq'] > max_freq:
                continue

            s_char = n_char.get(word, miss)
            s_context = n_context.get(word, miss)
            s_cooccur = n_cooccur.get(word, miss)
            s_morph = n_morph.get(word, miss)
            s_subst = n_subst.get(word, miss)
            s_topic = n_topic.get(word, miss)

            score = (w_char * s_char + w_context * s_context
                     + w_cooccur * s_cooccur + w_morph * s_morph
                     + w_subst * s_subst + w_topic * s_topic)
            hits = sum(1 for s in (s_char, s_context, s_cooccur,
                                   s_morph, s_subst, s_topic) if s > 0)
            score += self.CROSS_BONUS * max(0, hits - 1)

            labels = []
            if s_char > 0:
                labels.append('字符')
            if s_context > 0:
                labels.append('上下文')
            if s_cooccur > 0:
                labels.append('共现')
            if s_morph > 0:
                labels.append('构词')
            if s_subst > 0:
                labels.append('互替')
            if s_topic > 0:
                labels.append('共主题')

            results.append({
                'word': word,
                'freq': info['freq'],
                'score': round(score, 4),
                'strategies': '、'.join(labels),
                'hit_count': hits,
                'matched_patterns': '  '.join(
                    pattern_info.get(word, [])),
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        results = [r for r in results if r['word'] != keyword][:top_n]

        kw_freq = self._freq.get(keyword)
        if kw_freq is None:
            kw_freq = self._text.count(keyword)
        if kw_freq > 0:
            kw_entry = {
                'word': keyword,
                'freq': kw_freq,
                'score': 9999.0,
                'strategies': '用户输入',
                'hit_count': 6,
                'matched_patterns': '',
            }
            results = [kw_entry] + results

        return self._group_by_parent(results)

    # ------------------------------------------------------------------ #
    # 策略调度：根据模式决定策略执行流程                                          #
    # ------------------------------------------------------------------ #

    def _run_strategies(self, keyword: str, mode: str, vocab: dict):
        """返回 (raw_scores_dict, pattern_info)

        raw_scores_dict 的 key 固定为六个策略名：
          char_overlap / context_pattern / cooccurrence /
          morpheme / substitution / co_topic
        """
        res_char = self._strategy_char_overlap(keyword, vocab)
        res_morph = self._strategy_morpheme(keyword, vocab)

        if mode == 'low_freq':
            return self._run_low_freq(
                keyword, vocab, res_char.scores, res_morph.scores)
        return self._run_balanced(
            keyword, vocab, res_char.scores, res_morph.scores)

    def _run_balanced(self, keyword, vocab, raw_char, raw_morph):
        """均衡 / 高频模式：单跳 + 小幅扩展"""
        if self._find_all(keyword):
            seeds = [keyword]
        else:
            combined = {
                w: raw_char.get(w, 0) + raw_morph.get(w, 0)
                for w in set(raw_char) | set(raw_morph)
            }
            seeds = sorted(combined, key=combined.get, reverse=True)[:5]

        if seeds:
            res_ctx = self._strategy_context_pattern(seeds, vocab)
            raw_context = res_ctx.scores
            pattern_info = res_ctx.meta.get('patterns', {})
            raw_cooccur = self._strategy_cooccurrence(seeds, vocab).scores
            raw_subst = self._strategy_substitution(seeds, vocab).scores
            raw_topic = self._strategy_co_topic(seeds, vocab).scores
        else:
            raw_context, pattern_info = {}, {}
            raw_cooccur, raw_subst, raw_topic = {}, {}, {}

        return {
            'char_overlap': raw_char,
            'context_pattern': raw_context,
            'cooccurrence': raw_cooccur,
            'morpheme': raw_morph,
            'substitution': raw_subst,
            'co_topic': raw_topic,
        }, pattern_info

    def _run_low_freq(self, keyword, vocab, raw_char, raw_morph):
        """低频模式：两跳发散——先结构匹配，再用候选词做种子上下文扩展"""
        if self._find_all(keyword):
            seeds = [keyword]
            res_ctx = self._strategy_context_pattern(seeds, vocab)
            raw_context = res_ctx.scores
            pattern_info = res_ctx.meta.get('patterns', {})
            raw_cooccur = self._strategy_cooccurrence(seeds, vocab).scores
            raw_subst = self._strategy_substitution(seeds, vocab).scores
            raw_topic = self._strategy_co_topic(seeds, vocab).scores
        else:
            raw_context, pattern_info = {}, {}
            raw_cooccur, raw_subst, raw_topic = {}, {}, {}

        hop1 = {}
        for w in set(raw_char) | set(raw_morph):
            hop1[w] = raw_char.get(w, 0) + raw_morph.get(w, 0)

        hop1_seeds = [
            w for w in sorted(hop1, key=hop1.get, reverse=True)
            if w != keyword and self._freq.get(w, 0) >= 2
        ][:5]

        if hop1_seeds:
            hop2_ctx = self._strategy_context_pattern(
                hop1_seeds, vocab, _max_scan=200)
            hop2_cooccur = self._strategy_cooccurrence(
                hop1_seeds, vocab, _max_occ=100)
            hop2_subst = self._strategy_substitution(
                hop1_seeds, vocab, _max_scan=200)
            hop2_topic = self._strategy_co_topic(hop1_seeds, vocab)

            decay = self.HOP2_DECAY
            for w, s in hop2_ctx.scores.items():
                raw_context[w] = max(raw_context.get(w, 0), s * decay)
                hop2_patterns = hop2_ctx.meta.get('patterns', {})
                if w in hop2_patterns:
                    existing = set(pattern_info.get(w, []))
                    pattern_info[w] = sorted(
                        existing | set(hop2_patterns[w]))
            for w, s in hop2_cooccur.scores.items():
                raw_cooccur[w] = max(raw_cooccur.get(w, 0), s * decay)
            for w, s in hop2_subst.scores.items():
                raw_subst[w] = max(raw_subst.get(w, 0), s * decay)
            for w, s in hop2_topic.scores.items():
                raw_topic[w] = max(raw_topic.get(w, 0), s * decay)

        return {
            'char_overlap': raw_char,
            'context_pattern': raw_context,
            'cooccurrence': raw_cooccur,
            'morpheme': raw_morph,
            'substitution': raw_subst,
            'co_topic': raw_topic,
        }, pattern_info

    @staticmethod
    def _group_by_parent(results: list[dict], noise=None) -> list[dict]:
        """按包含关系分组，用噪声字判定谁是真正的父词。

        short ⊂ long 时：
          extra 全是噪声字 → short 是真术语，long 挂到 short 下面
          extra 不全是噪声字 → long 是复合术语，short 挂到 long 下面
        """
        if noise is None:
            noise = _LEFT_NOISE | _RIGHT_NOISE
        words = [r['word'] for r in results]
        n = len(words)
        child_of: dict[int, int] = {}

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                wi, wj = words[i], words[j]
                if len(wi) == len(wj) or wi not in wj:
                    continue

                short_idx, long_idx = (i, j) if len(wi) < len(wj) else (j, i)
                short_w, long_w = words[short_idx], words[long_idx]

                extra = long_w.replace(short_w, '', 1)
                if all(ch in noise for ch in extra):
                    parent_idx, child_idx = short_idx, long_idx
                else:
                    parent_idx, child_idx = long_idx, short_idx

                if child_idx not in child_of:
                    child_of[child_idx] = parent_idx

        for r in results:
            r.setdefault('children', [])

        for ci, pi in child_of.items():
            if pi in child_of:
                continue
            results[pi]['children'].append(results[ci])

        return [r for i, r in enumerate(results) if i not in child_of]
