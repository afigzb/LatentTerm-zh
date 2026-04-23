"""
过滤器层：按"语义维度"而非"流水线位置"组织所有过滤规则。

现状盘点（重构前）：过滤逻辑散落在 _utils / _vocab_builder / _pattern_miner /
_strategies / term_extractor / dict_filter 六个文件里，同一件事（如"两端
不能是噪字"）被写了三遍；jieba 在两个完全不同的场景下被混称"jieba 过滤"。

本层把过滤拆成三大维度：

  linguistic  — 语言学过滤（边界噪字 / 黑名单 / 副词短语 / 量词结构）
                和语料无关，只看字面；L1 / L2 候选共用一条管道
  statistical — 统计过滤（PMI / 自由度 / 频率 / 显著性）
                目前仍就地实现在 _vocab_builder / term_extractor 中，
                接口接入本层只是未来的扩展目标
  business    — 业务过滤（词性分组）
                面向 UI 勾选的展示层过滤，算法层不感知

jieba 的两个身份也被在此处正式区分：
  • PosTagger              — 查 jieba dict.txt 的 pos 字段，纯标注
  • check_adverbial_di     — 跑 jieba.posseg，识别"副词+地+动词"结构
"""

from .linguistic import (
    Verdict,
    LingPipeline,
    DEFAULT_LING_PIPELINE,
    L1_LING_PIPELINE,
    trim_noise,
    BLACK_WORDS,
    BAD_PREFIXES,
    QUANTIFIER_UNITS,
    NUM_PREFIX_CHARS,
)
from .pos_tagger import (
    PosTagger,
    POS_GROUPS,
    ALL_GROUPS,
    UNLISTED_LABEL,
)
from .pos_group_filter import PosGroupFilter, DEFAULT_EXCLUDED


__all__ = [
    # linguistic
    'Verdict', 'LingPipeline', 'DEFAULT_LING_PIPELINE', 'L1_LING_PIPELINE',
    'trim_noise',
    'BLACK_WORDS', 'BAD_PREFIXES', 'QUANTIFIER_UNITS', 'NUM_PREFIX_CHARS',
    # pos tagging
    'PosTagger', 'POS_GROUPS', 'ALL_GROUPS', 'UNLISTED_LABEL',
    # business
    'PosGroupFilter', 'DEFAULT_EXCLUDED',
]
