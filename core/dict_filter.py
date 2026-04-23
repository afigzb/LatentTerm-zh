"""
向后兼容门面：新代码请直接用 core.filters 里的 PosTagger / PosGroupFilter。

原 DictFilter 把"词性标注（富化）"和"按分组过滤（业务动作）"耦合在一起，
app.py 即使不开过滤也要调 tag_results，配置和生命周期都混在一起。

现已拆分：
  • PosTagger       — 富化：给结果打 pos_group，无副作用
  • PosGroupFilter  — 业务过滤：按 pos_group 把结果拆 (kept, removed)

本文件保留组合类 DictFilter，委托给上述两个对象，让 app.py / benchmark.py
这类老调用点零改动即可运行。
"""

from .filters.pos_tagger import (
    PosTagger, POS_GROUPS, ALL_GROUPS, UNLISTED_LABEL,
)
from .filters.pos_group_filter import PosGroupFilter, DEFAULT_EXCLUDED


__all__ = [
    'DictFilter',
    'POS_GROUPS', 'ALL_GROUPS', 'UNLISTED_LABEL', 'DEFAULT_EXCLUDED',
]


class DictFilter:
    """薄门面：组合 PosTagger + PosGroupFilter。

    老用法继续可用：
        df = DictFilter()
        df.tag_results(results)
        kept, removed = df.filter_results(results, excluded)
    """

    def __init__(self):
        self._tagger = PosTagger()
        self._group_filter = PosGroupFilter()

    @property
    def _word_pos(self):
        """向后兼容：旧 benchmark 里读 dict_filter._word_pos 看词条数。"""
        return self._tagger._word_pos

    def tag_word(self, word: str) -> str:
        return self._tagger.tag_word(word)

    def tag_results(self, results: list[dict]) -> list[dict]:
        return self._tagger.tag_results(results)

    def filter_results(
        self,
        results: list[dict],
        excluded_groups: set[str],
    ) -> tuple[list[dict], list[dict]]:
        return self._group_filter.filter_results(results, excluded_groups)
