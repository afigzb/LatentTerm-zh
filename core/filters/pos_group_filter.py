"""
PosGroupFilter：按词性分组做业务过滤（UI 勾选）。

依赖 PosTagger 先给每条结果打上 pos_group；本类只负责按 excluded_groups
把结果集拆成 (kept, removed) 两半。与算法打分完全解耦——改过滤策略不需要
碰 extractor，改打分策略不会影响过滤行为。

豁免规则集中在此处管理：
  • strategies == '用户输入' / '用户输入(辅)' 的行永远保留
    它们是关键词本身，过滤和打分正交——把关键词从结果里扒掉会让用户困惑。
"""


# 默认过滤分组（UI 上默认勾选的那几栏）
DEFAULT_EXCLUDED: set[str] = {
    "动词", "形容词", "副词", "代词", "数量词", "虚词", "方位时间",
}


class PosGroupFilter:
    """按 pos_group 把结果拆分为 (kept, removed)。"""

    PINNED_STRATEGIES = frozenset({'用户输入', '用户输入(辅)'})

    def filter_results(
        self,
        results: list[dict],
        excluded_groups: set[str],
    ) -> tuple[list[dict], list[dict]]:
        """返回 (kept, removed)，两者结构与输入一致。

        子术语独立过滤：父词保留时仍会按 excluded_groups 过滤它的 children。
        """
        kept, removed = [], []
        for r in results:
            if r.get('strategies') in self.PINNED_STRATEGIES:
                self._filter_children(r, excluded_groups)
                kept.append(r)
                continue
            if r.get('pos_group') in excluded_groups:
                removed.append(r)
            else:
                self._filter_children(r, excluded_groups)
                kept.append(r)
        return kept, removed

    @staticmethod
    def _filter_children(r: dict, excluded_groups: set[str]) -> None:
        children = r.get('children', [])
        if children:
            r['children'] = [
                c for c in children
                if c.get('pos_group') not in excluded_groups
            ]
