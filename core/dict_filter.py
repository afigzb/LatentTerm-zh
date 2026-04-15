"""
词典过滤器：基于 jieba 内置词典的词性标注，对提取结果做后置过滤。

只读取 jieba 的 dict.txt 做查表，不调用分词引擎。
"""

import os

POS_GROUPS: dict[str, set[str]] = {
    "动词":     {"v", "vd", "vn", "vg"},
    "形容词":   {"a", "ad", "an", "ag"},
    "副词":     {"d", "dg"},
    "代词":     {"r", "rg"},
    "数量词":   {"m", "mg", "q"},
    "虚词":     {"u", "p", "c", "xc", "e", "y", "o", "h", "k"},
    "方位时间": {"f", "s", "t", "tg"},
    "通用名词": {"n"},
    "专有名词": {"nr", "ns", "nt", "nz", "nrt"},
    "成语习语": {"i", "l"},
}

DEFAULT_EXCLUDED = {"动词", "形容词", "副词", "代词", "数量词", "虚词", "方位时间"}

ALL_GROUPS = list(POS_GROUPS.keys())

_POS_TO_GROUP: dict[str, str] = {}
for group_name, codes in POS_GROUPS.items():
    for code in codes:
        _POS_TO_GROUP[code] = group_name

UNLISTED_LABEL = "未收录"


class DictFilter:

    def __init__(self):
        self._word_pos: dict[str, str] = {}
        self._load_jieba_dict()

    def _load_jieba_dict(self):
        import jieba
        dict_path = os.path.join(jieba.__path__[0], "dict.txt")
        word_pos = {}
        with open(dict_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) >= 3:
                    word, _, pos = parts[0], parts[1], parts[2]
                    word_pos[word] = pos
                elif len(parts) == 2:
                    word_pos[parts[0]] = ""
        self._word_pos = word_pos

    def tag_word(self, word: str) -> str:
        """返回词的用户友好分组名称，不在词典中返回 '未收录'。"""
        pos = self._word_pos.get(word)
        if pos is None:
            return UNLISTED_LABEL
        return _POS_TO_GROUP.get(pos, UNLISTED_LABEL)

    def tag_results(self, results: list[dict]) -> list[dict]:
        """给结果列表中每个词（含子术语）打上词性标签。"""
        for r in results:
            r["pos_group"] = self.tag_word(r["word"])
            for child in r.get("children", []):
                child["pos_group"] = self.tag_word(child["word"])
        return results

    def filter_results(
        self,
        results: list[dict],
        excluded_groups: set[str],
    ) -> tuple[list[dict], list[dict]]:
        """根据要过滤的词性分组拆分结果。

        返回 (kept, removed)，两者结构与输入一致。
        用户输入的关键词行（strategies == '用户输入'）永远保留。
        """
        kept, removed = [], []
        for r in results:
            if r.get("strategies") == "用户输入":
                self._filter_children(r, excluded_groups)
                kept.append(r)
                continue
            if r["pos_group"] in excluded_groups:
                removed.append(r)
            else:
                self._filter_children(r, excluded_groups)
                kept.append(r)
        return kept, removed

    @staticmethod
    def _filter_children(r: dict, excluded_groups: set[str]):
        children = r.get("children", [])
        if children:
            r["children"] = [
                c for c in children if c.get("pos_group") not in excluded_groups
            ]
