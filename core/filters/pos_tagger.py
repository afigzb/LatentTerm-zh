"""
PosTagger：基于 jieba 内置词典 dict.txt 的词性标注（纯富化，无副作用）。

只查表、不跑分词引擎。和算法层完全解耦——任何结果集传进来都能拿到
pos_group 字段。原 core.dict_filter.DictFilter 的"标注"职责从此处独立出来，
与"按分组过滤"（PosGroupFilter）物理分离。
"""

import os


# 词性分组：jieba 细粒度 pos 码 → 用户友好的中文分组名
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

ALL_GROUPS = list(POS_GROUPS.keys())

UNLISTED_LABEL = "未收录"

_POS_TO_GROUP: dict[str, str] = {}
for _group_name, _codes in POS_GROUPS.items():
    for _code in _codes:
        _POS_TO_GROUP[_code] = _group_name


class PosTagger:
    """加载 jieba dict.txt，提供 word → 词性分组名 的 O(1) 查询。"""

    def __init__(self):
        self._word_pos: dict[str, str] = {}
        self._load_jieba_dict()

    def _load_jieba_dict(self):
        import jieba
        dict_path = os.path.join(jieba.__path__[0], "dict.txt")
        word_pos: dict[str, str] = {}
        with open(dict_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) >= 3:
                    word_pos[parts[0]] = parts[2]
                elif len(parts) == 2:
                    word_pos[parts[0]] = ""
        self._word_pos = word_pos

    def tag_word(self, word: str) -> str:
        """返回词的用户友好分组名称；不在词典中返回 '未收录'。"""
        pos = self._word_pos.get(word)
        if pos is None:
            return UNLISTED_LABEL
        return _POS_TO_GROUP.get(pos, UNLISTED_LABEL)

    def tag_results(self, results: list[dict]) -> list[dict]:
        """给结果列表中每个词（含子术语）打上 pos_group 字段。就地修改。"""
        for r in results:
            r["pos_group"] = self.tag_word(r["word"])
            for child in r.get("children", []):
                child["pos_group"] = self.tag_word(child["word"])
        return results
