import streamlit as st
import pandas as pd
from core.text_cleaner import clean_text
from core.term_extractor import TermExtractor, DEFAULT_CONFIG
from core.dict_filter import DictFilter

# 类型标签中文化（与 core/_pattern_miner.py 的类型体系一致）
_TYPE_ZH = {
    'person':   '人物',
    'place':    '地点',
    'creature': '生物',
    'skill':    '技·物',
    'group':    '组织',
    'misc':     '其他',
    '':         '',
}

st.set_page_config(page_title="AIWA 文本调试台", layout="wide")
st.title("AIWA 2.0 - 文本调试台")

# ── 文件读取设置 ──────────────────────────────────────────────────────────
_ENC_OPTIONS = {
    "auto":    "自动检测（charset_normalizer）",
    "utf-8":   "UTF-8",
    "gb18030": "GB18030（网络小说大概率是这种格式）",
    "gbk":     "GBK",
    "big5":    "Big5（繁体中文）",
    "utf-16":  "UTF-16",
    "latin-1": "Latin-1",
    "custom":  "自定义…",
}

with st.expander("⚙️ 文件读取设置", expanded=True):
    col_enc, col_cln = st.columns([3, 2])
    with col_enc:
        enc_key = st.selectbox(
            "文件编码",
            list(_ENC_OPTIONS.keys()),
            format_func=lambda k: _ENC_OPTIONS[k],
            help="不确定时选「自动检测」；绝大多数国产网文选 GB18030 即可",
        )
        custom_enc = ""
        if enc_key == "custom":
            custom_enc = st.text_input(
                "自定义编码名称",
                placeholder="例：gb2312、shift_jis、euc-kr",
            )
    with col_cln:
        enable_cleaning = st.checkbox(
            "启用文本清洗",
            value=False,
            help=(
                "勾选后执行两步清洗：\n"
                "① 去除零宽字符、控制字符等不可见脏字符\n"
                "② Unicode NFKC 标准化（全角→半角等）"
            ),
        )

encoding_to_use = (
    None        if enc_key == "auto"
    else custom_enc if enc_key == "custom"
    else enc_key
)

# ── 文件上传（全局，两个标签页共用同一份文件）──────────────────────────
uploaded_file = st.file_uploader("上传 TXT 小说文件（所有功能共用此文件）", type="txt")

if uploaded_file is not None:
    file_id = (
        uploaded_file.name
        + str(uploaded_file.size)
        + (encoding_to_use or "auto")
        + str(enable_cleaning)
    )

    if st.session_state.get("file_id") != file_id:
        with st.spinner("正在建立词表……"):
            raw_bytes = uploaded_file.read()
            cleaned_text, raw_text, encoding, stats = clean_text(
                raw_bytes,
                encoding=encoding_to_use,
                enable_cleaning=enable_cleaning,
            )

            if cleaned_text is None:
                st.error(f"解析失败！{raw_text}")
                st.stop()

            extractor = TermExtractor(min_len=2, max_len=8)
            extractor.build_index(cleaned_text)

            if "dict_filter" not in st.session_state:
                st.session_state["dict_filter"] = DictFilter()

            st.session_state["file_id"] = file_id
            st.session_state["cleaned_text"] = cleaned_text
            st.session_state["raw_text"] = raw_text
            st.session_state["encoding"] = encoding
            st.session_state["stats"] = stats
            st.session_state["extractor"] = extractor

    cleaned_text = st.session_state["cleaned_text"]
    raw_text     = st.session_state["raw_text"]
    encoding     = st.session_state["encoding"]
    stats        = st.session_state["stats"]
    extractor    = st.session_state["extractor"]

    st.success(f"文件已加载：{uploaded_file.name}（编码：{encoding}，共 {len(cleaned_text):,} 字符）")

    # ── 两个功能标签页 ────────────────────────────────────────────────────
    tab_clean, tab_extract = st.tabs(["📄 文本清洗", "🔍 术语提取"])

    # ════════════════════════════════════════════════════════════════════
    # 标签页一：文本清洗
    # ════════════════════════════════════════════════════════════════════
    with tab_clean:
        st.subheader("清洗报告")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("检测到编码",   stats["detected_encoding"])
        c2.metric("原始字符数",   f'{stats["original_length"]:,}')
        c3.metric("清除脏字符数", stats["dirty_chars_removed"])
        c4.metric("清洗后字符数", f'{stats["final_length"]:,}')

        st.divider()

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("原始文本（前 1000 字）")
            st.text_area("raw", raw_text[:1000], height=400,
                         disabled=True, label_visibility="collapsed")
        with col_r:
            st.subheader("清洗后文本（前 1000 字）")
            st.text_area("clean", cleaned_text[:1000], height=400,
                         disabled=True, label_visibility="collapsed")

    # ════════════════════════════════════════════════════════════════════
    # 标签页二：术语提取
    # ════════════════════════════════════════════════════════════════════
    with tab_extract:
        st.subheader("关键词术语聚合")

        cfg = DEFAULT_CONFIG

        col_kw, col_aux, col_min, col_max, col_top = st.columns([2, 2, 1, 1, 1])
        with col_kw:
            keyword = st.text_input("关键词", placeholder="例：霍雨浩、丹药")
        with col_aux:
            aux_keyword = st.text_input(
                "辅关键词（可选）", placeholder="例：魂环、武魂",
                help="填入后启用「双关键词联合模式」，只输出在原文中"
                     "和两个关键词都强相关（共现）的术语",
            )
        with col_min:
            min_freq = st.slider(
                "最低出现频次", min_value=2, max_value=2000,
                value=cfg['min_freq'], key="min_freq",
                help="低于此频次的候选词直接过滤",
            )
        with col_max:
            max_freq = st.number_input(
                "最高出现频次", min_value=0, value=cfg['max_freq'], step=10,
                key="max_freq",
                help="0 = 不限；设为正数后可排除出现过多的通用高频词",
            )
        with col_top:
            top_n = st.slider(
                "最多展示条数", min_value=10, max_value=10000,
                value=cfg['top_n'], key="top_n",
            )

        with st.expander("策略权重调节", expanded=False):
            st.caption("调节六种提取策略的权重，权重越高该策略对排序的影响越大。")
            wr1, wr2, wr3 = st.columns(3)
            with wr1:
                w_char = st.slider("字符包含", 0.0, 1.0, cfg['w_char'], 0.05,
                                   key="w_char")
            with wr2:
                w_context = st.slider("上下文模式", 0.0, 1.0, cfg['w_context'], 0.05,
                                      key="w_context")
            with wr3:
                w_cooccur = st.slider("共现近邻", 0.0, 1.0, cfg['w_cooccur'], 0.05,
                                      key="w_cooccur")
            wr4, wr5, wr6 = st.columns(3)
            with wr4:
                w_morph = st.slider("构词结构", 0.0, 1.0, cfg['w_morph'], 0.05,
                                    key="w_morph")
            with wr5:
                w_subst = st.slider("互替性", 0.0, 1.0, cfg['w_subst'], 0.05,
                                    key="w_subst")
            with wr6:
                w_topic = st.slider("段落共主题", 0.0, 1.0, cfg['w_topic'], 0.05,
                                    key="w_topic")

        if keyword:
            aux_kw = (aux_keyword or '').strip()
            if aux_kw and aux_kw == keyword.strip():
                aux_kw = ''
            spin_msg = (f"正在提取与「{keyword}」+「{aux_kw}」联合相关的术语……"
                        if aux_kw
                        else f"正在提取与「{keyword}」相关的术语……")
            with st.spinner(spin_msg):
                results = extractor.extract(
                    keyword, top_n=top_n, min_freq=min_freq,
                    w_char=w_char, w_context=w_context,
                    w_cooccur=w_cooccur, w_morph=w_morph,
                    w_subst=w_subst, w_topic=w_topic,
                    max_freq=max_freq,
                    aux_keyword=aux_kw,
                )

            if not results:
                if aux_kw:
                    st.warning(
                        f"没有找到与「{keyword}」+「{aux_kw}」联合相关且"
                        f"频次 ≥ {min_freq} 的术语，可能两个关键词在原文中"
                        "共现过少。请放宽频次门槛、更换关键词，或改用单关键词。"
                    )
                else:
                    st.warning(
                        f"没有找到与「{keyword}」相关且频次 ≥ {min_freq} 的术语，"
                        "请尝试降低最低频次或更换关键词。"
                    )
            else:
                dict_filter: DictFilter = st.session_state["dict_filter"]
                dict_filter.tag_results(results)

                pinned_strats = {'用户输入', '用户输入(辅)'}
                pinned_list = []
                idx = 0
                while (idx < len(results)
                       and results[idx].get('strategies') in pinned_strats):
                    pinned_list.append(results[idx])
                    idx += 1
                algo_results = results[idx:]
                pinned = pinned_list[0] if pinned_list else None

                for p in pinned_list:
                    p_children = p.get('children', [])
                    suffix = (f"，含 **{len(p_children)}** 个从属子术语"
                              if p_children else "")
                    role = ("**辅关键词**"
                            if p.get('strategies') == '用户输入(辅)'
                            else "**直接命中**")
                    st.info(
                        f"📌 {role}　关键词「**{p['word']}**」"
                        f"在全文中共出现 **{p['freq']:,}** 次{suffix}"
                    )

                algo_count = sum(
                    1 + len(r.get('children', []))
                    for r in algo_results
                )
                st.success(f"算法共找到 {algo_count} 个关联候选术语"
                           f"（{len(algo_results)} 个主词 + "
                           f"{algo_count - len(algo_results)} 个从属子术语）")

                # ── 词典过滤控件 ──
                with st.expander("📖 词典过滤", expanded=False):
                    enable_filter = st.checkbox(
                        "启用词典过滤",
                        value=False,
                        help="基于 jieba 词典的词性标注，过滤通用词汇，保留领域专有术语",
                    )
                    if enable_filter:
                        st.caption(
                            "勾选要**过滤掉**的词性分组（不在词典中的词标记为「未收录」，"
                            "通常是领域专有术语）"
                        )
                        fc1, fc2, fc3, fc4 = st.columns(4)
                        with fc1:
                            f_verb = st.checkbox("动词", value=True, key="f_verb")
                            f_adj  = st.checkbox("形容词", value=True, key="f_adj")
                            f_adv  = st.checkbox("副词", value=True, key="f_adv")
                        with fc2:
                            f_pron = st.checkbox("代词", value=True, key="f_pron")
                            f_num  = st.checkbox("数量词", value=True, key="f_num")
                            f_func = st.checkbox("虚词", value=True, key="f_func")
                        with fc3:
                            f_loc  = st.checkbox("方位时间", value=True, key="f_loc")
                            f_noun = st.checkbox(
                                "通用名词", value=False, key="f_noun",
                                help="开启后过滤力度较大，常见名词也会被移除",
                            )
                        with fc4:
                            f_proper = st.checkbox("专有名词", value=False, key="f_proper")
                            f_idiom  = st.checkbox("成语习语", value=False, key="f_idiom")

                        excluded = set()
                        if f_verb:   excluded.add("动词")
                        if f_adj:    excluded.add("形容词")
                        if f_adv:    excluded.add("副词")
                        if f_pron:   excluded.add("代词")
                        if f_num:    excluded.add("数量词")
                        if f_func:   excluded.add("虚词")
                        if f_loc:    excluded.add("方位时间")
                        if f_noun:   excluded.add("通用名词")
                        if f_proper: excluded.add("专有名词")
                        if f_idiom:  excluded.add("成语习语")
                    else:
                        excluded = set()

                if enable_filter and excluded:
                    display_results, removed_results = dict_filter.filter_results(
                        results, excluded)
                else:
                    display_results = results
                    removed_results = []

                if display_results:
                    flat_rows = []
                    for r in display_results:
                        flat_rows.append({
                            '词语': r['word'],
                            '层': r.get('tier', ''),
                            '类型': _TYPE_ZH.get(r.get('type', ''),
                                                r.get('type', '')),
                            '词性': r.get('pos_group', ''),
                            '出现频次': r['freq'],
                            '综合评分': r['score'],
                            '命中策略': r['strategies'],
                            '命中数': r['hit_count'],
                            '命中模板': r.get('templates', ''),
                            '匹配模式': r['matched_patterns'],
                            '原文证据': r.get('evidence', ''),
                            '子术语': '、'.join(
                                c['word'] for c in r.get('children', [])
                            ),
                        })

                    df = pd.DataFrame(flat_rows)
                    df.index = df.index + 1

                    st.dataframe(
                        df.style
                          .background_gradient(subset=["综合评分"], cmap="Blues")
                          .background_gradient(subset=["出现频次"], cmap="Greens")
                          .background_gradient(subset=["命中数"], cmap="Oranges")
                          .format({"综合评分": "{:.4f}"}),
                        use_container_width=True,
                        height=600,
                    )

                    has_children = any(
                        r.get('children') for r in display_results)
                    if has_children:
                        with st.expander("展开查看子术语详情"):
                            for r in display_results:
                                children = r.get('children', [])
                                if not children:
                                    continue
                                st.markdown(f"**{r['word']}** 的从属子术语：")
                                child_df = pd.DataFrame([
                                    {
                                        '子术语': c['word'],
                                        '层': c.get('tier', ''),
                                        '类型': _TYPE_ZH.get(
                                            c.get('type', ''),
                                            c.get('type', '')),
                                        '词性': c.get('pos_group', ''),
                                        '出现频次': c['freq'],
                                        '综合评分': c['score'],
                                        '命中策略': c['strategies'],
                                        '命中模板': c.get('templates', ''),
                                        '原文证据': c.get('evidence', ''),
                                    }
                                    for c in children
                                ])
                                child_df.index = child_df.index + 1
                                st.dataframe(child_df, use_container_width=True)

                    if removed_results:
                        with st.expander(
                            f"已过滤的词（{len(removed_results)} 个）"
                        ):
                            removed_rows = []
                            for r in removed_results:
                                removed_rows.append({
                                    '词语': r['word'],
                                    '词性': r.get('pos_group', ''),
                                    '出现频次': r['freq'],
                                    '综合评分': r['score'],
                                    '命中策略': r['strategies'],
                                })
                            removed_df = pd.DataFrame(removed_rows)
                            removed_df.index = removed_df.index + 1
                            st.dataframe(removed_df, use_container_width=True)

                    st.caption(
                        "**层**：L1=统计主干（PMI+自由度通过），"
                        "L2=模板孤岛（对白/命名/量词等高置信模板命中，含低频专名）　｜　"
                        "**类型**：来自模板命中的粗分类，用于类型先验加权　｜　"
                        "**词性**：基于 jieba 词典标注，「未收录」通常是领域专有术语　｜　"
                        "**命中策略**：字符=字符包含、上下文=上下文模式引导、"
                        "共现=共现近邻、构词=构词结构相似、互替=上下文互替性、"
                        "共主题=段落级共现　｜　"
                        "**命中数**：被几种策略同时发现（越多越可信）　｜　"
                        "**命中模板**：通道 B/C 的高精度模板 id　｜　"
                        "**匹配模式**：上下文策略命中的特征　｜　"
                        "**原文证据**：【】内为命中词，截取附近原文短句　｜　"
                        "**子术语**：包含该主词的更长术语（从属关系）"
                    )
        else:
            st.info("请在上方输入关键词，开始提取。")

else:
    st.info("请先上传一个 TXT 文件，所有功能将在此文件上运行。")
