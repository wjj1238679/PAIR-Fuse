# -*- coding: utf-8 -*-
"""E3b: 无注入的干净 bigram 检索器 —— 诚实的部署态候选覆盖率
对每条 test 查询: 用已知端实体描述+关系文本做 query, 对全部实体描述做 char-bigram
相似度检索, 取 top-K, 统计 gold 覆盖率 (不注入任何 gold)。
同时输出: 部署态下界指标(把现有融合排名中 gold 不在干净 top-K 的查询计为 miss)。
用法: python clean_retriever_coverage.py
"""
import json, os, re, unicodedata
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "dataset", "qt_cpp_kg")
K = 80  # 与论文 tail 候选规模一致

def norm(s):
    s = unicodedata.normalize("NFKC", str(s or ""))
    return "".join(s.split()).lower()

def bigrams(s):
    s = norm(s)
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i+2] for i in range(len(s)-1)}

def load_map(path, key_col=0, val_col=2, sep="\t"):
    m = {}
    with open(path, encoding="utf-8-sig", errors="ignore") as f:
        for ln in f:
            sp = ln.rstrip("\n").split(sep)
            if len(sp) > max(key_col, val_col):
                m[sp[key_col].strip()] = sp[val_col].strip()
    return m

def main():
    ent_text = load_map(os.path.join(DS, "entity2text.txt"))       # name -> desc
    rel_text = load_map(os.path.join(DS, "relation2text.txt"), 0, 1)
    # 预计算实体 bigram
    ents = list(ent_text.keys())
    ent_bg = {e: bigrams(ent_text[e]) for e in ents}
    # 名字本身也作为一个弱信号加入候选侧(模拟名称-描述混合检索)
    for e in ents:
        ent_bg[e] |= bigrams(e)

    triples = []
    with open(os.path.join(DS, "test.txt"), encoding="utf-8-sig") as f:
        for ln in f:
            sp = ln.rstrip("\n").split("\t")
            if len(sp) == 3:
                triples.append(tuple(sp))

    res = {}
    gold_rank = {}  # (task, idx) -> gold 在干净检索中的名次(1-based), None 表示不在 top-K
    for task in ["tail", "head"]:
        cov = 0
        ranks = []
        for i, (h, r, t) in enumerate(triples):
            known, gold = (h, t) if task == "tail" else (t, h)
            qtext = ent_text.get(known, known) + " " + rel_text.get(r, r.replace("/", " "))
            qbg = bigrams(qtext) | bigrams(known)
            scored = []
            for e in ents:
                inter = len(qbg & ent_bg[e])
                if inter:
                    scored.append((2.0 * inter / (len(qbg) + len(ent_bg[e])), e))
            scored.sort(key=lambda x: -x[0])
            top = [e for _, e in scored[:K]]
            g = norm(gold)
            topn = [norm(e) for e in top]
            rk = topn.index(g) + 1 if g in topn else None
            gold_rank[(task, i)] = rk
            ranks.append(rk)
            if rk is not None:
                cov += 1
        n = len(triples)
        res[task] = {
            "N": n, "K": K,
            "clean_retriever_coverage": round(cov / n, 4),
            "gold_rank_median_when_covered": sorted([x for x in ranks if x])[max(0, cov // 2)] if cov else None,
        }
        print(task, res[task])

    with open("clean_retriever_goldrank.json", "w", encoding="utf-8") as f:
        json.dump({f"{t}|{i}": r for (t, i), r in gold_rank.items()}, f)

    # ---- 部署态下界: gold 不在干净 top-K -> 该查询必然 miss ----
    def load_pred(path):
        with open(path, encoding="utf-8-sig") as f:
            return [json.loads(l) for l in f if l.strip()]

    preds = {
        ("head", "LLM-only (P1)"): "outputs/qt_cpp_kg/head/output_head.chat.txt",
        ("head", "PAIR-Fuse (Safe-RRF)"): "outputs/qt_cpp_kg/head/head_fuse_safe_tie.txt",
        ("tail", "LLM-only (P1)"): "outputs/qt_cpp_kg/tail/output_tail.chat.txt",
        ("tail", "PAIR-Fuse (Safe-RRF)"): "outputs/qt_cpp_kg/tail/tail_fuse_safe_tie.txt",
        ("tail", "Linear fusion"): "outputs/qt_cpp_kg/ablation/tail/tail_fuse_linear.txt",
        ("head", "Linear fusion"): "outputs/qt_cpp_kg/ablation/head/head_fuse_linear.txt",
    }
    deploy = {}
    for (task, name), path in preds.items():
        if not os.path.exists(os.path.join(HERE, path)):
            continue
        P = load_pred(os.path.join(HERE, path))
        hits1 = hits10 = 0
        n = len(triples)
        for i, (h, r, t) in enumerate(triples):
            gold = norm(t) if task == "tail" else norm(h)
            if gold_rank[(task, i)] is None:
                continue  # 部署态下无法命中
            plist = [norm(x) for x in P[i]["Prediction"]]
            if plist and plist[0] == gold:
                hits1 += 1
            if gold in plist[:10]:
                hits10 += 1
        deploy[f"{task}/{name}"] = {"Hits@1_lb": round(hits1 / n, 4), "Hits@10_lb": round(hits10 / n, 4)}
    res["deployment_lower_bound"] = deploy
    print(json.dumps(deploy, ensure_ascii=False, indent=2))

    with open("coverage_stages_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
