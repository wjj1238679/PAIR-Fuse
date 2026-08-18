# -*- coding: utf-8 -*-
"""A1: 构建干净候选集(无 gold 注入,检索顺序,gold 不置顶)
输出到 dataset/qt_cpp_kg/clean/:
  candidates_{head,tail}.jsonl        部署版: 纯检索 top-80, 无 gold
  candidates_{head,tail}_eval.jsonl   评测版: 若 gold 缺失则追加到末尾, 并记录 gold_injected 标记
每行: {"Key","Question","Candidates":[{"name","desc"}...],"Gold","GoldInCandidates","GoldInjected"}
Question 文本沿用原输出文件中的格式(与 v1/v2 一致)。
用法: python build_clean_candidates.py
"""
import json, os, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "dataset", "qt_cpp_kg")
OUT = os.path.join(DS, "clean")
K = 80

def norm(s):
    s = unicodedata.normalize("NFKC", str(s or ""))
    return "".join(s.split()).lower()

def bigrams(s):
    s = norm(s)
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i+2] for i in range(len(s)-1)}

def load_map(path, key_col=0, val_col=2):
    m = {}
    with open(path, encoding="utf-8-sig", errors="ignore") as f:
        for ln in f:
            sp = ln.rstrip("\n").split("\t")
            if len(sp) > max(key_col, val_col):
                m[sp[key_col].strip()] = sp[val_col].strip()
    return m

def main():
    os.makedirs(OUT, exist_ok=True)
    ent_text = load_map(os.path.join(DS, "entity2text.txt"))
    rel_text = load_map(os.path.join(DS, "relation2text.txt"), 0, 1)
    ents = list(ent_text.keys())
    ent_bg = {}
    for e in ents:
        ent_bg[e] = bigrams(ent_text[e]) | bigrams(e)

    # 原输出文件中的 Question 文本(保持与 v1/v2 一致的查询表述)
    question = {}
    for task, fn in [("head", "outputs/qt_cpp_kg/head/output_head.chat.txt"),
                     ("tail", "outputs/qt_cpp_kg/tail/output_tail.chat.txt")]:
        with open(os.path.join(HERE, fn), encoding="utf-8-sig") as f:
            for i, ln in enumerate(f):
                if ln.strip():
                    question[(task, i)] = json.loads(ln)["Question"]

    test = []
    with open(os.path.join(DS, "test.txt"), encoding="utf-8-sig") as f:
        for ln in f:
            sp = ln.rstrip("\n").split("\t")
            if len(sp) == 3:
                test.append(tuple(sp))

    stats = {}
    for task in ["head", "tail"]:
        rows, rows_eval = [], []
        cov = 0
        for i, (h, r, t) in enumerate(test):
            known, gold = (t, h) if task == "head" else (h, t)
            key = f"{known}\t{r}"
            qtext = ent_text.get(known, known) + " " + rel_text.get(r, r.replace("/", " "))
            qbg = bigrams(qtext) | bigrams(known)
            scored = []
            for e in ents:
                inter = len(qbg & ent_bg[e])
                if inter:
                    scored.append((2.0 * inter / (len(qbg) + len(ent_bg[e])), e))
            scored.sort(key=lambda x: -x[0])
            top = [e for _, e in scored[:K]]
            # 候选不得包含查询实体自身
            top = [e for e in top if norm(e) != norm(known)][:K]

            gold_in = norm(gold) in {norm(e) for e in top}
            if gold_in:
                cov += 1
            cands = [{"name": e, "desc": ent_text.get(e, "")[:200]} for e in top]
            row = {"Key": key, "Question": question.get((task, i), f"{known} {r} [MASK]"),
                   "Gold": gold, "GoldInCandidates": gold_in, "GoldInjected": False,
                   "Candidates": cands}
            rows.append(row)
            # 评测版: gold 缺失时追加到末尾(不置顶), 并标记
            row_ev = dict(row)
            if not gold_in:
                row_ev["Candidates"] = cands + [{"name": gold, "desc": ent_text.get(gold, "")[:200]}]
                row_ev["GoldInjected"] = True
            rows_eval.append(row_ev)

        for fn, rs in [(f"candidates_{task}.jsonl", rows), (f"candidates_{task}_eval.jsonl", rows_eval)]:
            with open(os.path.join(OUT, fn), "w", encoding="utf-8") as w:
                for r_ in rs:
                    w.write(json.dumps(r_, ensure_ascii=False) + "\n")
        stats[task] = {"N": len(test), "K": K, "retriever_gold_coverage": round(cov / len(test), 4),
                       "gold_injected_eval_version": len(test) - cov}
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    with open(os.path.join(OUT, "build_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
