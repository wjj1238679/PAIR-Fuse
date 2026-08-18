# -*- coding: utf-8 -*-
"""对照实验评测: MiniMax-M2.5 在污染协议候选上的输出, 分别用旧匹配(bug版)+含重复348行计数
和修正匹配+唯一查询计数 两套口径评测, 用于论文 V-A 节 Table VII 的同模型对照。
用法: python eval_dirty.py
"""
import json, os, re, sys, unicodedata
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from group_metrics import load_triples, match_fixed  # match_fixed = 修正匹配

# ---------- 旧版(有 bug 的)匹配 ----------
def canon_old(name):
    s = unicodedata.normalize("NFKC", str(name or "")).lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace("recrod", "record")
    s = re.sub(r"(类|類|class|类型|類型)$", "", s)
    s = re.sub(r"[^0-9a-z_]+", "", s)
    return s

def eq_old(a, b):
    ca, cb = canon_old(a), canon_old(b)
    if not ca or not cb:
        return False
    if ca == cb:
        return True
    if ca.startswith(cb) or cb.startswith(ca):
        return abs(len(ca) - len(cb)) <= 2
    return False

def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return [json.loads(l) for l in f if l.strip()]

def main():
    DS = os.path.join(HERE, "dataset", "qt_cpp_kg")
    allt = load_triples("train.txt") + load_triples("valid.txt") + load_triples("test.txt")
    tt, th = defaultdict(set), defaultdict(set)
    for h, r, t in allt:
        tt[(h, r)].add(t)
        th[(r, t)].add(h)
    test = load_triples("test.txt")

    for task in ["tail", "head"]:
        P = load(os.path.join(HERE, "outputs", "qt_cpp_kg", "clean", f"dirty_{task}_P1_minimax.jsonl"))
        # 预测文件按 test.txt 行序(含重复查询)
        res = {}
        for label, matcher, dedup in [("旧匹配+含重复(=旧论文协议)", eq_old, False),
                                      ("修正匹配+含重复", match_fixed, False)]:
            ranks = []
            seen = set()
            for i, (h, r, t) in enumerate(test):
                if i >= len(P):
                    break
                key = f"{h}\t{r}" if task == "tail" else f"{t}\t{r}"
                if dedup and key in seen:
                    continue
                seen.add(key)
                golds = tt[(h, r)] if task == "tail" else th[(r, t)]
                plist = [str(x) for x in P[i]["Prediction"]]
                rk = None
                for j, p in enumerate(plist):
                    if any(matcher(p, g) for g in golds):
                        rk = j + 1
                        break
                ranks.append(rk or len(plist) + 1)
            n = len(ranks)
            res[label] = {
                "N": n,
                "H@1": round(sum(1 for x in ranks if x <= 1) / n, 4),
                "H@3": round(sum(1 for x in ranks if x <= 3) / n, 4),
                "H@10": round(sum(1 for x in ranks if x <= 10) / n, 4),
                "MRR": round(sum(1.0 / x for x in ranks) / n, 4),
            }
        fb = sum(1 for r in P if r["LLMSignals"].get("meta", {}).get("fallback"))
        print(f"== dirty {task} (MiniMax-M2.5, 污染候选) rows={len(P)} fallback={fb}")
        for k, v in res.items():
            print("  ", k, v)

if __name__ == "__main__":
    main()
