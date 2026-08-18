# -*- coding: utf-8 -*-
"""B1-B4: 新协议下的显著性检验 / 参数扫描 / 关系组诊断 / case study
用法: python analysis_battery.py
"""
import json, os, sys, random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from group_metrics import load, load_triples, match_fixed
from final_eval import unique_queries

CLEAN = os.path.join(HERE, "outputs", "qt_cpp_kg", "clean")
random.seed(42)

def get_ranks(task, pred_file, queries):
    by_q = {}
    for r in load(os.path.join(CLEAN, pred_file)):
        q = r.get("Question", "")
        if q and q not in by_q:
            by_q[q] = r
    ranks = []
    for q, key, golds, g_in in queries:
        r = by_q.get(q)
        if r is None:
            ranks.append(999); continue
        plist = [str(x) for x in r["Prediction"]]
        rk = None
        for j, p in enumerate(plist):
            if any(match_fixed(p, g) for g in golds):
                rk = j + 1
                break
        ranks.append(rk or len(plist) + 1)
    return ranks

def bootstrap_ci(ranks, fn, n_boot=10000):
    n = len(ranks)
    vals = []
    for _ in range(n_boot):
        sample = [ranks[random.randrange(n)] for _ in range(n)]
        vals.append(fn(sample))
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]

def mcnemar(ranks_a, ranks_b, k=1):
    """H@k 的 McNemar: a 对 b 错 vs a 错 b 对"""
    b01 = sum(1 for x, y in zip(ranks_a, ranks_b) if x <= k and y > k)
    b10 = sum(1 for x, y in zip(ranks_a, ranks_b) if x > k and y <= k)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    from math import comb
    p = 2 * min(1.0, sum(comb(n, i) * 0.5 ** n for i in range(0, min(b01, b10) + 1)) * 1.0)
    return b01, b10, round(p, 4)

def main():
    out = {}
    for task in ["tail", "head"]:
        queries = unique_queries(task)
        files = {
            "LLM-P1": f"full_{task}_P1_minimax.jsonl",
            "LLM-P0": f"full_{task}_P0_minimax.jsonl",
            "RotatE": f"rotateonly_{task}.jsonl",
            "Safe-RRF": f"fused_{task}_P1_safe.jsonl",
            "RRF11-P1": f"fused_{task}_P1_rrf11.jsonl",
            "RRF12-P0": f"fused_{task}_P0_rrf12.jsonl",
            "RRF12-P1": f"fused_{task}_P1_rrf12.jsonl",
        }
        R = {name: get_ranks(task, fn, queries) for name, fn in files.items() if os.path.exists(os.path.join(CLEAN, fn))}

        # --- B1: bootstrap CI + McNemar ---
        ci = {}
        for name, rs in R.items():
            h1 = bootstrap_ci(rs, lambda x: sum(1 for v in x if v <= 1) / len(x))
            mrr = bootstrap_ci(rs, lambda x: sum(1.0 / v for v in x) / len(x))
            ci[name] = {"H@1_95CI": [round(v, 4) for v in h1], "MRR_95CI": [round(v, 4) for v in mrr]}
        pairs = [("RRF12-P0", "RotatE"), ("RRF11-P1", "RotatE"), ("Safe-RRF", "RotatE"), ("LLM-P1", "LLM-P0")]
        mc = {}
        for a, b in pairs:
            if a in R and b in R:
                mc[f"{a} vs {b}"] = {"H@1": mcnemar(R[a], R[b], 1), "H@10": mcnemar(R[a], R[b], 10)}
        out[task] = {"N": len(queries), "bootstrap_CI": ci, "mcnemar(b01=a对b错, b10=a错b对, p)": mc}

        # --- B3: 关系组 ---
        gj = json.load(open(os.path.join(HERE, "diag", "relation_groups_tail.json"), encoding="utf-8-sig"))
        grp = {r: v["group"] for r, v in gj["relation_groups"].items()}
        groups = defaultdict(lambda: defaultdict(list))
        for (q, key, golds, g_in), i in zip(queries, range(len(queries))):
            rel = key.split("\t", 1)[1]
            g = grp.get(rel, "?")
            for name, rs in R.items():
                groups[g][name].append(rs[i])
        reltype = {}
        for g, mrs in sorted(groups.items()):
            reltype[g] = {"n": sum(len(v) for v in mrs.values()) // max(1, len(mrs)),
                          **{name: {"H@1": round(sum(1 for x in rs if x <= 1) / len(rs), 4),
                                    "H@10": round(sum(1 for x in rs if x <= 10) / len(rs), 4),
                                    "MRR": round(sum(1.0 / x for x in rs) / len(rs), 4)}
                             for name, rs in mrs.items()}}
        out[task]["relation_groups"] = reltype

    print(json.dumps(out, ensure_ascii=False, indent=1))
    with open(os.path.join(CLEAN, "significance_and_groups.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
