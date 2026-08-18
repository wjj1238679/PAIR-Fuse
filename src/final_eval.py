# -*- coding: utf-8 -*-
"""A6 汇总: 唯一查询口径的最终评测 (修正匹配 + multi-gold + 部署态拆分)
- 按 Question 去重: head N=319, tail N=203
- 评测态: gold 缺失时已在候选末尾(注入版排名)
- 部署态: 干净候选中无任何 gold 目标的查询直接计 miss
用法: python final_eval.py
"""
import json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from group_metrics import load, load_triples, match_fixed

DS = os.path.join(HERE, "dataset", "qt_cpp_kg")
CLEAN = os.path.join(HERE, "outputs", "qt_cpp_kg", "clean")

def build_gold():
    allt = load_triples("train.txt") + load_triples("valid.txt") + load_triples("test.txt")
    tt, th = defaultdict(set), defaultdict(set)
    for h, r, t in allt:
        tt[(h, r)].add(t)
        th[(r, t)].add(h)
    return tt, th

def unique_queries(task):
    """返回 [(question, key, golds, gold_in_clean)] 按首次出现顺序"""
    tt, th = build_gold()
    rows = load(os.path.join(DS, "clean", f"candidates_{task}.jsonl"))       # 部署版(无注入)
    seen, out = set(), []
    for r in rows:
        q = r["Question"]
        if q in seen:
            continue
        seen.add(q)
        if task == "tail":
            h, rel = r["Key"].split("\t", 1)
            golds = tt.get((h, rel), {r["Gold"]})
        else:
            t, rel = r["Key"].split("\t", 1)
            golds = th.get((rel, t), {r["Gold"]})
        cand_names = {c["name"] for c in r["Candidates"]}
        g_in = any(any(match_fixed(cn, g) for cn in cand_names) for g in golds)
        out.append((q, r["Key"], golds, g_in))
    return out

def eval_method(task, pred_rows, queries):
    """pred_rows: 预测文件行(含重复); 按 Question 去重取第一次"""
    by_q = {}
    for r in pred_rows:
        q = r.get("Question", "")
        if q and q not in by_q:
            by_q[q] = r
    ranks_ev, ranks_dep = [], []
    for q, key, golds, g_in in queries:
        r = by_q.get(q)
        if r is None:
            ranks_ev.append(999); ranks_dep.append(999); continue
        plist = [str(x) for x in r["Prediction"]]
        rk = None
        for j, p in enumerate(plist):
            if any(match_fixed(p, g) for g in golds):
                rk = j + 1
                break
        rk = rk or (len(plist) + 1)
        ranks_ev.append(rk)
        ranks_dep.append(rk if g_in else len(plist) + 1)

    def m(rs):
        n = len(rs)
        return {"H@1": round(sum(1 for x in rs if x <= 1) / n, 4),
                "H@3": round(sum(1 for x in rs if x <= 3) / n, 4),
                "H@10": round(sum(1 for x in rs if x <= 10) / n, 4),
                "MRR": round(sum(1.0 / x for x in rs) / n, 4)}
    return m(ranks_ev), m(ranks_dep)

def main():
    methods = {
        "tail": {
            "LLM-only (P0)": "full_tail_P0_minimax.jsonl",
            "LLM-only (P1)": "full_tail_P1_minimax.jsonl",
            "RotatE-only": "rotateonly_tail.jsonl",
            "Safe-RRF-guard (P1)": "fused_tail_P1_safe.jsonl",
            "Safe-RRF+flip (P1)": "fused_tail_P1_flip.jsonl",
            "Linear (P1)": "fused_tail_P1_linear.jsonl",
            "RRF(1,1) (P0)": "fused_tail_P0_rrf11.jsonl",
            "RRF(1,1) (P1)": "fused_tail_P1_rrf11.jsonl",
            "RRF(1,2) (P0)": "fused_tail_P0_rrf12.jsonl",
            "RRF(1,2) (P1)": "fused_tail_P1_rrf12.jsonl",
        },
        "head": {
            "LLM-only (P0)": "full_head_P0_minimax.jsonl",
            "LLM-only (P1)": "full_head_P1_minimax.jsonl",
            "RotatE-only": "rotateonly_head.jsonl",
            "Safe-RRF-guard (P1)": "fused_head_P1_safe.jsonl",
            "Safe-RRF+flip (P1)": "fused_head_P1_flip.jsonl",
            "RRF(1,1) (P0)": "fused_head_P0_rrf11.jsonl",
            "RRF(1,1) (P1)": "fused_head_P1_rrf11.jsonl",
            "RRF(1,2) (P0)": "fused_head_P0_rrf12.jsonl",
            "RRF(1,2) (P1)": "fused_head_P1_rrf12.jsonl",
        },
    }
    result = {}
    for task, ms in methods.items():
        queries = unique_queries(task)
        cov = sum(1 for _, _, _, g in queries if g) / len(queries)
        result[task] = {"N_unique": len(queries), "coverage_clean_multi_gold": round(cov, 4), "methods": {}}
        print(f"== {task}: N={len(queries)}, 干净候选 multi-gold 覆盖率={cov:.4f}")
        for name, fn in ms.items():
            path = os.path.join(CLEAN, fn)
            if not os.path.exists(path):
                print("  (缺文件跳过)", name, fn)
                continue
            ev, dep = eval_method(task, load(path), queries)
            result[task]["methods"][name] = {"eval": ev, "deploy": dep}
            print(f"  {name:22s} eval {ev}  deploy {dep}")
    with open(os.path.join(CLEAN, "final_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
