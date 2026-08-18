# -*- coding: utf-8 -*-
"""E1: RotatE 在 validation split 上的全图 filtered 排名评估 (head/tail, 总体+按关系组)
用途: 为 PAIR-Fuse 的"按结构通道可靠性选择融合策略"提供验证集依据 (零 API 成本)。
用法: python rotate_valid_eval.py
"""
import json, os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "dataset", "qt_cpp_kg")
CKPT = os.path.join(DS, "rotate", "rotate_qt_dim1280_recip.pt")

def load_triples(fn):
    out = []
    with open(os.path.join(DS, fn), encoding="utf-8-sig") as f:
        for ln in f:
            sp = ln.rstrip("\n").split("\t")
            if len(sp) == 3:
                out.append(tuple(sp))
    return out

def main():
    ck = torch.load(CKPT, map_location="cpu")
    ents, rels = ck["ents"], ck["rels"]
    e2i = {e: i for i, e in enumerate(ents)}
    r2i = {r: i for i, r in enumerate(rels)}
    E = ck["ent"]["weight"]          # [Ne, 2d]
    R = ck["rel"]["weight"]          # [Nr, d] (相位)
    d = R.shape[1]
    Ec = torch.complex(E[:, :d], E[:, d:])   # [Ne, d] 复数
    Rc = torch.polar(torch.ones_like(R), R)  # [Nr, d] 复数

    train, valid, test = load_triples("train.txt"), load_triples("valid.txt"), load_triples("test.txt")
    allt = train + valid + test
    # filtered 设置所需: 每个 (r,t) 的真实头集合 / 每个 (h,r) 的真实尾集合
    true_heads = {}
    true_tails = {}
    for h, r, t in allt:
        true_tails.setdefault((h, r), set()).add(t)
        true_heads.setdefault((r, t), set()).add(h)

    # 关系分组 (复用 diag/relation_groups_tail.json)
    groups = {}
    gpath = os.path.join(HERE, "diag", "relation_groups_tail.json")
    if os.path.exists(gpath):
        gj = json.load(open(gpath, encoding="utf-8-sig"))
        groups = {r: v["group"] for r, v in gj["relation_groups"].items()}

    def evaluate(triples, task):
        ranks = []
        skipped = 0
        per_rel = {}
        for h, r, t in triples:
            if task == "tail":
                qn, rn, gold = h, r, t
                known_true = true_tails.get((h, r), set())
            else:
                qn, rn, gold = t, r + "^inv", h
                known_true = true_heads.get((r, t), set())
            if qn not in e2i or rn not in r2i or gold not in e2i:
                skipped += 1
                ranks.append(len(ents))
                rel = r
                per_rel.setdefault(rel, []).append(len(ents))
                continue
            q = Ec[e2i[qn]] * Rc[r2i[rn]]          # [d]
            scores = -torch.abs(q.unsqueeze(0) - Ec).sum(dim=1)  # [Ne] L1
            scores[e2i[qn]] = float("-inf")       # 排除查询实体自身(候选集语义)
            gold_i = e2i[gold]
            gs = scores[gold_i].item()
            better = (scores > gs).sum().item()
            equal = (scores == gs).sum().item()
            rank = better + 1
            # filtered: 减去排名在 gold 之前的其他已知正确实体
            filt = 0
            for e in known_true:
                if e != gold and e in e2i and scores[e2i[e]].item() >= gs:
                    filt += 1
            rank -= filt
            rank = max(1, rank)
            ranks.append(rank)
            per_rel.setdefault(r, []).append(rank)

        def metrics(rs):
            n = len(rs)
            return {
                "N": n,
                "Hits@1": round(sum(1 for x in rs if x <= 1) / n, 4),
                "Hits@3": round(sum(1 for x in rs if x <= 3) / n, 4),
                "Hits@10": round(sum(1 for x in rs if x <= 10) / n, 4),
                "MRR": round(sum(1.0 / x for x in rs) / n, 4),
                "MR": round(sum(rs) / n, 1),
            }

        out = {"overall": metrics(ranks), "skipped_no_embedding": skipped}
        if groups:
            by_group = {}
            for r, rs in per_rel.items():
                by_group.setdefault(groups.get(r, "?"), []).extend(rs)
            out["by_group"] = {g: metrics(rs) for g, rs in sorted(by_group.items())}
            out["by_relation"] = {r: metrics(rs) for r, rs in sorted(per_rel.items())}
        return out

    res = {"valid": {"tail": evaluate(valid, "tail"), "head": evaluate(valid, "head")},
           "test_fullgraph": {"tail": evaluate(test, "tail"), "head": evaluate(test, "head")}}
    print(json.dumps(res, ensure_ascii=False, indent=2))
    with open("rotate_valid_eval_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
