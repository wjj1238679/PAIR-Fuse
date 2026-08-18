# -*- coding: utf-8 -*-
import argparse, json, os, re, sys, math, time
from collections import defaultdict


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, required=False, help="兼容原有参数")
    ap.add_argument("--input", type=str, required=True, help="输入：link_prediction 生成的 JSONL")
    ap.add_argument("--output", type=str, required=True, help="输出：融合后的 JSONL")
    ap.add_argument("--weights", type=str, default="2.0,0.5,0.6,0.8",
                    help="RRF 各路排名权重，逗号分隔；会自动对齐实际可用的排名源数量")
    ap.add_argument("--k_rrf", type=int, default=8, help="RRF 中的 k")

    # [新增] LLM 截断参数，用于 K 值消融实验
    ap.add_argument("--llm_top_k", type=int, default=0,
                    help="消融实验参数：只取LLM输出的前K个结果参与打分(0表示不截断，使用全部)")

    # [新增] 核心绝杀：guard_top_k
    ap.add_argument("--guard_top_k", type=int, default=1,
                    help="核心机制：强行锁死 LLM 的前 K 个预测不参与结构化重排降级")

    ap.add_argument("--tau", type=float, default=0.025, help="兼容旧参数（未使用，仅记录）")
    ap.add_argument("--topm", type=int, default=8, help="兼容旧参数（未使用，仅记录）")
    ap.add_argument("--alpha", type=float, default=0.6, help="兼容旧参数（未使用，仅记录）")
    ap.add_argument("--encoding", type=str, default="utf-8-sig", help="输入文件编码")
    return ap.parse_args()


# ---------- 基础工具 ----------
def safe_iter_json_lines(path, encoding="utf-8-sig"):
    """
    鲁棒读取：逐行，先取 {...} 子串再 json 解析；失败则跳过该行但不中断整体。
    """
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        for ln in f:
            s = (ln or "").strip()
            if not s:
                continue
            # 行不是纯 JSON 时，尝试提取第一个 {...}
            if not s.startswith("{"):
                m = re.search(r"\{.*\}", s)
                if not m:
                    continue
                s = m.group(0)
            try:
                yield json.loads(s)
            except Exception:
                # 宽松修复：去尾逗号
                s2 = re.sub(r",(\s*[}\]])", r"\1", s)
                try:
                    yield json.loads(s2)
                except Exception:
                    continue


def to_rank(score_dict: dict) -> dict:
    """
    将“分数越大越好”的字典转成“名次（1=最好）”。
    若分数相同，按候选名称稳定排序。
    """
    items = list(score_dict.items())
    items.sort(key=lambda kv: (-float(kv[1]), str(kv[0])))
    return {k: i + 1 for i, (k, _) in enumerate(items)}


def rrf_fuse(ranks_list, k: int = 8, weights=None) -> dict:
    """
    Reciprocal Rank Fusion:
      score(c) = Σ_i [ w_i * 1 / (k + rank_i(c)) ]
    """
    if not ranks_list:
        return {}
    if weights is None:
        weights = [1.0] * len(ranks_list)
    # 所有候选全集
    all_cands = set()
    for rk in ranks_list:
        all_cands.update(rk.keys())
    scores = {}
    BIG = 10 ** 9
    for c in all_cands:
        s = 0.0
        for w_i, rk in zip(weights, ranks_list):
            r = rk.get(c, BIG)
            s += float(w_i) * (1.0 / (k + float(r)))
        scores[c] = s
    return scores


# ---------- 文本相似度（轻量，不依赖外部库） ----------
def _tokens(s: str):
    s = (s or "").lower()
    # 去掉标点，只保留字母数字和中文
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s)
    toks = [t for t in s.split() if t]
    return toks


def _vec(toks):
    from collections import Counter
    c = Counter(toks)
    n = math.sqrt(sum(v * v for v in c.values())) or 1.0
    return {k: v / n for k, v in c.items()}


def _cos(a, b):
    # 只遍历较小向量以提速
    if len(a) > len(b):
        a, b = b, a
    return sum(a.get(k, 0.0) * b.get(k, 0.0) for k in a.keys())


def simple_semantic_rank(question: str, cand_desc_map: dict):
    """
    用 question 与 (候选名 + desc) 的余弦相似度打分 → rank
    cand_desc_map: { name -> "name desc" }
    """
    qv = _vec(_tokens(question))
    scores = {name: _cos(qv, _vec(_tokens(text))) for name, text in cand_desc_map.items()}
    return to_rank(scores)


# ---------- 主要融合逻辑 ----------
def main():
    args = parse_args()
    t0 = time.time()

    # 解析权重
    try:
        w_all = [float(x) for x in str(args.weights).split(",") if str(x).strip() != ""]
    except Exception:
        w_all = [2.0, 0.5, 0.6, 0.8]

    in_path = args.input
    out_path = args.output
    out_dir = os.path.dirname(out_path)
    if out_dir:  # 只有当路径不为空时，才去创建文件夹
        os.makedirs(out_dir, exist_ok=True)

    total, triggered = 0, 0
    written = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for obj in safe_iter_json_lines(in_path, encoding=args.encoding):
            total += 1

            head = obj.get("HeadEntity", "")
            ques = obj.get("Question", "")
            # —— 候选集合（尽量从各种字段里兜底取到完整候选）——
            cand_names = []
            # 1) 优先 Prediction（list）
            pred = obj.get("Prediction")
            if isinstance(pred, list) and pred:
                cand_names = [str(x) for x in pred if str(x).strip()]
            # 2) 其次 LLMSignals.top_order
            if not cand_names:
                top_order0 = (((obj.get("LLMSignals") or {}).get("top_order")) or [])
                if isinstance(top_order0, list) and top_order0:
                    cand_names = [str(x) for x in top_order0 if str(x).strip()]
            # 3) 再次 final_answer
            if not cand_names and isinstance(obj.get("final_answer"), list):
                cand_names = [str(x) for x in obj["final_answer"] if str(x).strip()]
            # 4) 最后 candidates / names / entities（防御性）
            if not cand_names:
                for key in ("candidates", "names", "entities"):
                    val = obj.get(key)
                    if isinstance(val, list) and val:
                        cand_names = [str(x) for x in val if str(x).strip()]
                        break

            # 若仍然没有候选，写个空预测占位，绝不丢行
            if not cand_names:
                out = {
                    "HeadEntity": head,
                    "Question": ques,
                    "Prediction": [],
                    "FusedSignals": {"meta": {"fallback": True, "reason": "empty_candidates"}}
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                written += 1
                continue

            # 组装可用的排名源
            ranks = []
            names_set = set(cand_names)

            # [截断机制]：决定哪些实体能拿到 LLM 的排序分
            if args.llm_top_k > 0:
                allowed_llm_cands = set(cand_names[:args.llm_top_k])
            else:
                allowed_llm_cands = set(cand_names)

            # A) LLM 原顺序 (应用截断)
            rk_llm = {}
            for i, name in enumerate(cand_names):
                if name in allowed_llm_cands:
                    rk_llm[name] = i + 1
            if rk_llm:
                ranks.append(rk_llm)

            # B) positional_scores -> rank (应用截断)
            pos_scores = ((obj.get("LLMSignals") or {}).get("positional_scores")) or {}
            pos_scores = {str(k): float(v) for k, v in pos_scores.items() if
                          str(k) in names_set and str(k) in allowed_llm_cands}
            if pos_scores:
                ranks.append(to_rank(pos_scores))

            # C) top_order (应用截断)
            top_order = (((obj.get("LLMSignals") or {}).get("top_order")) or [])
            if isinstance(top_order, list) and top_order:
                rk_top = {}
                r = 1
                for name in top_order:
                    sname = str(name)
                    if sname in names_set and sname in allowed_llm_cands and sname not in rk_top:
                        rk_top[sname] = r;
                        r += 1
                if rk_top:
                    ranks.append(rk_top)

            # D) 简易文本相似度 (这是结构化兜底，保留全部候选人打分)
            desc_map = {}
            cands_obj = (obj.get("Candidates") or [])
            if isinstance(cands_obj, list) and cands_obj:
                for it in cands_obj:
                    if isinstance(it, dict):
                        nm = str(it.get("name") or it.get("Name") or "").strip()
                        if nm and nm in names_set:
                            desc = str(it.get("desc") or it.get("Desc") or it.get("description") or "")
                            desc_map[nm] = (nm + " " + desc).strip()
            for nm in cand_names:
                desc_map.setdefault(nm, nm)
            if desc_map and ques:
                ranks.append(simple_semantic_rank(ques, desc_map))

            # 若可用排名源数量 < 权重长度，自动对齐；若更多，则截断
            if len(ranks) == 0:
                # 极端保护：回退原顺序
                final_order = cand_names[:]
                rrf_scores = {n: 1.0 / (args.k_rrf + i + 1) for i, n in enumerate(final_order)}
                delta = 0.0
            else:
                if len(w_all) < len(ranks):
                    wts = w_all + [1.0] * (len(ranks) - len(w_all))
                else:
                    wts = w_all[:len(ranks)]

                rrf_scores = rrf_fuse(ranks, k=args.k_rrf, weights=wts)

                # 先按融合分数做全局降序排序
                fused_order = sorted(cand_names, key=lambda x: (-rrf_scores.get(x, 0.0), x))

                # ========================================================
                # [核心绝杀] guard_top_k 机制：锁死大模型最有把握的头部预测
                # ========================================================
                if args.guard_top_k > 0 and len(cand_names) >= args.guard_top_k:
                    # 提取大模型原始排序的前 K 个（即原始 cand_names 的前 K 个）
                    guarded_ents = cand_names[:args.guard_top_k]
                    # 剔除掉被保护的实体，剩下的按结构化融合分数排序
                    remaining_ents = [x for x in fused_order if x not in guarded_ents]
                    # 强行拼接：绝对的第一名 + 重新优化的长尾
                    final_order = guarded_ents + remaining_ents
                else:
                    final_order = fused_order

                # Top1-Top2 分差
                delta = 0.0
                if len(final_order) >= 2:
                    delta = float(rrf_scores.get(final_order[0], 0.0) - rrf_scores.get(final_order[1], 0.0))

            # 触发统计（有额外信号就算触发）
            trig = 1 if (pos_scores or top_order) else 0
            triggered += trig

            out = {
                "HeadEntity": head,
                "Question": ques,
                "Prediction": final_order,
                "FusedSignals": {
                    "rrf_scores": rrf_scores,
                    "delta_top12": delta,
                    "used_ranks": [len(r) for r in ranks],
                    "meta": {
                        "k_rrf": args.k_rrf,
                        "weights": w_all,
                        "llm_top_k_ablation": args.llm_top_k,
                        "guard_top_k": args.guard_top_k,  # 记录 guard 机制参数
                        "tau": args.tau,
                        "topm": args.topm,
                        "alpha": args.alpha
                    }
                }
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    stats = {
        "total_samples": total,
        "triggered": triggered,
        "trigger_ratio": round(triggered / total, 4) if total else 0.0,
        "elapsed_sec": round(time.time() - t0, 3),
        "reranker": "none",
        "written": written
    }
    print(
        f"[fusion - Guard K={args.guard_top_k}, Ablation K={args.llm_top_k}] stats: {json.dumps(stats, ensure_ascii=False)}")


if __name__ == "__main__":
    main()