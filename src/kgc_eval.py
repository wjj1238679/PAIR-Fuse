# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import math
import unicodedata
from collections import defaultdict
from typing import List, Dict, Tuple, Iterable, Optional

############################################
# 统一的名称规范化 & 宽松匹配
############################################

_SUFFIXES = ("类", "接口", "枚举", "结构体", "方法", "函数", "窗口", "控件", "对象", "类型", "類", "類型")
_RE_SPACE = re.compile(r"\s+")
_RE_TAIL = re.compile(r"(?:%s)$" % "|".join(map(re.escape, _SUFFIXES)))

def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", str(s or ""))

def norm_name_for_match(x: str) -> str:
    s = nfkc(x)
    s = _RE_SPACE.sub("", s)
    s = s.replace("Recrod", "Record")
    s = _RE_TAIL.sub("", s)
    s = s.replace("类类", "类")
    return s

def canon(name: str) -> str:
    s = nfkc(name).lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace("recrod", "record")
    s = re.sub(r"(类|類|class|类型|類型)$", "", s)
    # 2026-08-15 修复: 原正则 [^0-9a-z_]+ 会剥掉所有非 ASCII 字符,
    # 导致纯中文实体名永远无法匹配(评测 bug)。保留 CJK 统一表意文字。
    s = re.sub(r"[^0-9a-z_一-鿿]+", "", s)
    return s

def names_equalish(a: str, b: str) -> bool:
    # 先做规范化精确匹配(覆盖纯中文实体名)
    if norm_name_for_match(a) == norm_name_for_match(b):
        return True
    ca, cb = canon(a), canon(b)
    if not ca or not cb:
        return False
    if ca == cb:
        return True
    if ca.startswith(cb) or cb.startswith(ca):
        if abs(len(ca) - len(cb)) <= 2:
            return True
    def strip_tail(s: str) -> str:
        for t in ("lei", "类", "類", "class", "类型", "類型"):
            if s.endswith(t):
                return s[: -len(t)]
        return s
    return strip_tail(ca) == strip_tail(cb)

def canonicalize_set(names: Iterable[str]) -> set:
    return {norm_name_for_match(n) for n in names if n is not None}

############################################
# 关系 ID/名称映射
############################################

def read_relid_map(ds_dir: str) -> dict:
    rel2id = {}
    for p in (os.path.join(ds_dir, "relation2id.txt"),
              os.path.join(ds_dir, "get_neighbor", "relation2id.txt")):
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8-sig") as f:
                for ln in f:
                    s = ln.strip()
                    if not s: continue
                    cols = re.split(r"\t+", s)
                    if len(cols) < 2: continue
                    a, b = cols[0].strip(), cols[1].strip()
                    rid = None; name = None
                    if re.fullmatch(r"\d+", b):
                        name, rid = a, int(b)
                    elif re.fullmatch(r"\d+", a):
                        name, rid = b, int(a)
                    if rid is not None:
                        if name: rel2id[name] = rid
                        rel2id[str(rid)] = rid
    return rel2id

def rel_to_id(r: Optional[str], rel2id: dict) -> Optional[int]:
    if r is None: return None
    r = str(r).strip()
    if not r: return None
    if re.fullmatch(r"\d+", r): return int(r)
    return rel2id.get(r)

############################################
# 数据载入
############################################

def read_triples(ds_dir: str) -> List[Tuple[str, str, str]]:
    out = []
    for fn in ("train.txt", "valid.txt", "test.txt"):
        fp = os.path.join(ds_dir, fn)
        if not os.path.exists(fp): continue
        with open(fp, "r", encoding="utf-8-sig") as f:
            for ln in f:
                s = ln.strip()
                if not s: continue
                parts = s.split('\t')
                if len(parts) != 3:
                    parts = re.split(r'\s+', s)
                    if len(parts) != 3: continue
                h, r, t = parts
                out.append((h, r, t))
    return out

def read_relations(ds_dir: str) -> List[str]:
    rels = []
    for p in (os.path.join(ds_dir, "relation2id.txt"),
              os.path.join(ds_dir, "get_neighbor", "relation2id.txt")):
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8-sig") as f:
                for ln in f:
                    s = ln.strip()
                    if not s: continue
                    cols = re.split(r'\t+', s)
                    if len(cols) >= 1:
                        rels.append(cols[0])
    return list(dict.fromkeys(rels))

############################################
# 关系路径规范化（关键修复）
############################################

def _normalize_rel_path(rel: str) -> str:
    """
    关系路径规范化：
    - 把类似 '/ /方法/原理/应用' 里的空段去掉：'/\\s*/' -> '/'
    - 多重斜杠压缩：'///' -> '/'
    - 去首尾空白
    """
    x = nfkc(rel or "")
    x = re.sub(r'/\s*/', '/', x)  # 去掉空段
    x = re.sub(r'/+', '/', x)     # 折叠多斜杠
    return x.strip()

############################################
# 从 Question 提取关系 r
############################################

def extract_rel_from_question(question: str, known_rels: List[str]) -> str:
    q = nfkc(question)
    q_c = canon(q)
    best_rel = ""; best_len = -1
    for r in (known_rels or []):
        rc = canon(r)
        if rc and rc in q_c and len(rc) > best_len:
            best_len = len(rc); best_rel = r
    if best_rel:
        return _normalize_rel_path(best_rel)
    m = re.search(r'(/[^/\[\]]+(?:/[^/\[\]]+)*)\s*\[MASK\]', q)
    if m:
        return _normalize_rel_path(m.group(1).strip())
    qm = re.split(r'\[MASK\]', q, 1)
    head_part = qm[0] if qm else q
    toks = re.split(r'[\s，,。；;]+', head_part.strip())
    if toks:
        return _normalize_rel_path(toks[-1])
    return _normalize_rel_path(q)

############################################
# 预测文件读取（兼容原始 JSONL & to-eval）
############################################

def load_predictions(path: str) -> List[dict]:
    rows = []
    with open(path, "rb") as fb:
        data = fb.read()
    decoded = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "cp936", "latin-1"):
        try:
            decoded = data.decode(enc); break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        decoded = data.decode("utf-8", errors="ignore")

    for ln in decoded.splitlines():
        s = ln.strip()
        if not s: continue
        try:
            obj = json.loads(s)
        except Exception:
            m = re.search(r'\{.*\}', s)
            if not m: continue
            try:
                obj = json.loads(m.group(0))
            except Exception:
                continue

        head = obj.get("HeadEntity", "")
        q = obj.get("Question", "")
        key = obj.get("Key", "")
        rel = obj.get("Relation", "")  # 兼容直接给 Relation
        pred = obj.get("Prediction", obj.get("final_answer", []))

        if isinstance(pred, str):
            cand = [x for x in (p.strip() for p in pred.split('|')) if x]
        elif isinstance(pred, list):
            cand = [str(x).strip() for x in pred if str(x).strip()]
        else:
            cand = []

        rows.append({"HeadEntity": head, "Question": q, "Prediction": cand, "Key": key, "Relation": rel})
    return rows

############################################
# 评估指标
############################################

def dcg_at_k(rel_list: List[int], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(rel_list[:k], start=1):
        if rel > 0:
            dcg += 1.0 / math.log2(i + 1.0)
    return dcg

def ideal_dcg_at_k(num_pos: int, k: int) -> float:
    num = min(num_pos, k)
    idcg = 0.0
    for i in range(1, num + 1):
        idcg += 1.0 / math.log2(i + 1.0)
    return idcg

def eval_one(pred_list: List[str], gold_names: Iterable[str], ks: List[int]) -> Dict[str, float]:
    gold = list(gold_names)
    gold_canon = {canon(g) for g in gold if g is not None}
    first_hit_rank = None
    rel_list = [0] * len(pred_list)
    hit_any = False
    hits_positions = []
    for idx, name in enumerate(pred_list):
        ok = any(names_equalish(name, g) for g in gold)
        if ok:
            rel_list[idx] = 1
            hit_any = True
            if first_hit_rank is None:
                first_hit_rank = idx + 1
            hits_positions.append(idx + 1)
    if first_hit_rank is None:
        first_hit_rank = len(pred_list) + 1
    out = {}
    for k in ks:
        hitk = 1.0 if any(p <= k for p in hits_positions) else 0.0
        out[f"Hits@{k}"] = hitk
        dcg = dcg_at_k(rel_list, k)
        idcg = ideal_dcg_at_k(len(gold_canon), k)
        out[f"NDCG@{k}"] = (dcg / idcg) if idcg > 0 else 0.0
        if out[f"NDCG@{k}"] > 1.0:
            out[f"NDCG@{k}"] = 1.0
    out["MR"] = float(first_hit_rank)
    out["MRR"] = 1.0 / first_hit_rank if first_hit_rank <= len(pred_list) else 0.0
    if hits_positions:
        hits_positions.sort()
        ap_sum = 0.0
        correct_so_far = 0
        for p in range(1, len(pred_list) + 1):
            if p in hits_positions:
                correct_so_far += 1
                ap_sum += (correct_so_far / p)
        denom = min(len(gold_canon), len(hits_positions)) if len(gold_canon) > 0 else 1
        out["MAP"] = ap_sum / denom if denom > 0 else 0.0
    else:
        out["MAP"] = 0.0
    out["RecallAnywhere"] = 1.0 if hit_any else 0.0
    return out

def average_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not all_metrics: return {}
    keys = list(all_metrics[0].keys())
    agg = {k: 0.0 for k in keys}
    for m in all_metrics:
        for k in keys:
            agg[k] += float(m.get(k, 0.0))
    n = float(len(all_metrics))
    for k in keys:
        agg[k] = agg[k] / n
    return agg

############################################
# 主评估流程（支持 task=head/tail；Key/Relation/Question 三路解析）
############################################

def evaluate(pred_path: str,
             dataset_dir: str,
             ks: List[int],
             mode: str = "fuzzy",
             save_overall: str = "",
             group_by: str = "",
             used_pred_out: str = "",
             task: str = "tail",
             dump_skipped: Optional[str] = None) -> Tuple[Dict, Dict, str]:

    triples = read_triples(dataset_dir)
    rel_names = read_relations(dataset_dir)
    rel2id_map = read_relid_map(dataset_dir)

    gold_hr_to_t: Dict[Tuple[str, str], set] = defaultdict(set)  # (h,r)->{t}
    gold_tr_to_h: Dict[Tuple[str, str], set] = defaultdict(set)  # (t,r)->{h}
    for h, r, t in triples:
        gold_hr_to_t[(h, r)].add(t)
        gold_tr_to_h[(t, r)].add(h)

    rows = load_predictions(pred_path)
    if not rows:
        return {"N": 0}, {}, pred_path

    metrics = []
    used_lines = 0
    skipped = []

    for idx, ex in enumerate(rows):
        head_ent = ex.get("HeadEntity", "")
        q = ex.get("Question", "")
        pred_list = ex.get("Prediction", [])
        key = ex.get("Key", "")
        rel_field = ex.get("Relation", "")

        if not head_ent and q:
            m = re.match(r"(.+?)\s+/", q)
            if m: head_ent = m.group(1).strip()
        if not pred_list or not head_ent:
            skipped.append({"index": idx, "reason": "empty-pred-or-head", "ex": ex})
            continue

        # --- 关系三路解析：Key > Relation 字段 > Question ---
        rel = ""
        tail_from_key = None
        if key and "\t" in key:
            parts = key.split("\t", 1)
            tail_from_key, rel = parts[0], parts[1]
        elif rel_field:
            rel = rel_field
        else:
            rel = extract_rel_from_question(q, rel_names)

        # 统一规范化关系路径（关键）
        rel = _normalize_rel_path(rel)

        # --- 基于 task 分支 ---
        gold_targets = set()

        if task == "tail":
            h_ent = head_ent
            rid_q = rel_to_id(rel, rel2id_map)
            if rid_q is not None:
                ch = canon(h_ent)
                for (h2, r2), ts in gold_hr_to_t.items():
                    rid2 = rel_to_id(r2, rel2id_map)
                    if rid2 == rid_q and (h2 == h_ent or canon(h2) == ch):
                        gold_targets |= ts
            if not gold_targets:
                cr = canon(rel)
                key_candidates = [(h_ent, rel)]
                if cr:
                    for (_, r2, _) in triples:
                        if canon(r2) == cr and (h_ent, r2) not in key_candidates:
                            key_candidates.append((h_ent, r2))
                for k2 in key_candidates:
                    if k2 in gold_hr_to_t:
                        gold_targets |= gold_hr_to_t[k2]

        else:  # task == "head"
            tail_ent = tail_from_key if tail_from_key else head_ent
            rid_q = rel_to_id(rel, rel2id_map)
            if rid_q is not None:
                ct = canon(tail_ent)
                for (t2, r2), hs in gold_tr_to_h.items():
                    rid2 = rel_to_id(r2, rel2id_map)
                    if rid2 == rid_q and (t2 == tail_ent or canon(t2) == ct):
                        gold_targets |= hs
            if not gold_targets:
                cr = canon(rel)
                key_candidates = [(tail_ent, rel)]
                if cr:
                    for (_, r2, _) in triples:
                        if canon(r2) == cr and (tail_ent, r2) not in key_candidates:
                            key_candidates.append((tail_ent, r2))
                for k2 in key_candidates:
                    if k2 in gold_tr_to_h:
                        gold_targets |= gold_tr_to_h[k2]
            if not gold_targets and (not rel or not rel.strip()):
                rels_for_tail = [(t2, r2) for (t2, r2) in gold_tr_to_h.keys() if t2 == tail_ent]
                if len(rels_for_tail) == 1:
                    only_pair = rels_for_tail[0]
                    gold_targets |= gold_tr_to_h[only_pair]
            if not gold_targets:
                ct = canon(tail_ent)
                if ct:
                    for (t2, r2), hs in gold_tr_to_h.items():
                        if canon(t2) == ct:
                            gold_targets |= hs

        if not gold_targets:
            skipped.append({
                "index": idx,
                "reason": "no-gold-target",
                "parsed": {"head_ent": head_ent, "tail_from_key": tail_from_key, "rel": rel},
                "key": key,
                "question": q
            })
            continue

        m = eval_one(pred_list, gold_targets, ks)
        metrics.append(m)
        used_lines += 1

    overall = average_metrics(metrics)
    overall["N"] = used_lines
    overall["Mode"] = mode
    overall["Ks"] = ks
    overall["GroupBy"] = group_by or ""

    by_group = {}

    if save_overall:
        with open(save_overall, "w", encoding="utf-8") as w:
            w.write(json.dumps(overall, ensure_ascii=False, indent=2))

    if dump_skipped:
        with open(dump_skipped, "w", encoding="utf-8") as w:
            for it in skipped:
                w.write(json.dumps(it, ensure_ascii=False) + "\n")

    used_pred = pred_path
    return overall, by_group, used_pred

############################################
# to-eval：把 Prediction 列表转成 "|" 串（兼容旧流程）
############################################

def to_eval(in_path: str, out_path: str) -> int:
    n = 0
    with open(in_path, "r", encoding="utf-8-sig") as f, \
            open(out_path, "w", encoding="utf-8") as w:
        for ln in f:
            s = ln.strip()
            if not s: continue
            try:
                obj = json.loads(s)
            except Exception:
                m = re.search(r'\{.*\}', s)
                if not m: continue
                try:
                    obj = json.loads(m.group(0))
                except Exception:
                    continue
            head = obj.get("HeadEntity", "")
            q = obj.get("Question", "")
            pred = obj.get("Prediction", obj.get("final_answer", []))
            if isinstance(pred, str):
                pred_str = pred
            elif isinstance(pred, list):
                pred_str = "|".join(str(x).strip() for x in pred if str(x).strip())
            else:
                pred_str = ""
            out = {"HeadEntity": head, "Question": q, "Prediction": pred_str}
            w.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    return n

############################################
# CLI
############################################

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_eval = sub.add_parser("eval", help="Evaluate predictions.")
    ap_eval.add_argument("--pred", type=str, required=True, help="Path to prediction file (raw or to-eval).")
    ap_eval.add_argument("--dataset_dir", type=str, required=True)
    ap_eval.add_argument("--ks", type=int, nargs="+", default=[1, 3, 10])
    ap_eval.add_argument("--mode", type=str, default="fuzzy")
    ap_eval.add_argument("--group_by", type=str, default="")
    ap_eval.add_argument("--save_overall", type=str, default="")
    ap_eval.add_argument("--task", choices=["tail", "head"], default="tail",
                         help="评估方向：tail=(h,r)->t；head=(t,r)->h。默认 tail。")
    ap_eval.add_argument("--dump_skipped", type=str, default="",
                         help="把被跳过的样本（无 gold 键等）写入指定 JSONL 文件，便于定位。")

    ap_to = sub.add_parser("to-eval", help="Convert raw jsonl to eval jsonl with pipe-separated predictions.")
    ap_to.add_argument("--in", dest="in_path", type=str, required=True)
    ap_to.add_argument("--out", dest="out_path", type=str, required=True)

    args = ap.parse_args()

    if args.cmd == "to-eval":
        n = to_eval(args.in_path, args.out_path)
        print(f"[to-eval] converted lines: {n} → {args.out_path}")
        return

    if args.cmd == "eval":
        overall, by_group, used_pred = evaluate(
            pred_path=args.pred,
            dataset_dir=args.dataset_dir,
            ks=args.ks,
            mode=args.mode,
            group_by=args.group_by,
            save_overall=args.save_overall,
            task=args.task,
            dump_skipped=(args.dump_skipped or None)
        )
        print(f"[eval] used pred file: {args.pred} ")
        pretty = json.dumps(overall, ensure_ascii=False, indent=2)
        print(pretty)
        with open("metrics_by_group.json", "w", encoding="utf-8") as w:
            w.write(json.dumps(by_group, ensure_ascii=False, indent=2))
        print("[eval] group metrics saved to: metrics_by_group.json")
        return

if __name__ == "__main__":
    main()
