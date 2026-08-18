# -*- coding: utf-8 -*-
# RRF fusion (LLM + struct/rule) with robust field autodetect, pair/tail fallback,
# explicit score direction (asc/desc), stronger normalization, overlap & flip diagnostics.
# —— 覆盖原文件版 fuse_llm_struct_rule.py
import argparse, json, codecs, re, math
from typing import Dict, List, Tuple, Optional, Any

############################
# name normalize
############################
def nfkc(s: str) -> str:
    try:
        import unicodedata
        return unicodedata.normalize("NFKC", str(s or ""))
    except Exception:
        return str(s or "")

_SUFFIXES = (
    "类","類","接口","枚举","結構體","结构体","方法","函数","窗口","控件","对象","类型","類型",
    "class","interface","enum","struct","method","function","window","widget","object","type"
)

def _strip_suffix_once(s: str) -> str:
    for suf in _SUFFIXES:
        if len(s) >= len(suf) and s.endswith(suf):
            return s[:-len(suf)]
    return s

def norm_zhcpp(s: str) -> str:
    s = nfkc(s).lower()
    s = s.replace("->", "::").replace(".", "::")
    s = re.sub(r"<[^<>]*>", "", s)        # strip template
    s = re.sub(r"\(\s*\)$", "", s)        # strip trailing ()
    s = re.sub(r"\s+", "", s)             # remove whitespaces
    s = s.replace("recrod", "record")     # common typo
    s = _strip_suffix_once(s)             # strip suffix once
    s = re.sub(r"::+", "::", s).strip(":")
    return s

def pick_norm(name_norm: str):
    if (name_norm or "").lower() in ("cpp","zhcpp","zh-cpp"):
        return norm_zhcpp
    return lambda x: nfkc(x or "")

############################
# io helpers
############################
def load_jsonl(path: str) -> List[dict]:
    rows = []
    with codecs.open(path, "r", "utf-8-sig", "ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                o = json.loads(s)
            except Exception:
                m = re.search(r"\{.*\}", s)
                o = json.loads(m.group(0)) if m else None
            if o is not None:
                rows.append(o)
    return rows

def dump_jsonl(path: str, objs: List[dict]):
    with codecs.open(path, "w", "utf-8-sig", "ignore") as fo:
        for o in objs:
            fo.write(json.dumps(o, ensure_ascii=False) + "\n")

############################
# key helpers
############################
def key_of_llm(o: dict) -> Tuple[str, str]:
    head = o.get("HeadEntity") or o.get("Head") or o.get("head") or ""
    rel  = o.get("Relation") or o.get("Rel") or o.get("Predicate") or ""
    if not rel:
        q = o.get("Question") or o.get("question") or ""
        m = re.search(r'(/[^/\[\]]+(?:/[^/\[\]]+)*)\s*\[MASK\]', q)
        rel = m.group(1).strip() if m else q
    return (nfkc(head), nfkc(rel))

def split_struct_key(key: str) -> Tuple[str, str]:
    s = nfkc(key)
    parts = s.split("\t")
    if len(parts) >= 2:
        return (parts[0].strip(), parts[1].strip())
    return (s.strip(), "")

############################
# score field detection
############################
_NUM_KEYS_CAND = ["struct", "score", "prob", "p", "s", "val", "value"]

def is_num_dict(v: Any) -> bool:
    if not isinstance(v, dict) or not v:
        return False
    try:
        for x in v.values():
            float(x)
            break
        return True
    except Exception:
        return False

def detect_struct_layout(row: dict,
                         prefer_field: Optional[str],
                         prefer_score_key: Optional[str]) -> Tuple[str, Optional[str]]:
    # return (container_field, score_key or None)
    if prefer_field and prefer_field in row:
        v = row.get(prefer_field)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            score_key = None
            if prefer_score_key and any(prefer_score_key in d for d in v if isinstance(d, dict)):
                score_key = prefer_score_key
            if not score_key:
                for k in _NUM_KEYS_CAND:
                    if any(isinstance(d.get(k, None), (int,float)) for d in v if isinstance(d, dict)):
                        score_key = k; break
            return prefer_field, score_key
        if is_num_dict(v):
            return prefer_field, None

    for k, v in row.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            score_key = None
            for sk in ([prefer_score_key] if prefer_score_key else []) + _NUM_KEYS_CAND:
                if sk and any(isinstance(d.get(sk, None), (int,float)) for d in v if isinstance(d, dict)):
                    score_key = sk; break
            if score_key:
                return k, score_key
    for k, v in row.items():
        if is_num_dict(v):
            return k, None
    return "", None

def build_struct_map(path: str,
                     override_container: str,
                     override_score_key: str) -> Tuple[
                        Dict[Tuple[str,str], Dict[str,float]],
                        Dict[str, Dict[str,float]],
                        str, Optional[str]
                     ]:
    rows = load_jsonl(path)
    container = None
    score_key = None
    for o in rows:
        container, score_key = detect_struct_layout(o, override_container, override_score_key)
        if container:
            break
    pair_map: Dict[Tuple[str,str], Dict[str,float]] = {}
    tail_map: Dict[str, Dict[str,float]] = {}
    if not container:
        return pair_map, tail_map, "", None

    for o in rows:
        key_str = o.get("Key") or o.get("key") or ""
        tail, rel = split_struct_key(key_str)
        cand: Dict[str,float] = {}
        v = o.get(container)
        if isinstance(v, dict) and not score_key:
            for name, sc in v.items():
                try: cand[str(name)] = float(sc)
                except Exception: pass
        elif isinstance(v, list):
            for d in v:
                if not isinstance(d, dict): continue
                name = d.get("name") or d.get("Name") or d.get("cand") or d.get("candidate")
                if name is None: continue
                sc = d.get(score_key) if score_key else None
                if sc is None:
                    for kk in d.keys():
                        if isinstance(d[kk], (int,float)):
                            sc = d[kk]; break
                if sc is None: continue
                try: cand[str(name)] = float(sc)
                except Exception: continue
        if cand:
            pair_map[(nfkc(tail), nfkc(rel))] = cand
            tail_map[nfkc(tail)] = cand
    return pair_map, tail_map, container, score_key

############################
# ranking helpers
############################
def rank_from_scores(score_dict: Dict[str,float], order: str) -> Dict[str,int]:
    if order == "asc":
        items = sorted(score_dict.items(), key=lambda kv: (kv[1], kv[0]))
    else:
        items = sorted(score_dict.items(), key=lambda kv: (-kv[1], kv[0]))
    return {name: i+1 for i, (name, _) in enumerate(items)}

def rrf_contrib(rank: int, k_rrf: int) -> float:
    return 1.0 / (k_rrf + rank)

############################
# fuse one sample (RRR + flip gate)
############################
def fuse_one(llm_list: List[str],
             struct_rank_maps: List[Dict[str,int]],
             rule_rank_maps: List[Dict[str,int]],
             weights: List[float],
             k_rrf: int,
             expand: int,
             guard_top1: bool,
             max_k: int,
             normalizer,
             # flip gate params (may be None/disabled)
             flip_enable: bool,
             flip_alt_rank_le: int,
             flip_top1_rank_ge: int,
             flip_min_overlap: int,
             flip_margin: float,
             llm_topk_for_tiebreak: int,
             diag: Optional[dict] = None) -> Tuple[List[str], bool, bool]:

    # canon: norm -> preferred raw form (prefer LLM form)
    canon: Dict[str, str] = {}
    llm_norm = [normalizer(x) for x in llm_list]
    for nrm, raw in zip(llm_norm, llm_list):
        if nrm and nrm not in canon:
            canon[nrm] = raw

    def norm_rank_map(m: Dict[str,int]) -> Dict[str,int]:
        out = {}
        for c, r in m.items():
            nrm = normalizer(c)
            if not nrm: continue
            if nrm not in out or r < out[nrm]:
                out[nrm] = r
            if nrm not in canon:
                canon[nrm] = c
        return out

    struct_norm = [norm_rank_map(m) for m in struct_rank_maps]
    rule_norm   = [norm_rank_map(m)   for m in rule_rank_maps]
    channels = [ {n:i+1 for i,n in enumerate(llm_norm)} ] + struct_norm + rule_norm

    # pad / trim weights
    if len(weights) < len(channels):
        pad = weights[-1] if len(weights)>0 else 0.6
        weights = weights + [pad]*(len(channels)-len(weights))
    elif len(weights) > len(channels):
        weights = weights[:len(channels)]

    base_set = set(llm_norm)

    # expansion from struct/rule if requested
    extra_scores: Dict[str,float] = {}
    if expand>0 and (struct_norm or rule_norm):
        for ch_idx, ch in enumerate(channels[1:], start=1):
            w = weights[ch_idx]
            if w == 0: continue
            for c, r in ch.items():
                if c in base_set: continue
                extra_scores[c] = extra_scores.get(c, 0.0) + w * rrf_contrib(r, k_rrf)
    extra_sorted = []
    if extra_scores:
        extra_sorted = [c for c,_ in sorted(extra_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:expand]]

    union = list(llm_norm) + [c for c in extra_sorted if c not in base_set]

    # fused score
    fused: Dict[str,float] = {}
    for c in union:
        s = 0.0
        for w, ch in zip(weights, channels):
            r = ch.get(c)
            if w!=0 and r is not None:
                s += w * rrf_contrib(r, k_rrf)
        fused[c] = s

    order_norm = [c for c,_ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]

    flipped = False
    flipped_to_struct = False

    # ---- FLIP GATE (optional) ----
    if flip_enable and union:
        llm_top1 = llm_norm[0] if llm_norm else None
        # best struct candidate across struct channels
        struct_best = None
        struct_best_rank = 10**9
        struct_union_keys = set()
        for ch in struct_norm:
            struct_union_keys |= set(ch.keys())
            for c, r in ch.items():
                if r < struct_best_rank:
                    struct_best_rank = r
                    struct_best = c

        # structural rank of llm top1 (lower is better)
        llm_top1_struct_rank = 10**9
        for ch in struct_norm:
            r = ch.get(llm_top1, None)
            if r is not None and r < llm_top1_struct_rank:
                llm_top1_struct_rank = r

        # overlap (how many LLM cands appear in struct)
        overlap = len(set(union) & struct_union_keys)

        # candidate to flip to
        alt = struct_best
        # alt must be in union to consider; if not, no flip (需要 expand>0 才可能纳入)
        if alt in union and llm_top1:
            alt_score = fused.get(alt, -1e9)
            top1_score = fused.get(order_norm[0], -1e9)

            # 两条路径满足其一即可：
            ok_by_tiebreak = (alt in llm_norm[:max(1, llm_topk_for_tiebreak)]) \
                              and (struct_best_rank <= flip_alt_rank_le) \
                              and (llm_top1_struct_rank >= flip_top1_rank_ge) \
                              and (overlap >= flip_min_overlap)

            ok_by_margin = (alt_score - top1_score) >= float(flip_margin)

            if ok_by_tiebreak or ok_by_margin:
                # 执行翻盘：把 alt 放到首位
                order_norm = [x for x in order_norm if x != alt]
                order_norm.insert(0, alt)
                flipped = True
                flipped_to_struct = (alt == struct_best)

    # guard top1 (optional)
    if guard_top1 and llm_norm:
        top1 = llm_norm[0]
        if top1 in order_norm:
            order_norm.remove(top1)
            order_norm.insert(0, top1)
            flipped = False
            flipped_to_struct = False

    if max_k and max_k>0:
        order_norm = order_norm[:max_k]

    # diag summary (optional)
    if diag is not None:
        diag["overlap"] = overlap if flip_enable else None
        diag["llm_top1_struct_rank"] = llm_top1_struct_rank if flip_enable else None
        diag["struct_best_rank"] = struct_best_rank if flip_enable else None

    return [canon.get(n, n) for n in order_norm], flipped, flipped_to_struct

############################
# main
############################
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True)
    ap.add_argument("--struct", action="append", default=[])
    ap.add_argument("--rule",   action="append", default=[])
    ap.add_argument("--output", required=True)

    ap.add_argument("--weights", default="2.0,0.6,0.6", help="[LLM, struct..., rule...]")
    ap.add_argument("--k_rrf", type=int, default=8)
    ap.add_argument("--expand", type=int, default=0)
    ap.add_argument("--max_k",  type=int, default=0)
    ap.add_argument("--guard_top1", action="store_true")

    # layout & score direction
    ap.add_argument("--struct_field", default="", help="e.g. Candidates / scores / StructScores")
    ap.add_argument("--struct_score_key", default="", help="e.g. struct / score / prob")
    ap.add_argument("--struct_order", choices=["desc","asc","auto"], default="desc",
                    help="desc=分数大优; asc=分数小优(距离). auto=desc")
    ap.add_argument("--rule_field", default="")
    ap.add_argument("--rule_score_key", default="")
    ap.add_argument("--rule_order", choices=["desc","asc","auto"], default="desc")

    # blend modes & top-k restriction
    ap.add_argument("--blend_mode", choices=["rrf","topk_tiebreak"], default="rrf")
    ap.add_argument("--restrict_struct_topk", type=int, default=0,
                    help=">0 时，仅取每个 struct 通道的前 topk 候选参与")
    ap.add_argument("--tiebreak_topk", type=int, default=3,
                    help="topk_tiebreak/flip 时，LLM 前多少名允许结构打破顺序")

    # flip gate
    ap.add_argument("--flip_enable", action="store_true")
    ap.add_argument("--flip_alt_rank_le", type=int, default=1)
    ap.add_argument("--flip_top1_rank_ge", type=int, default=6)
    ap.add_argument("--flip_min_overlap", type=int, default=2)
    ap.add_argument("--flip_margin", type=float, default=0.0)

    # normalization & debug
    ap.add_argument("--name_norm", default="zhcpp", choices=["cpp","zhcpp","none"])
    ap.add_argument("--debug_stats", action="store_true")
    ap.add_argument("--diag_changed_out", default="", help="如需输出被翻盘样本，给一个 jsonl 路径")

    args = ap.parse_args()
    normalizer = pick_norm(args.name_norm)
    weights = [float(x) for x in re.split(r"[,\s]+", (args.weights or "").strip()) if x]

    llm_rows = load_jsonl(args.input)

    # build struct maps
    struct_pair_maps, struct_tail_maps = [], []
    struct_fields_used, struct_score_keys  = [], []
    for p in (args.struct or []):
        pm, tm, used_field, used_score = build_struct_map(p, args.struct_field, args.struct_score_key)
        struct_pair_maps.append(pm);  struct_tail_maps.append(tm)
        struct_fields_used.append(used_field or ""); struct_score_keys.append(used_score or "")

    # rule maps
    rule_pair_maps, rule_tail_maps = [], []
    rule_fields_used, rule_score_keys = [], []
    for p in (args.rule or []):
        pm, tm, used_field, used_score = build_struct_map(p, args.rule_field, args.rule_score_key)
        rule_pair_maps.append(pm);  rule_tail_maps.append(tm)
        rule_fields_used.append(used_field or ""); rule_score_keys.append(used_score or "")

    out_rows, written = [], 0
    pair_match_rows = 0
    tail_only_rows  = 0
    overlap_rows = 0
    overlap_sum  = 0
    top1_changed = 0
    changed_to_struct = 0

    changed_samples = []

    for o in llm_rows:
        key_pair = key_of_llm(o)
        tail_only = key_pair[0]
        pred = o.get("Prediction") or []
        if not isinstance(pred, list) or not pred:
            out_rows.append(o); written += 1; continue

        # collect struct rank maps
        s_rank_maps = []
        matched_pair = False
        for pm, tm in zip(struct_pair_maps, struct_tail_maps):
            d = pm.get(key_pair)
            if d:
                matched_pair = True
            else:
                d = tm.get(tail_only)
            if d:
                # restrict topk if requested
                ranks = rank_from_scores(d, order=args.struct_order if args.struct_order!="auto" else "desc")
                if args.restrict_struct_topk and args.restrict_struct_topk > 0:
                    # keep only top-k ranks
                    ranks = {c:r for c,r in ranks.items() if r <= args.restrict_struct_topk}
                s_rank_maps.append(ranks)
        if matched_pair: pair_match_rows += 1
        elif s_rank_maps: tail_only_rows += 1

        # rule rank maps
        r_rank_maps = []
        for pm, tm in zip(rule_pair_maps, rule_tail_maps):
            d = pm.get(key_pair) or tm.get(tail_only)
            if d:
                ranks = rank_from_scores(d, order=args.rule_order if args.rule_order!="auto" else "desc")
                r_rank_maps.append(ranks)

        # overlap for diag
        llm_norm_set = set([normalizer(x) for x in pred if x is not None])
        this_overlap = 0
        for m in s_rank_maps: this_overlap += len(llm_norm_set & set(m.keys()))
        for m in r_rank_maps: this_overlap += len(llm_norm_set & set(m.keys()))
        if this_overlap > 0: overlap_rows += 1
        overlap_sum += this_overlap

        # blend
        diag = {}
        new_order, flipped, flipped_to_struct = fuse_one(
            llm_list=pred,
            struct_rank_maps=s_rank_maps,
            rule_rank_maps=r_rank_maps,
            weights=weights[:],
            k_rrf=args.k_rrf,
            expand=args.expand,
            guard_top1=args.guard_top1,
            max_k=args.max_k,
            normalizer=normalizer,
            flip_enable=args.flip_enable,
            flip_alt_rank_le=args.flip_alt_rank_le,
            flip_top1_rank_ge=args.flip_top1_rank_ge,
            flip_min_overlap=args.flip_min_overlap,
            flip_margin=args.flip_margin,
            llm_topk_for_tiebreak=args.tiebreak_topk,
            diag=diag
        )

        # topk_tiebreak（仅在没 flip 的情况下，对 LLM 前K内用结构 rank 打破）
        if args.blend_mode == "topk_tiebreak" and not args.guard_top1 and not flipped:
            K = max(1, args.tiebreak_topk)
            head = new_order[:K]
            tail = new_order[K:]
            # 排序 key：先看“该候选在所有 struct 通道的最小名次（越小越好）”，再看原顺序
            def best_struct_rank(nm: str) -> int:
                rmin = 10**9
                for m in s_rank_maps:
                    r = m.get(normalizer(nm))
                    if r is not None and r < rmin:
                        rmin = r
                return rmin
            head_sorted = sorted(head, key=lambda nm: (best_struct_rank(nm), head.index(nm)))
            new_order = head_sorted + tail

        if new_order != pred:
            old_top1 = pred[0] if pred else None
            new_top1 = new_order[0] if new_order else None
            top1_flip = (old_top1 != new_top1)
            if top1_flip:
                top1_changed += 1
                if s_rank_maps:
                    struct_best_name = None
                    struct_best_rank = 10 ** 9
                    for m in s_rank_maps:
                        for c, r in m.items():
                            if r < struct_best_rank:
                                struct_best_rank = r
                                struct_best_name = c
                    if struct_best_name is not None:
                        changed_to_struct += int(norm_zhcpp(new_top1) == struct_best_name)

            if args.diag_changed_out:
                changed_samples.append({
                    "Key": {"HeadOrTail": key_pair[0], "Rel": key_pair[1]},
                    "BaselineTop1": pred[0] if pred else None,
                    "NewTop1": new_order[0] if new_order else None,
                    "Diag": diag
                })
            o["Prediction"] = new_order

        meta = o.setdefault("Meta", {})
        meta["RRF"] = {
            "k": args.k_rrf,
            "weights": weights[:],
            "expand": args.expand,
            "max_k": args.max_k,
            "guard_top1": bool(args.guard_top1),
            "channels": {"llm": 1, "struct": len(struct_pair_maps), "rule": len(rule_pair_maps)},
            "struct_fields_used": struct_fields_used,
            "struct_cand_score_fields": struct_score_keys,
            "rule_fields_used": rule_fields_used,
            "rule_cand_score_fields": rule_score_keys,
            "name_norm": args.name_norm,
            "struct_order": args.struct_order,
            "rule_order": args.rule_order,
            "blend_mode": args.blend_mode,
            "restrict_struct_topk": args.restrict_struct_topk,
            "tiebreak_topk": args.tiebreak_topk,
            "flip_enable": bool(args.flip_enable),
            "flip_alt_rank_le": args.flip_alt_rank_le,
            "flip_top1_rank_ge": args.flip_top1_rank_ge,
            "flip_min_overlap": args.flip_min_overlap,
            "flip_margin": args.flip_margin,
        }
        out_rows.append(o); written += 1

    dump_jsonl(args.output, out_rows)

    if args.diag_changed_out and changed_samples:
        dump_jsonl(args.diag_changed_out, changed_samples)

    out_meta = {
        "written": written, "in": args.input, "out": args.output,
        "struct_fields_used": struct_fields_used,
        "struct_cand_score_fields": struct_score_keys,
        "rule_fields_used": rule_fields_used,
        "rule_cand_score_fields": rule_score_keys
    }
    if args.debug_stats:
        N = len(llm_rows) if llm_rows else 1
        out_meta.update({
            "pair_match_rows": pair_match_rows,
            "pair_match_rate": pair_match_rows / float(N),
            "tail_only_match_rows": tail_only_rows,
            "tail_only_match_rate": tail_only_rows / float(N),
            "any_struct_match_rate": (pair_match_rows + tail_only_rows) / float(N),
            "cand_overlap_rows": overlap_rows,
            "cand_overlap_rate": overlap_rows / float(N),
            "cand_overlap_avg": overlap_sum / float(N),
            "top1_changed": top1_changed,
            "changed_to_struct_top1": changed_to_struct
        })
    print(json.dumps(out_meta, ensure_ascii=False))

if __name__ == "__main__":
    main()
