# -*- coding: utf-8 -*-
# export_struct_scores_from_rotate.py
# 依据 LLM 预测逐行计算 RotatE 结构分，兼容 head/tail 两个任务。
# 输出：每行 JSON 形如：
# { "Key": "Head\tRelation" (tail任务) 或 "Tail\tRelation" (head任务),
#   "HeadEntity"/"TailEntity": "...",
#   "Relation": "...",
#   "Candidates": [ {"name": "...", "struct": 0~1}, ... ] }

import argparse, json, os, re, unicodedata, codecs
from typing import List, Dict, Optional
import torch
import torch.nn as nn

def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", str(s or ""))

def canon_name(x: str) -> str:
    s = nfkc(x).lower().strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("recrod", "record")
    s = re.sub(r"(类|類|class|类型|類型)$", "", s)
    s = re.sub(r"[^0-9a-z_]+", "", s)
    return s

def load_jsonl_any(path: str):
    rows = []
    with open(path, "rb") as fb:
        data = fb.read()
    txt = None
    for enc in ("utf-8-sig","utf-8","gb18030","cp936","latin-1"):
        try:
            txt = data.decode(enc); break
        except UnicodeDecodeError:
            continue
    if txt is None:
        txt = data.decode("utf-8", errors="ignore")
    for ln in txt.splitlines():
        s = ln.strip()
        if not s: continue
        try:
            rows.append(json.loads(s))
        except Exception:
            m = re.search(r"\{.*\}", s)
            if m:
                try: rows.append(json.loads(m.group(0)))
                except Exception: pass
    return rows

def read_relations(ds_dir: Optional[str]) -> List[str]:
    if not ds_dir: return []
    rels = []
    for p in (os.path.join(ds_dir, "relation2id.txt"),
              os.path.join(ds_dir, "get_neighbor", "relation2id.txt")):
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8-sig") as f:
                for ln in f:
                    s = ln.strip()
                    if not s: continue
                    cols = re.split(r"\t+", s)
                    if cols: rels.append(cols[0])
    out, seen = [], set()
    for r in rels:
        if r not in seen:
            seen.add(r); out.append(r)
    return out

def extract_rel_from_question(question: str, known_rels: List[str]) -> str:
    q = nfkc(question)
    qc = re.sub(r"\s+", "", q).lower()
    best, bestlen = "", -1
    for r in known_rels or []:
        rc = re.sub(r"\s+", "", nfkc(r)).lower()
        if rc and rc in qc and len(rc) > bestlen:
            best, bestlen = r, len(rc)
    if best: return best
    m = re.search(r'(/[^/\[\]]+(?:/[^/\[\]]+)*)\s*\[MASK\]', q)
    if m: return m.group(1).strip()
    pre = re.split(r'\[MASK\]', q, 1)[0] if "[MASK]" in q else q
    toks = re.split(r'[\s，,。；;]+', pre.strip())
    return toks[-1] if toks else ""

# ---------- RotatE ----------
class NameIndexer:
    def __init__(self, names: List[str]):
        self.names = list(names)
        self.name2id = {n:i for i,n in enumerate(self.names)}
        self.canon2id = {}
        for i,n in enumerate(self.names):
            c = canon_name(n)
            if c and c not in self.canon2id:
                self.canon2id[c] = i
    def get_id(self, name: str):
        if name in self.name2id: return self.name2id[name]
        return self.canon2id.get(canon_name(name), None)

class RotatE(nn.Module):
    def __init__(self, n_ent: int, n_rel: int, dim: int, gamma: float = 12.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim // 2
        self.gamma = gamma
        self.ent = nn.Embedding(n_ent, self.dim * 2)
        self.rel = nn.Embedding(n_rel, self.dim)
    def score_triples(self, h, r, t):
        he = self.ent(h); te = self.ent(t); rr = self.rel(r)
        h_re, h_im = he[..., :self.dim], he[..., self.dim:]
        t_re, t_im = te[..., :self.dim], te[..., self.dim:]
        r_re, r_im = torch.cos(rr), torch.sin(rr)
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        diff = torch.abs(hr_re - t_re) + torch.abs(hr_im - t_im)
        return self.gamma - torch.sum(diff, dim=-1)

def load_rotate_ckpt(ckpt_path: str, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    ents, rels = ckpt["ents"], ckpt["rels"]
    dim = int(ckpt["dim"])
    model = RotatE(len(ents), len(rels), dim=dim).to(device)
    model.ent.load_state_dict(ckpt["ent"])
    model.rel.load_state_dict(ckpt["rel"])
    model.eval()
    return model, NameIndexer(ents), NameIndexer(rels)

def minmax01(x):
    if not x: return x
    mn, mx = float(min(x)), float(max(x))
    if mx <= mn: return [0.0 for _ in x]
    return [(float(v)-mn)/(mx-mn) for v in x]

def score_tail_given_head(model, ent_ix, rel_ix,
                          head_name: str, rel_name: str,
                          cand_tails: List[str], device="cpu"):
    hid = ent_ix.get_id(head_name)
    rid = rel_ix.get_id(rel_name)
    if hid is None or rid is None:
        return [0.0 for _ in cand_tails]
    tids, ok = [], []
    for nm in cand_tails:
        tid = ent_ix.get_id(nm)
        if tid is None:
            tids.append(0); ok.append(False)
        else:
            tids.append(tid); ok.append(True)
    with torch.no_grad():
        if not tids:
            return []
        h = torch.full((len(tids),), hid, dtype=torch.long, device=device)
        r = torch.full((len(tids),), rid, dtype=torch.long, device=device)
        t = torch.tensor(tids, dtype=torch.long, device=device)
        s = model.score_triples(h, r, t).tolist()
    s = [sv if flag else -1e9 for sv, flag in zip(s, ok)]
    return minmax01(s)

def score_head_given_tail(model, ent_ix, rel_ix,
                          tail_name: str, rel_name: str,
                          cand_heads: List[str], device="cpu"):
    tid = ent_ix.get_id(tail_name)
    rid = rel_ix.get_id(rel_name)
    if tid is None or rid is None:
        return [0.0 for _ in cand_heads]
    hs, ok = [], []
    for nm in cand_heads:
        hid = ent_ix.get_id(nm)
        if hid is None:
            hs.append(0); ok.append(False)
        else:
            hs.append(hid); ok.append(True)
    if not hs:
        return []
    with torch.no_grad():
        h = torch.tensor(hs, dtype=torch.long, device=device)
        r = torch.full((len(hs),), rid, dtype=torch.long, device=device)
        t = torch.full((len(hs),), tid, dtype=torch.long, device=device)
        s = model.score_triples(h, r, t).tolist()
    s = [sv if flag else -1e9 for sv, flag in zip(s, ok)]
    return minmax01(s)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["head","tail"], required=True)
    ap.add_argument("--pred", required=True, help="LLM 排序 jsonl（含 HeadEntity/Question/Key 等）")
    ap.add_argument("--ckpt", required=True, help="RotatE ckpt 路径（如 dataset\\qt_cpp_kg\\rotate\\rotate_qt_dim1280_recip.pt）")
    ap.add_argument("--out",  required=True, help="输出 JSONL （会覆盖）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dataset_dir", default="", help="可选；用于更稳地从 Question 识别 Relation")
    args = ap.parse_args()

    known_rels = read_relations(args.dataset_dir) if args.dataset_dir else []

    model, ent_ix, rel_ix = load_rotate_ckpt(args.ckpt, device=args.device)
    rows = load_jsonl_any(args.pred)

    wrote = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with codecs.open(args.out, "w", "utf-8") as w:
        for obj in rows:
            rel_field = obj.get("Relation","") or ""
            q = obj.get("Question","")
            cand = obj.get("Prediction", [])
            if not isinstance(cand, list) or not cand:
                cands2 = obj.get("Candidates", [])
                cand = [c.get("name","") for c in cands2 if isinstance(c,dict) and c.get("name")]
            if not cand:
                continue

            if args.task == "tail":
                # 给定 (head, rel)，打分 tail
                head = obj.get("HeadEntity","") or obj.get("Head","")
                reln = rel_field or extract_rel_from_question(q, known_rels)
                scores = score_tail_given_head(model, ent_ix, rel_ix, head, reln, cand, device=args.device)
                # 重要：tail 任务把 head 自身候选的分数压到 0
                hc = canon_name(head)
                items = []
                for nm, sc in zip(cand, scores):
                    items.append({"name": nm, "struct": (0.0 if canon_name(nm)==hc else float(sc))})
                out = {"Key": f"{head}\t{reln}", "HeadEntity": head, "Relation": reln, "Candidates": items}

            else:
                # 给定 (tail, rel)，打分 head
                tail = obj.get("HeadEntity","") or obj.get("TailEntity","") or obj.get("Tail","")  # 兼容你的历史字段
                reln = rel_field or extract_rel_from_question(q, known_rels)
                scores = score_head_given_tail(model, ent_ix, rel_ix, tail, reln, cand, device=args.device)
                out = {"Key": f"{tail}\t{reln}", "TailEntity": tail, "Relation": reln,
                       "Candidates": [{"name": nm, "struct": float(sc)} for nm, sc in zip(cand, scores)]}

            w.write(json.dumps(out, ensure_ascii=False) + "\n")
            wrote += 1

    print(json.dumps({"wrote": wrote, "out": args.out}, ensure_ascii=False))

if __name__ == "__main__":
    main()
