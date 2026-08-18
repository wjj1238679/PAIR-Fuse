# -*- coding: utf-8 -*-
"""构建污染协议下的 RotatE-only / Hard re-rank / Safe-RRF 预测(复用旧结构分文件),
并用旧匹配+含重复计数评测,得到 Table VII 的完整同模型对照。
用法: python dirty_baselines.py
"""
import json, os, sys, subprocess
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from group_metrics import load_triples

CLEAN = os.path.join(HERE, "outputs", "qt_cpp_kg", "clean")

def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return [json.loads(l) for l in f if l.strip()]

def main():
    test = load_triples("test.txt")
    for task in ["tail", "head"]:
        struct_file = os.path.join(HERE, "dataset", "qt_cpp_kg", "rotate",
                                   "O.struct_scores.1280.jsonl" if task == "tail" else "S.struct_scores.1280.jsonl")
        S = {}
        for r in load(struct_file):
            S[r["Key"]] = r
        llm = load(os.path.join(CLEAN, f"dirty_{task}_P1_minimax.jsonl"))
        rot_out, hard_out = [], []
        for i, (h, r, t) in enumerate(test):
            if i >= len(llm):
                break
            key = f"{h}\t{r}" if task == "tail" else f"{t}\t{r}"
            s = S.get(key)
            if s is None:
                continue
            q = llm[i]["Question"]
            struct_order = [c["name"] for c in sorted(s["Candidates"], key=lambda x: -x["struct"])]
            rot_out.append({"Question": q, "Prediction": struct_order})
            top = struct_order[:1]
            hard = top + [x for x in llm[i]["Prediction"] if x not in top]
            hard_out.append({"Question": q, "Prediction": hard})
        with open(os.path.join(CLEAN, f"dirty_rotateonly_{task}.jsonl"), "w", encoding="utf-8") as w:
            for r_ in rot_out:
                w.write(json.dumps(r_, ensure_ascii=False) + "\n")
        with open(os.path.join(CLEAN, f"dirty_hard_{task}.jsonl"), "w", encoding="utf-8") as w:
            for r_ in hard_out:
                w.write(json.dumps(r_, ensure_ascii=False) + "\n")
        print(task, "rotate/hard written:", len(rot_out))
        # Safe-RRF(guard) 融合: dirty LLM 输出 + 旧结构分
        subprocess.run([sys.executable, "fuse_llm_struct_rule.py",
                        "--input", os.path.join(CLEAN, f"dirty_{task}_P1_minimax.jsonl"),
                        "--struct", struct_file,
                        "--output", os.path.join(CLEAN, f"dirty_safe_{task}.jsonl"),
                        "--k_rrf", "8", "--weights", "2.0,0.10",
                        "--blend_mode", "topk_tiebreak", "--restrict_struct_topk", "1",
                        "--tiebreak_topk", "3", "--guard_top1"],
                       cwd=HERE, capture_output=True)
        print(task, "safe fused")

if __name__ == "__main__":
    main()
