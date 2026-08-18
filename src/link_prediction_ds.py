# -*- coding: utf-8 -*-
import argparse, json, logging, os, re, time, sys
from collections import defaultdict
from typing import List, Dict, Tuple


def _tok_key_for_model(m):
    try:
        m = str(m).lower()
    except Exception:
        m = str(m or "").lower()
    return "max_completion_tokens" if m.startswith("gpt-5") else "max_tokens"


import openai
from tqdm import tqdm

# ----------------------------
# 内置中文 prompt（作为兜底）
# ----------------------------
PROMPT_BUILTIN = {
    "init_query": (
        "你是一个专门为 Qt/C++ 教学知识图谱做链路预测排序的助手。任务：根据给定的问题（含 [MASK]）"
        "与候选实体列表（JSON 数组，含 name/desc），将候选实体按成为真实答案的可能性排序。\n"
        "请只在心中完成以下步骤：\n"
        "1) 基于关系语义做类型与约束判断；\n"
        "2) 逐步排除不匹配候选；\n"
        "3) 对剩余候选按关键证据细粒度对比并自检。\n"
        "严格规则：\n"
        "1) 只能使用候选中的 name；\n"
        "2) 最终只输出 JSON：{{\"final_answer\":[\"候选名1\",\"候选名2\", ...]}}；\n"
        "3) 必须包含并排序所有候选（顺序=可能性由高到低）；\n"
        "4) 严禁输出任何解释或额外文本。\n"
        "如果理解，请仅回答 Yes。\n"
    ),
    "first_give_demonstration": "目标问题是: {question}。面会给示例与补充事实。请在心中总结筛选规则，不要输出任何内容",
    "analogy_demonstration": "类比示例: {selected_analogy_demonstrations}。请在心中总结关系语义、正反例特征与易混边界。不要输出。",
    "supplement_demonstration": "补充事实: {selected_supplement_demonstrations}。请在心中提取用于筛除/对比的线索。不要输出。",
    "final_query_template": (
        "候选实体 (JSON 数组):\n{order_of_candidate}\n\n"
        "问题:\n{question}\n\n"
        "在心中完成：①类型一致性与关系约束粗筛；②逐步排除不匹配；③对剩余候选做基于证据的细粒度对比；④自检（是否遗漏或顺序反转）。\n"
        "硬规则：若候选名仅由符号/标点组成（含三字符组 ??=、??/ 等），或明显属于预处理宏/运算符，在本题未明确涉及该类关系时，必须整体放在序列末尾。\n"
        "请严格输出 JSON，列表长度必须等于上方候选的数量：{\"final_answer\": [\"候选名1\",\"候选名2\", ...]}。\n"
        "每个元素必须与候选的 name 完全一致（逐字复制）；不得包含任何额外字符、解释或换行。"
    ),
    "directly_ask": (
        "候选实体 (JSON 数组):\n{order_of_candidate}\n\n"
        "问题:\n{question}\n\n"
        "请严格输出 JSON：{\"final_answer\": [\"候选名1\",\"候选名2\", ...]}。\n"
        "只能使用候选中的 name，必须包含所有候选。禁止输出解释。"
    )
}


# ----------------------------
# 1) API KEY 处理
# ----------------------------
def normalize_api_keys(arg_key: str, num_proc: int) -> List[str]:
    def looks_like_key(s: str) -> bool:
        return s.startswith("sk-") or s.startswith("sk-proj-")

    if looks_like_key(arg_key):
        return [arg_key] * num_proc
    if not os.path.exists(arg_key):
        raise FileNotFoundError(f"API key file not found: {arg_key}")
    with open(arg_key, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    keys = [ln for ln in lines if looks_like_key(ln)]
    if not keys:
        raise ValueError(f"No valid API keys found in file: {arg_key}")
    if len(keys) == 1 and num_proc > 1:
        keys = keys * num_proc
    if len(keys) != num_proc:
        raise ValueError(f"#keys ({len(keys)}) != num_process ({num_proc}). ")
    return keys


# ----------------------------
# 4) I/O 与主函数 (精简了旧版用不到的类，直接放主体逻辑)
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--query", type=str, choices=["tail", "head"], default="head")
    ap.add_argument("--candidate_num", type=int, default=12)
    ap.add_argument("--prompt_path", type=str, default="prompts/link_prediction.json")
    ap.add_argument("--prompt_name", type=str, default="chat")
    ap.add_argument("--openai_model", type=str, default="deepseek-chat")  # 默认换成 deepseek
    ap.add_argument("--openai_timeout", type=int, default=120)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--max_api_retries", type=int, default=4)
    ap.add_argument("--num_process", type=int, default=1)
    ap.add_argument("--api_key", type=str, required=True)

    # [新增] DeepSeek Base URL 参数
    ap.add_argument("--base_url", type=str, default="https://api.deepseek.com", help="API base URL for DeepSeek")

    ap.add_argument("--samples", type=str, default="", help="可选：显式指定样本文件")
    ap.add_argument("--out", type=str, default="", help="可选：显式指定输出文件")

    return ap.parse_args()


def main_single(args, api_key: str):
    import os, json, time, re, sys
    from typing import List, Dict

    def pick(ex: dict, keys, default=None):
        if not isinstance(keys, (list, tuple)): keys = [keys]
        for k in keys:
            if k in ex: return ex[k]
            for kk in ex.keys():
                if isinstance(kk, str) and kk.lower() == str(k).lower(): return ex[kk]
        return default

    def to_candidates(raw):
        if raw is None: return []
        if isinstance(raw, dict): raw = [raw]
        out = []
        if isinstance(raw, list):
            for it in raw:
                if isinstance(it, dict):
                    name = pick(it, ["name", "Name", "label", "Label", "title", "Title"], "")
                    desc = pick(it, ["desc", "Desc", "description", "text", "Text", "Description"], "")
                    name = (name or "").strip()
                    if not name and "final_answer" in it and isinstance(it["final_answer"], str):
                        name = it["final_answer"].strip()
                    if name: out.append({"name": name, "desc": str(desc or "")})
                elif isinstance(it, str):
                    nm = it.strip()
                    if nm: out.append({"name": nm, "desc": ""})
        return out

    samples_path = getattr(args, "samples", "")
    out_path = getattr(args, "out", "")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def load_prompt_templates(prompt_path: str, prompt_name: str) -> Dict:
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if prompt_name in obj: return obj[prompt_name]
            if "chat" in obj and prompt_name == "chat": return obj["chat"]
            return obj
        except:
            return PROMPT_BUILTIN

    templates = load_prompt_templates(args.prompt_path, args.prompt_name)

    def build_messages(templates: Dict, question: str, candidates: List[Dict]) -> List[Dict]:
        def render_cands(cands):
            return json.dumps([{"name": c.get("name", ""), "desc": c.get("desc", "")} for c in cands],
                              ensure_ascii=False, indent=2)

        init_query_raw = templates.get("init_query", "请仅回答 Yes。")
        final_tmpl_raw = templates.get("final_query_template", PROMPT_BUILTIN["final_query_template"])
        SAFE_OC, SAFE_Q = "<<<ORDER_OF_CANDIDATE>>>", "<<<QUESTION>>>"
        tmp = final_tmpl_raw.replace("{order_of_candidate}", SAFE_OC).replace("{question}", SAFE_Q)
        tmp = tmp.replace("{", "{{").replace("}", "}}")
        final_tmpl_safe = tmp.replace(SAFE_OC, "{order_of_candidate}").replace(SAFE_Q, "{question}")
        final_query = final_tmpl_safe.format(order_of_candidate=render_cands(candidates), question=question)

        messages = [
            {"role": "system", "content": "你是一个专门为 Qt/C++ 教学知识图谱做链路预测排序的助手。请严格输出JSON。"},
            {"role": "user", "content": init_query_raw},
            {"role": "assistant", "content": "Yes"},
            {"role": "user", "content": final_query}
        ]
        return messages

    # ---------- 完美适配 DeepSeek API ----------
    USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def call_llm(messages: List[Dict], model: str, max_output_tokens: int, timeout_s: int = 60):
        import openai
        base_url = getattr(args, "base_url", "https://api.deepseek.com")
        kwargs = dict(model=model, messages=messages, temperature=0.0, max_tokens=max_output_tokens)

        try:
            # 兼容 openai >= 1.0.0
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url)
            resp = client.chat.completions.create(**kwargs, timeout=timeout_s)
            u = getattr(resp, "usage", None)
            if u is not None:
                USAGE["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
                USAGE["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
                USAGE["calls"] += 1
            return resp.choices[0].message.content or ""
        except ImportError:
            # 兼容 openai < 1.0.0 旧版本
            openai.api_key = api_key
            openai.api_base = base_url if base_url.endswith("/v1") else base_url.rstrip("/") + "/v1"
            resp = openai.ChatCompletion.create(**kwargs, request_timeout=timeout_s)
            return resp["choices"][0]["message"]["content"] or ""

    def call_llm_with_retry(messages, model, max_output_tokens, n_retry=3):
        import time
        for i in range(n_retry):
            try:
                return call_llm(messages, model, max_output_tokens, timeout_s=getattr(args, "openai_timeout", 60))
            except Exception as e:
                wait = 2.0 * (i + 1)
                print(f"[warn] LLM 调用失败 {i + 1}/{n_retry}: {e}. {wait:.1f}s 后重试...", file=sys.stderr)
                time.sleep(wait)
        raise RuntimeError("LLM 多次重试仍失败")

    def parse_final_order(text: str, candidate_names: List[str]) -> List[str]:
        # 剥离推理模型的 <think>...</think> 段, 避免其中的 JSON 样例干扰解析
        text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
        def try_json(s):
            try:
                obj = json.loads(s)
                if isinstance(obj, dict) and isinstance(obj.get("final_answer"), list):
                    return [str(x) for x in obj["final_answer"]]
            except Exception:
                return None

        ans = try_json(text)
        if ans is None:
            m = re.search(r"\{.*\}", text, flags=re.S)
            if m: ans = try_json(m.group(0))
        if ans is None: ans = []
        cand_set = set(candidate_names)
        cleaned = []
        for x in ans:
            x2 = (x or "").strip()
            if x2 in cand_set and x2 not in cleaned: cleaned.append(x2)
        for x in candidate_names:
            if x not in cleaned: cleaned.append(x)
        return cleaned

    total, t0 = 0, time.time()

    # --- 断点续跑 + 增量写盘(防止进程被杀后结果全丢) ---
    done_questions = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8-sig") as rf:
            for ol in rf:
                ol = ol.strip()
                if not ol:
                    continue
                try:
                    done_questions.add(json.loads(ol).get("Question", ""))
                except Exception:
                    pass
        if done_questions:
            print(f"[resume] 输出文件已有 {len(done_questions)} 条, 跳过这些查询", file=sys.stderr)
    out_f = open(out_path, "a", encoding="utf-8")

    def emit(row):
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.flush()

    with open(samples_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line and line[0] == "\ufeff": line = line[1:]
            line = line.strip()
            if not line: continue

            ex = None
            if line.startswith("{") and line.endswith("}"):
                try:
                    ex = json.loads(line)
                except Exception:
                    ex = None
            if ex is None: continue

            head = pick(ex, ["HeadEntity", "Head", "head", "h"], "")
            question = pick(ex, ["Question", "question", "text", "query", "q"], "")
            cands = to_candidates(pick(ex, ["Candidates", "candidates", "candidateEntities", "order_of_candidate"]))

            if not cands: cands = to_candidates(pick(ex, ["final_answer", "names", "entities"]))

            if not cands:
                continue  # 数据不完整跳过

            if question in done_questions:
                continue  # 断点续跑: 已完成

            wrote = False
            cand_names = [c["name"] for c in cands if c.get("name")]

            try:
                messages = build_messages(templates, question, cands)
                rsp = call_llm_with_retry(messages, args.openai_model, getattr(args, "max_tokens", 512))
                final_order = parse_final_order(rsp, cand_names)
                positional_scores = {name: 1.0 / (i + 1) for i, name in enumerate(final_order)}

                emit({
                    "HeadEntity": head,
                    "Question": question,
                    "Prediction": final_order,
                    "LLMSignals": {
                        "top_order": final_order,
                        "positional_scores": positional_scores,
                        "meta": {
                            "prompt_name": getattr(args, "prompt_name", ""),
                            "model": getattr(args, "openai_model", ""),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    }
                })
                wrote = True
            except Exception as e:
                fallback = cand_names
                emit({
                    "HeadEntity": head, "Question": question, "Prediction": fallback,
                    "LLMSignals": {"top_order": fallback,
                                   "positional_scores": {n: 1.0 / (i + 1) for i, n in enumerate(fallback)},
                                   "meta": {"fallback": True, "error": str(e)[:200]}}
                })
                wrote = True
            finally:
                if wrote: total += 1

            if total % 10 == 0:
                dt = time.time() - t0
                print(f"[link_prediction DS] processed={total}, avg={dt / total:.3f}s/obj", file=sys.stderr)

    out_f.close()
    print(
        json.dumps({"samples": total, "out": out_path, "elapsed_sec": round(time.time() - t0, 3),
                    "usage": USAGE}, ensure_ascii=False))


if __name__ == "__main__":
    args = parse_args()
    keys = normalize_api_keys(args.api_key, args.num_process)
    main_single(args, keys[0])