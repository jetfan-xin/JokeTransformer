import json
import re
import random
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ========== 配置区域 ==========
REFERENCE_JSON = "eval_30b.json"          # 你保存 Qwen3-30B 那个 JSON 的路径
OUTPUT_JSON = "eval_30b_8b_compare.json"  # 输出的新 JSON 文件路径
MODEL_NAME = "Qwen/Qwen3-8B"
JOKES_PER_COMBO = 20                      # 每个 combo 让 8B 生成多少条
MAX_TOKENS_PER_JOKE = 40                 # 控制长度用
SEED = 42                                 # 随机种子，保证可复现
# =============================


def parse_numbered_list(text: str) -> List[str]:
    """
    把像
    1. joke1
    2) joke2
    这种编号的输出解析成 ["joke1", "joke2", ...]
    """
    lines = text.split("\n")
    jokes = []
    current = []
    pattern = re.compile(r"^\d+[\.\)]\s*(.*)")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if m:
            if current:
                jokes.append(" ".join(current).strip())
            current = [m.group(1)]
        else:
            if current:
                current.append(line)

    if current:
        jokes.append(" ".join(current).strip())

    return jokes


def load_reference_jokes(path: str) -> List[Dict]:
    """
    读取你已经有的 Qwen3-30B-A3B 的 JSON，返回列表：
    [
      {"combo": "...", "joke_text": "..."},
      ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ref_list = data["Qwen3-30B-A3B"]
    return ref_list


def init_qwen3_8b(model_name: str):
    """
    加载 Qwen3-8B 模型 + tokenizer，关闭 thinking 模式时用 chat_template。
    用 torch_dtype="auto" + device_map="auto"（按 HF 官方用法）。
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    return model, tokenizer


def generate_jokes_for_combo(
    combo: str,
    num_jokes: int,
    model,
    tokenizer,
    max_tokens_per_joke: int = 40,
) -> List[str]:
    """
    用 Qwen3-8B 为一个 combo 生成 num_jokes 条笑话。
    返回解析后的 joke 文本列表。
    """
    prompt = (
        f"Write exactly {num_jokes} distinct, very short, funny jokes that "
        f"naturally use the following concept/nouns: [{combo}].\n"
        f"Format: Output a numbered list from 1 to {num_jokes}.\n"
        f"Constraints: No intro, no outro, no explanations. Just the jokes."
    )

    messages = [
        {"role": "system", "content": "You are a creative comedian AI."},
        {"role": "user", "content": prompt},
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        # Qwen3-8B 文档里有 thinking 模式，这里明确关掉以免输出 <think>
        enable_thinking=False,
    )

    inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)

    max_new_tokens = num_jokes * max_tokens_per_joke

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 截去 prompt 部分，只保留新生成的 token
    gen_ids = outputs[0][len(inputs["input_ids"][0]):]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    jokes = parse_numbered_list(text)

    # 容错：如果解析不到编号但整个文本非空，当作一条
    if not jokes and text.strip():
        jokes = [text.strip()]

    return jokes


def main():
    random.seed(SEED)

    # 1. 读取参考的 30B jokes
    ref_list = load_reference_jokes(REFERENCE_JSON)
    print(f"Loaded {len(ref_list)} reference jokes from 30B JSON.")

    # 2. 提取 unique combos
    combos = [item["combo"] for item in ref_list]
    unique_combos = sorted(set(combos))
    print(f"Found {len(unique_combos)} unique combos.")

    # 3. 加载 Qwen3-8B
    model, tokenizer = init_qwen3_8b(MODEL_NAME)

    # 4. 为每个 combo 生成 20 条 8B 笑话
    combo_to_jokes_8b: Dict[str, List[str]] = {}

    for combo in unique_combos:
        print(f"\n=== Generating jokes for combo: {combo!r} ===")
        jokes = generate_jokes_for_combo(
            combo=combo,
            num_jokes=JOKES_PER_COMBO,
            model=model,
            tokenizer=tokenizer,
            max_tokens_per_joke=MAX_TOKENS_PER_JOKE,
        )
        print(f"Generated {len(jokes)} jokes for combo '{combo}'.")
        # 如果不足 20 条，也照样存（后面随机抽时会在现有里抽）
        combo_to_jokes_8b[combo] = jokes

        # 小预览
        for j in jokes[:3]:
            print("  -", j)

    # 5. 构造 Qwen3-8B 的 eval 列表：
    # 对于 ref_list 里的每一条，按 combo 从对应池子里随机抽一条
    eval_8b_list = []
    for item in ref_list:
        combo = item["combo"]
        jokes_pool = combo_to_jokes_8b.get(combo, [])
        if not jokes_pool:
            # 理论上不会发生，除非生成阶段出错
            chosen = "[ERROR: no jokes generated for this combo]"
        else:
            chosen = random.choice(jokes_pool)

        eval_8b_list.append({
            "combo": combo,
            "joke_text": chosen,
        })

    # 6. 写出最终 JSON
    out_data = {
        "Qwen3-30B-A3B": ref_list,   # 原始
        "Qwen3-8B": eval_8b_list,    # 对应的 8B 版本
    }

    Path(OUTPUT_JSON).write_text(
        json.dumps(out_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\nSaved eval JSON to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()