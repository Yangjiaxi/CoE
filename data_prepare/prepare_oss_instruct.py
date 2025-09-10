import json
from collections import defaultdict, Counter
from pathlib import Path

from tqdm import tqdm


def load_jsonl_iter(file, limit=None):
    with Path(file).open("r") as f:
        for idx, line in enumerate(f):
            if limit and idx > limit:
                return
            if line.strip():
                yield json.loads(line.strip())


def chat_format(ex, sign):
    messages = [
        {"role": "user", "content": ex["problem"]},
        {"role": "assistant", "content": ex["solution"]},
    ]
    return {
        "dataset": "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "id": sign,
        "messages": messages,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to ossinstruct raw JSONL (from HuggingFace)")
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = []
    lang_cnt = Counter()
    for idx, ex in tqdm(enumerate(load_jsonl_iter(args.input))):
        lang = ex.get("lang", "unknown")
        lang_cnt[lang] += 1
        data_sign = f"ossinstruct-{idx}_{lang}"
        data.append(chat_format(ex, data_sign))

    output_file = output_dir / f"ossinstruct-full-{len(data)}.jsonl"
    with output_file.open("w") as f:
        for l in data:
            washed = {"dataset": l["dataset"], "id": l["id"], "messages": l["messages"]}
            f.write(f"{json.dumps(washed, indent=None, sort_keys=False, ensure_ascii=False)}\n")
    print(f"\tSaved => {output_file}")
