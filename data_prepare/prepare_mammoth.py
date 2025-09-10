import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def chat_format(ex, sign):
    messages = [
        {"role": "user", "content": ex["instruction"]},
        {"role": "assistant", "content": ex["output"]},
    ]
    return {
        "dataset": "TIGER-Lab/MathInstruct",
        "id": sign,
        "source": ex["source"],
        "messages": messages,
    }


def find_category(ex):
    source = ex["source"]
    _, category, name = source.split("/")
    name = name.split(".json")[0]
    category = category.strip().lower()
    return category, name


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to mammoth JSON file (list of dicts with source/Instruction/output)",
    )
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    TARGET_DATA = "cot"
    TARGET_SUFFIX = "cot-no_aqua"  # exclude aqua sources by default

    output_dir = Path(args.output_dir) / "mammoth"
    output_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.input).open("r") as f:
        raw = json.load(f)

    data = []
    source_counter = defaultdict(lambda: 0)

    for idx, ex in tqdm(enumerate(raw), total=len(raw)):
        cat, name = find_category(ex)
        data_sign = f"mammoth-{idx}_{cat}-{name}"
        if cat == TARGET_DATA and "aqua" not in name:
            source_counter[name] += 1
            data.append(chat_format(ex, data_sign))

    assert len(data) > 0, f"No {TARGET_SUFFIX} data found in {args.input}"

    output_file = output_dir / f"{TARGET_SUFFIX}-{len(data)}.jsonl"
    with output_file.open("w") as f:
        for l in data:
            washed = {"dataset": l["dataset"], "id": l["id"], "messages": l["messages"]}
            f.write(f"{json.dumps(washed, indent=None, sort_keys=False, ensure_ascii=False)}\n")
    print(f"\tSaved => {output_file}")
