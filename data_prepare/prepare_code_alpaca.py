import json
from pathlib import Path

from tqdm import tqdm


def build_user_content(instruction: str, input_text: str) -> str:
    """Combine instruction and input as user message content."""
    if input_text and input_text.strip():
        content = instruction.strip() + "\n" + input_text.strip()
    else:
        content = instruction.strip()
    return (
        "Write a response that appropriately completes the request.\n\n### Instruction:\n"
        + content
        + "\n\n### Response:"
    )


def chat_format(ex: dict, idx: int) -> dict:
    instruction = ex.get("instruction", "")
    input_text = ex.get("input", "")
    output = ex.get("output", "")
    user_content = build_user_content(instruction, input_text)
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]
    return {
        "dataset": "code_alpaca",
        "id": f"code_alpaca_{idx}",
        "messages": messages,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to code_alpaca_20k.json (JSON list of dicts)",
    )
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.input).open("r") as f:
        raw = json.load(f)

    data = []
    for idx, ex in tqdm(enumerate(raw), total=len(raw)):
        data.append(chat_format(ex, idx))

    output_file = output_dir / f"code_alpaca-{len(data)}.jsonl"
    with output_file.open("w") as f:
        for rec in data:
            washed = {"dataset": rec["dataset"], "id": rec["id"], "messages": rec["messages"]}
            f.write(json.dumps(washed, indent=None, sort_keys=False, ensure_ascii=False) + "\n")
    print(f"\tSaved => {output_file}")
