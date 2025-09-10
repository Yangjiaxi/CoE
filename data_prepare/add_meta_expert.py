import argparse
import json
from pathlib import Path

from tqdm import tqdm


def add_meta_fields(input_path: Path, output_path: Path, meta_tag: str, data_name: str) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r") as f_in, output_path.open("w") as f_out:
        for line in tqdm(f_in, desc=str(input_path.name)):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            new_d = {"meta": meta_tag, "data": data_name, **d}
            f_out.write(json.dumps(new_d, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Add meta-expert fields to JSONL for MoE router supervision.")
    parser.add_argument("--meta_config", type=str, required=True)
    parser.add_argument("--source_base", type=str, required=True)
    parser.add_argument("--output_base", type=str, required=True)
    parser.add_argument("--cwd", type=str, default=".")
    args = parser.parse_args()

    cwd = Path(args.cwd).resolve()
    source_base = Path(args.source_base).resolve()
    output_base = Path(args.output_base).resolve()

    with open(args.meta_config) as f:
        meta_tagging = json.load(f)

    total = 0
    for meta_tag, name_to_file in meta_tagging.items():
        for data_name, file_path in name_to_file.items():
            input_path = (cwd / file_path).resolve()
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")

            try:
                stem = input_path.relative_to(source_base)
            except ValueError:
                raise ValueError(f"Input path {input_path} is not under source_base {source_base}")

            output_path = output_base / stem
            n = add_meta_fields(input_path, output_path, meta_tag, data_name)
            total += n
            print(f"=> {output_path} Done ({n} records)")

    print(f"Total: {total} records processed.")


if __name__ == "__main__":
    main()
