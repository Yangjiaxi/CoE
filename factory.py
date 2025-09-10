from pathlib import Path

DATA_ROOT = Path("data/processed/")

data_lut = {
    # General (from Tulu tagged)
    # "tulu_g.tech": str(DATA_ROOT / "tulu-tagged/g.tech-81324.jsonl"),
    # "tulu_g.stem": str(DATA_ROOT / "tulu-tagged/g.tech-81324.jsonl"),
    # "tulu_g.human": str(DATA_ROOT / "tulu-tagged/g.human-130803.jsonl"),
    "tulu_g.growth": str(DATA_ROOT / "tulu-tagged/g.growth-56249.jsonl"),
    # Coding
    "tulu_coding": str(DATA_ROOT / "tulu-tagged/coding-32580.jsonl"),
    "code_alpaca": str(DATA_ROOT / "code_alpaca-20016.jsonl"),
    "oss_instruct": str(DATA_ROOT / "ossinstruct-full-75197.jsonl"),
    # Math
    "tulu_math": str(DATA_ROOT / "tulu-tagged/math-13645.jsonl"),
    "mammoth_cot_no_aqua": str(DATA_ROOT / "mammoth/cot-no_aqua-99143.jsonl"),
    ## Multilingual
    # "tulu_multilingual": str(DATA_ROOT / "tulu-tagged/multilingual-54336.jsonl"),
}

# Example domain combinations for expert training
EXPERT_DATA_PRESETS = {
    "math": "tulu_math-mammoth_cot_no_aqua",
    "coding": "tulu_coding-code_alpaca-oss_instruct",
    "general": "tulu_g.growth",
}

if __name__ == "__main__":

    def count_file_lines(file):
        with open(file, "r") as f:
            return sum(1 for _ in f)

    for domain, details in EXPERT_DATA_PRESETS.items():
        print(f"{domain}: {details}")
        data_list = details.split("-")
        data_list = [data_lut[e] for e in data_list]

        total_lines = sum(count_file_lines(e) for e in data_list)
        print(f"\tTotal lines: {total_lines}")
