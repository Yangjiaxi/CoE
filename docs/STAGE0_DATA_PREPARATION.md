# Stage 0: Data Preparation

Classify labeled data into **math**, **code**, or **general** domains for expert training.

## Data Sources & Scripts

- **Math**: Tulu (math-tagged), MathInstruct (Mammoth)
- **Code**: Tulu (coding-tagged), Code Alpaca, OSS-Instruct
- **General**: Tulu (general-tagged: STEM, Humanities, Growth)

## Run Commands

**Tulu Tagged**
```bash
python data_prepare/prepare_tagged_tulu.py --input data/raw/tulu-parsed-297507.jsonl --subject_mapping data/raw/subject_mapping.jsonl
```

**Code Alpaca**
```bash
curl -o data/raw/code_alpaca_20k.json https://raw.githubusercontent.com/sahil280114/codealpaca/refs/heads/master/data/code_alpaca_20k.json
python data_prepare/prepare_code_alpaca.py --input data/raw/code_alpaca_20k.json --output_dir data/processed
```

**OSS-Instruct**
```bash
wget "https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K/resolve/main/data-oss_instruct-decontaminated.jsonl?download=true" -O data/raw/oss_instruct-decontaminated.jsonl
python data_prepare/prepare_oss_instruct.py --input data/raw/oss_instruct-decontaminated.jsonl --output_dir data/processed
```

**Mammoth (MathInstruct)**
```bash
wget "https://huggingface.co/datasets/TIGER-Lab/MathInstruct/resolve/main/MathInstruct.json?download=true" -O data/raw/MathInstruct.json
python data_prepare/prepare_mammoth.py --input data/raw/MathInstruct.json --output_dir data/processed
```

## Output Format

JSONL with fields: `dataset`, `id`, `messages` (user/assistant turns).

For Stage 3 router supervision, add `meta` via `add_meta_expert.py`:
```bash
python data_prepare/add_meta_expert.py --meta_config meta_config.json --source_base data/processed --output_base data/with_meta
```

`meta` values: `math`, `coding`, `general_growth` (must match expert names).

## Config

Set `DATA_ROOT` in `factory.py` to your data directory. `data_lut` maps short names (e.g. `tulu_math`, `mammoth_cot`) to full paths.
