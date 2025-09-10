import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm


def load_jsonl_iter(file, limit=None):
    with Path(file).open("r") as f:
        for idx, line in enumerate(f):
            if limit and idx > limit:
                return
            if line.strip():
                yield json.loads(line.strip())


def left_shrink(value, permitted):
    if ":" not in value:
        return False, ""
    while value.split(":", 1)[0].strip().lower() not in permitted:
        if ":" not in value:
            return False, value
        value = value.split(":", 1)[-1].strip()
    return True, value


def extract_languages(text):
    outside_text = re.search(r"^([^(]*)", text).group(1).strip()
    inside_text = re.search(r"\(([^)]*)\)", text).group(1).split(",")
    inside_text = [language.strip() for language in inside_text]
    return outside_text, inside_text


def parse_response(response, ex):
    """Parse GPT tagging response into structured fields."""
    lines = response.split("\n")
    parsed_data = {}
    permitted_keys = [
        "math_topic",
        "difficulty_level",
        "programming_language",
        "task_type",
        "subject",
        "conversation_type",
    ]
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            key, value = key.strip(), value.strip()
            if key == "SPOKEN_LANGUAGE":
                if "(" in value or ")" in value:
                    outside, inside_list = extract_languages(value)
                    if outside == "Chinese":
                        value = "Chinese"
                    elif outside == "Ilonggo":
                        value = "Ilonggo, Hiligaynon"
                    else:
                        value = ", ".join(inside_list)
                if "," in value:
                    languages = [e.strip() for e in value.split(",")]
                    parsed_data["primary_langauge"] = languages[0]
                    parsed_data["spoken_languages"] = languages
                    parsed_data["multiple_language"] = True
                else:
                    parsed_data["primary_langauge"] = value
                    parsed_data["spoken_languages"] = value
                    parsed_data["multiple_language"] = False
            elif key == "CATEGORY":
                if "," not in value:
                    parsed_data["category"] = value.lower()
            else:
                status, left_process_line = left_shrink(line, permitted_keys)
                if not status:
                    break
                current_key, current_values = None, []
                for segment in left_process_line.split(","):
                    segment = segment.strip()
                    if ":" in segment:
                        if current_key and current_key in permitted_keys:
                            parsed_data[current_key] = current_values
                        current_key, value = segment.split(":", 1)
                        current_key = current_key.strip().lower()
                        current_values = [value.strip()]
                    else:
                        current_values.append(segment)
                if current_key:
                    parsed_data[current_key] = current_values
    return parsed_data


def sanity_check(d):
    if "primary_langauge" not in d:
        return -1
    category = d.get("category", None)
    if not category:
        return -1
    keys_mapping = {
        "general": ["subject", "conversation_type"],
        "math": ["math_topic", "difficulty_level"],
        "coding": ["programming_language", "task_type"],
    }
    if category not in keys_mapping:
        return 0
    exist_keys = [k for k in keys_mapping[category] if k in d]
    return len(exist_keys)


def parse_generated(ex):
    generated = ex["generated"].strip().split("\n\n")[0]
    parsed = parse_response(generated, ex)
    sanity = sanity_check(parsed)
    return {"parsed": parsed, "sanity": sanity}


def wash_tag(e):
    return e.replace("-", " ").lower()


def parse_major_response(response):
    """Parse subject_mapping generated field: (A) xxx, (B) yyy -> [Humanities..., STEM...]"""
    ans_re = re.compile(r"\([A-E]\)")
    keys_mapping = {
        "A": "Humanities and Social Sciences",
        "B": "STEM (Science, Technology, Engineering, and Mathematics)",
        "C": "Business and Economics",
        "D": "Education and Personal Development",
        "E": "Health and Wellness",
    }
    matches = ans_re.findall(response)
    if not matches:
        return None
    ords = [m.strip("()") for m in matches]
    return [keys_mapping[o] for o in ords if o in keys_mapping]


def load_subject_mapping(mapping_file):
    """Load subject->major mapping from JSONL with {text, generated} format."""
    mapping = {}
    with Path(mapping_file).open("r") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            subj = d.get("text", d.get("subject", ""))
            generated = d.get("generated", "")
            majors = parse_major_response(generated)
            if majors:
                mapping[wash_tag(subj)] = majors
    return mapping


def do_save(data, folder, file_name):
    save_file = folder.joinpath(f"{file_name}-{len(data)}.jsonl")
    with save_file.open("w") as f:
        for l in tqdm(data):
            washed = {"dataset": l["dataset"], "id": l["id"], "messages": l["messages"]}
            f.write(f"{json.dumps(washed, indent=None, ensure_ascii=False)}\n")
    print(f"\t=> Save {save_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to GPT-tagged JSONL")
    parser.add_argument("--subject_mapping", type=str, required=True, help="Path to subject->major mapping JSONL")
    parser.add_argument("--output_dir", type=str, default="data/processed/tulu-tagged")
    args = parser.parse_args()

    # Load and parse
    collected = []
    for ex in tqdm(load_jsonl_iter(args.input), disable=False):
        parsed = parse_generated(ex)
        if parsed["sanity"] == 2:
            collected.append({**ex, "parsed": parsed})

    single_lang_data, multiple_lang_data = [], []
    for e in collected:
        if e["parsed"]["parsed"]["multiple_language"]:
            multiple_lang_data.append(e)
        else:
            single_lang_data.append(e)

    # Load subject mapping: JSONL with {"text": "fine_grained_subject", "generated": "(B) STEM ..."}
    mapping = load_subject_mapping(args.subject_mapping)

    major_languages = ["English"]
    single_lang_general_tagged = defaultdict(list)
    single_lang_coding_tagged = []
    single_lang_math_tagged = []
    single_lang_not_major = []

    for e in single_lang_data:
        P_category = e["parsed"]["parsed"]["category"]
        P_primary_lang = e["parsed"]["parsed"]["primary_langauge"]
        if P_primary_lang not in major_languages:
            single_lang_not_major.append(e)
        elif P_category == "general":
            all_subjects = [wash_tag(tag) for tag in e["parsed"]["parsed"]["subject"]]
            for tagged_subject in all_subjects:
                if tagged_subject in mapping:
                    for major_subject in mapping[tagged_subject]:
                        single_lang_general_tagged[major_subject].append(e)
        elif P_category == "coding":
            single_lang_coding_tagged.append(e)
        elif P_category == "math":
            single_lang_math_tagged.append(e)

    c1 = single_lang_general_tagged.get("Education and Personal Development", [])
    c2 = single_lang_general_tagged.get("Business and Economics", [])
    single_lang_general_tagged["Holistic Growth and Well-being"] = c1 + c2

    for k in single_lang_general_tagged:
        dedup = {e["id"]: e for e in single_lang_general_tagged[k]}
        single_lang_general_tagged[k] = list(dedup.values())

    naming_mapping = {
        "STEM (Science, Technology, Engineering, and Mathematics)": "g.tech",
        "Humanities and Social Sciences": "g.human",
        "Holistic Growth and Well-being": "g.growth",
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for k, v in single_lang_general_tagged.items():
        if k in naming_mapping and v:
            do_save(v, output_dir, naming_mapping[k])

    if single_lang_coding_tagged:
        do_save(single_lang_coding_tagged, output_dir, "coding")
    if single_lang_math_tagged:
        do_save(single_lang_math_tagged, output_dir, "math")
    multilingual_data = single_lang_not_major + multiple_lang_data
    if multilingual_data:
        do_save(multilingual_data, output_dir, "multilingual")
