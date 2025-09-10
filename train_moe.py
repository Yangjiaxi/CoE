"""
Main training script for Sewing-MoE pipeline.
Supports: ffn (Stage 1 domain experts), moe-full, moe-rest, moe-router (Stage 2/3).
"""

import logging
import os
import pickle
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, HfArgumentParser

from factory import data_lut
from models_dev.modeling_moe_llama import MoeLlamaForCausalLM
from trainer_moe import (
    DataArguments,
    DeepSpeedTrainer,
    ModelArguments,
    MoeDeepSpeedTrainer,
    TrainingArguments,
    get_model_param_count,
    parameter_numel,
    smart_tokenizer_and_embedding_resize,
    encode_with_messages_format,
    encode_with_messages_format_chatml,
    encode_with_messages_format_chatml_full,
    add_system_prompt,
)

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

import warnings

warnings.filterwarnings(
    "ignore",
    message="Creating a tensor from a list of numpy.ndarrays is extremely slow.*",
    category=UserWarning,
    module="transformers.data.data_collator",
)


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    exp_run_folder = Path(training_args.output_dir)
    training_args.logging_dir = str(exp_run_folder.joinpath("logs").resolve())

    is_global_main_process = training_args.process_index == 0

    if is_global_main_process:
        exp_run_folder.mkdir(exist_ok=True, parents=True)
        raw_config = {**vars(model_args), **vars(data_args), **vars(training_args)}

        if training_args.exp_name is not None:
            exp_name = training_args.exp_name
        else:
            now = datetime.now()
            timestamp = now.strftime("%m-%d/%H-%M-%S")
            exp_name = f"Run-{timestamp}"

        if HAS_WANDB and training_args.report_to == "wandb":
            os.environ["WANDB_CACHE_DIR"] = str(exp_run_folder.resolve())
            wandb.init(name=exp_name, project="sewing-moe", group=training_args.group_name)

        with exp_run_folder.joinpath("args_config.pkl").open("wb") as f:
            pickle.dump(raw_config, f)

    if is_global_main_process:
        print(f"Load: {training_args.cache_dir} / {model_args.model_name_or_path}")

    # Expand data path abbreviations
    abbr_data_names = data_args.data_path
    expanded_data_files = []
    for name in abbr_data_names.split("-"):
        name = name.strip()
        expanded = data_lut.get(name, name)
        expanded_data_files.append(expanded)
    data_args.data_path = ",,".join(expanded_data_files)

    # Load model
    load_kwargs = dict(
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    if training_args.training_part in ["full", "ffn"]:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            local_files_only=True,
            **load_kwargs,
        )
    else:
        model = MoeLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            local_files_only=True,
            **load_kwargs,
        )

    model.config.use_cache = False

    if is_global_main_process:
        print(f"Model.dtype: {model.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        local_files_only=True,
        trust_remote_code=True,
    )

    # Chat template setup
    if data_args.data_format == "special_token":
        special_tokens = ["<|system|>", "<|user|>", "<|assistant|>"]
        if data_args.force_system_prompt:
            tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}\n{%- set ns = namespace(found=false) -%}{%- for message in messages -%}{%- if message.role == 'system' -%}{%- set ns.found = true -%}{%- endif -%}{%- endfor -%}\n{%- if not ns.found -%}{{'<|system|>' + '\n' + 'You are a helpful assistant.' + '\n'}}{%- endif %}\n{% for message in messages %}{% if message.role == \"system\" %}<|system|>\n{{ message.content.strip() }}\n{% elif message.role == \"user\" %}<|user|>\n{{ message.content.strip() }}\n{% elif message.role == \"assistant\" %}<|assistant|>\n{{ message.content.strip() }}{{ eos_token }}\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
        else:
            tokenizer.chat_template = '{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}\n{% for message in messages %}{% if message.role == "system" %}<|system|>\n{{ message.content.strip() }}\n{% elif message.role == "user" %}<|user|>\n{{ message.content.strip() }}\n{% elif message.role == "assistant" %}<|assistant|>\n{{ message.content.strip() }}{{ eos_token }}\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}'
    elif data_args.data_format.startswith("chatml"):
        special_tokens = ["<|im_start|>", "<|im_end|>"]
        if data_args.force_system_prompt:
            tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}\n{%- set ns = namespace(found=false) -%}{%- for message in messages -%}{%- if message.role == 'system' -%}{%- set ns.found = true -%}{%- endif -%}{%- endfor -%}\n{%- if not ns.found -%}{{'<|im_start|>system' + '\n' + 'You are a helpful assistant.<|im_end|>' + '\n'}}{%- endif %}\n{% for message in messages %}{{'<|im_start|>' + message.role + '\n' + message.content.strip() }}{{'<|im_end|>' + '\n'}}{% endfor %}\n{% if add_generation_prompt %}{{'<|im_start|>assistant' + '\n'}}{% endif %}"
        else:
            tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}\n{% for message in messages %}{{'<|im_start|>' + message.role + '\n' + message.content.strip() }}{{'<|im_end|>' + '\n'}}{% endfor %}\n{% if add_generation_prompt %}{{'<|im_start|>assistant' + '\n'}}{% endif %}"
    else:
        raise ValueError(f"Invalid {data_args.data_format = }")

    special_tokens_dict = {
        "pad_token": "<PAD>",
        "additional_special_tokens": special_tokens,
    }
    num_new_tokens = smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    if data_args.data_format == "special_token":
        chat_fmt_fn = encode_with_messages_format
        important_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<PAD>", tokenizer.eos_token]
    elif data_args.data_format == "chatml":
        chat_fmt_fn = encode_with_messages_format_chatml
        important_tokens = ["<|im_start|>", "<|im_end|>", "<PAD>"]
    elif data_args.data_format == "chatml_full":
        chat_fmt_fn = encode_with_messages_format_chatml_full
        important_tokens = ["<|im_start|>", "<|im_end|>", "<PAD>"]
    else:
        raise ValueError(f"Invalid {data_args.data_format = }")

    important_token_ids = [tokenizer.convert_tokens_to_ids(e) for e in important_tokens]

    meta_mapping = None
    if hasattr(model.config, "router_loss_type") and model.config.router_loss_type is not None:
        meta_mapping = {v: int(k) for k, v in model.config.ordinal_to_expert.items()}

    preprocess_fn = partial(
        chat_fmt_fn,
        tokenizer=tokenizer,
        max_seq_length=training_args.max_seq_length,
        meta_mapping=meta_mapping,
    )
    with training_args.main_process_first(desc="Processing conversational data...", local=False):
        data_list = data_args.data_path.split(",,")
        data_list = [e for e in data_list if e]
        if is_global_main_process:
            print(f"data_list: {data_list}")
        raw_dataset = load_dataset(
            "json",
            data_files={"train": data_list},
            cache_dir="./dataset_cache",
        )
        if data_args.force_system_prompt:
            raw_dataset = raw_dataset.map(
                add_system_prompt,
                batched=False,
                num_proc=os.cpu_count() // 2,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Add system prompt.",
            )

        lm_datasets = raw_dataset.map(
            preprocess_fn,
            batched=False,
            num_proc=os.cpu_count() // 2,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets.set_format(type="pt")
        lm_datasets = lm_datasets.filter(lambda example: (example["labels"] != -100).any())

        train_dataset = lm_datasets["train"]

    print(f"Process rank-{training_args.process_index}: Len={len(train_dataset)}")

    if training_args.training_part.startswith("moe"):
        trainer = MoeDeepSpeedTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        )
    else:
        trainer = DeepSpeedTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        )

    # Set trainable parameters
    if training_args.training_part == "full":
        print("Training mode: Full parameter finetuning.")
    elif training_args.training_part == "ffn":
        print("Training mode: MoE Experts FFN finetuning.")
        for name, param in model.named_parameters():
            if "mlp" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if num_new_tokens == 0:
            model.enable_input_require_grads()
        else:
            input_embeddings = model.get_input_embeddings().weight
            output_embeddings = model.get_output_embeddings().weight
            input_embeddings.requires_grad = True
            output_embeddings.requires_grad = True

            def create_tensor_mask_with_ids(num, the_tensor, new_token_ids):
                the_mask = torch.zeros_like(the_tensor)
                for new_token_id in new_token_ids:
                    the_mask[new_token_id] = 1.0

                def mask_apply(grad):
                    return grad * the_mask.to(grad.device)

                return mask_apply

            input_embeddings.register_hook(
                create_tensor_mask_with_ids(num_new_tokens, input_embeddings, important_token_ids)
            )
            output_embeddings.register_hook(
                create_tensor_mask_with_ids(num_new_tokens, output_embeddings, important_token_ids)
            )
    elif training_args.training_part.startswith("moe"):
        if training_args.training_part.endswith("full"):
            print("Fine-tuning all parameters!")
        elif training_args.training_part.endswith("router"):
            print("Training mode: MoE Routers (Gates) Finetuning.")
            for name, param in model.named_parameters():
                if "mlp.gate" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            model.enable_input_require_grads()
        elif training_args.training_part.endswith("rest"):
            print("Training mode: MoE **rest parameters** (emb, attn, router, lm_head) Finetuning.")
            for name, param in model.named_parameters():
                if "mlp" in name and "mlp.gate" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            raise ValueError(f"Unknown training part `{training_args.training_part}`")
    else:
        raise ValueError(f"Unknown training part `{training_args.training_part}`")

    if is_global_main_process:
        print(f"{get_model_param_count(model, trainable_only=True) = }")
        print(f"{get_model_param_count(model, trainable_only=False) = }")

    trainer.train()

    trainer.save_state()
    final_save_path = exp_run_folder.joinpath("final")
    trainer.save_model(output_dir=str(final_save_path.resolve()))

    if is_global_main_process:
        saved_locker = final_save_path.joinpath(".saved")
        with saved_locker.open("w") as f:
            f.write("saved")
        logging.info(f"\tMake: {saved_locker.resolve()}")

    logging.info("=" * 75)
    logging.info("All done!")
    logging.info("=" * 75)


if __name__ == "__main__":
    train()
