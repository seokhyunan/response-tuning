from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
from transformers import enable_full_determinism
import os
import argparse
from peft import LoraConfig, prepare_model_for_kbit_training
from lora_merge import merge
import torch
import yaml

TULU_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3.1-8B')
    parser.add_argument('--dataset_path', type=str, default='datasets/train/lima.jsonl')
    parser.add_argument('--tuning_type', type=str, default='it')
    parser.add_argument('--config_dir', type=str, default='training_configs')
    parser.add_argument('--continue_training', action='store_true', default=False)
    parser.add_argument('--attn_implementation', type=str, default="flash_attention_2")
    parser.add_argument('--peft', type=str, default="qlora")
    args = parser.parse_args()

    if "gemma-2" in args.model:
        print("Gemma-2 does not support flash_attention_2. Using eager instead.")
        args.attn_implementation = "eager"

    return args

def yaml_loader(path):
    with open(path, 'r') as f:
        d = yaml.load(f, Loader=yaml.FullLoader)
    return d

def process_training_data(dataset, tokenizer, tuning_type='it'):
    def construct_chat(example, tokenizer=None, tuning_type='it', user_role='user', assistant_role='assistant'):
        if tuning_type == 'it':
            messages = [
                {"role": user_role, "content": example["prompt"]},
                {"role": assistant_role, "content": example["completion"]},
            ]
        elif tuning_type == 'rt':
            messages = [{"role": assistant_role, "content": example["completion"]}]
        else:
            raise NotImplementedError
        training_seq = tokenizer.apply_chat_template(messages, add_special_tokens=False, add_generation_prompt=False, tokenize=False)
        return {"training_seq": training_seq}
    dataset = dataset.map(construct_chat, fn_kwargs={"tokenizer": tokenizer, "tuning_type": tuning_type}).shuffle(seed=0)
    return dataset

def get_output_path(model_name, dataset_path, tuning_type, peft="qlora"):
    model_name = model_name.split('/')[-1]
    model_name += f"_{peft}" if peft else ""
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    save_path = os.path.join("outputs", "finetuned_models", model_name, tuning_type, dataset_name, "lora")
    os.makedirs(save_path, exist_ok=True)
    return save_path

def prepare_training(model_path, config_dir, output_path, attn_implementation="flash_attention_2", peft="qlora"):
    config_name = f"{peft}.yaml" if peft else "full.yaml"
    training_args = yaml_loader(os.path.join(config_dir, config_name))
    training_args["output_dir"] = output_path
    training_args = TrainingArguments(**training_args)
    peft_config = LoraConfig(**yaml_loader(os.path.join(config_dir, f"peft/{peft}.yaml"))) if peft else None
    if peft == "lora":
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, load_in_8bit=True, device_map="auto")
    elif peft == "qlora":
        bnb_config = BitsAndBytesConfig(**yaml_loader(os.path.join(config_dir, "peft/qlora_bnb.yaml")))
        model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
                device_map="auto",
                attn_implementation=attn_implementation,
            )
    else: 
        raise NotImplementedError
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.model_max_length = 2048
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.chat_template = TULU_CHAT_TEMPLATE
    model = prepare_model_for_kbit_training(model)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    return model, tokenizer, training_args, peft_config

def get_trainer(model, tokenizer, train_dataset, training_args, peft_config):
    response_template = "\n<|assistant|>\n"
    response_template = tokenizer.encode(response_template, add_special_tokens=False)[3:]

    collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template=response_template)
    trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="training_seq",
        data_collator=collator,
        args=training_args,
        packing=False,
        max_seq_length=2048,
        peft_config=peft_config,
    )

    return trainer

def main():
    args = parse_args()
    enable_full_determinism(0)

    output_path = get_output_path(args.model, args.dataset_path, args.tuning_type, args.peft)
    print(f"Saving model to {output_path}")
    model, tokenizer, training_args, peft_config = prepare_training(args.model, args.config_dir, output_path, attn_implementation=args.attn_implementation, peft=args.peft)
    train_dataset = Dataset.from_json(args.dataset_path)
    train_dataset = process_training_data(train_dataset, tokenizer, tuning_type=args.tuning_type)
    trainer = get_trainer(model, tokenizer, train_dataset, training_args, peft_config)

    trainer.train(resume_from_checkpoint=args.continue_training)
    trainer.save_state()
    trainer.save_model()
    tokenizer.save_pretrained(output_path)

    merge(output_path, attn_implementation=args.attn_implementation, save_path=os.path.dirname(output_path))

if __name__ == '__main__':
    main()