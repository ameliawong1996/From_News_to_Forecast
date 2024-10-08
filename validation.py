import os
import sys

import argparse
import torch
import json
from tqdm import tqdm
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "",
    validation_data: str = "",
    save_file: str = "",
    test_number: int = 300
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (base_model), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=1024,
        stream_output=False,
        max_tokens=2600,
        **kwargs,
    ):
        input_before_trunc = tokenizer(input, return_tensors="pt")
        input_after_trunc = tokenizer.decode(input_before_trunc['input_ids'].view(-1)[:max_tokens])[3:]
        prompt = prompter.generate_prompt(instruction, input_after_trunc)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)
    
    if validation_data:
        with open(validation_data, 'r', encoding='utf-8') as file:
            val_data = json.load(file)
        results = []
        if not save_file:
            save_file = f'{validation_data}-test-results.json'
        print(f"Start to validate {validation_data}, save results into {save_file}")
        if test_number == 0:
            pass
        elif test_number < len(val_data):
            val_data = val_data[:test_number]
        else:
            pass

        for data_point in tqdm(val_data):
            instruction_dp = data_point['instruction']
            input_dp = data_point['input']
            output = evaluate(instruction_dp, input_dp)
            results.append({'instruction': instruction_dp, 'input': input_dp, 'output': output})

            with open(save_file, 'w', encoding='utf-8') as file:
                json.dump(results, file, ensure_ascii=False, indent=4)

    else:
        while True:
            instruction = input("Input:")
            if len(instruction.strip()) == 0:
                break
            print("Response:", evaluate(instruction))


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_weights', default=None, type=str,
                        help="If None, perform inference on the base model")
    parser.add_argument('--val_data', default=None, type=str,
                        help="")
    parser.add_argument('--save', default=None, type=str,
                        help="If None, perform inference on the base model")
    parser.add_argument('--load_8bit', action='store_true',
                        help='only use CPU for inference')
    parser.add_argument('--prompter_name', default="alpaca", type=str,
                        help="alpaca for v3 and before, ts_test for v4 and after")
    args = parser.parse_args()
    main(args.load_8bit, args.base_model, args.lora_weights, args.prompter_name, args.val_data, args.save)


