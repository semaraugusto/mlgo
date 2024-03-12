from argparse import ArgumentParser
from dataclasses import replace
import torch

# Load model directly
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    LlamaForCausalLM,
    LlamaModel,
    LlamaConfig,
)


def get_model(hf_path: str):
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    print(model)
    return model


def print_tokenizer(hf_path: str):
    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    print(tokenizer)


def get_config(hf_path: str) -> LlamaConfig:
    config = AutoConfig.from_pretrained(hf_path)
    return config


def main():
    parser = ArgumentParser()
    parser.add_argument("--hf-path", help="Path for hugging face published weights")
    args = parser.parse_args()
    # hf_weights_path = "lgodwangl/new_01m"
    print(args)
    causal_model: LlamaForCausalLM = get_model(args.hf_path)
    # causal_model = print_tokenizer(args.hf_path)
    # model: LlamaModel = causal_model.model
    # file_path = args.hf_path.replace("/", "_")
    # torch.save(causal_model.state_dict(), f"{file_path}.pt")
    print("LlamaModel:::::::::::\n", causal_model)
    # config: LlamaConfig = get_config(args.hf_path)

    # print("LlamaConfig:::::::::::\n", model)
    # print(config)
    # print_tokenizer(args.hf_path)


if __name__ == "__main__":
    main()
