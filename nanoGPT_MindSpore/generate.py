import argparse

from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.chinese_char_tokenizer import ChineseCharTokenizer
from src.model import GPT, GPTConfig

VOCAB_SIZE = 5424

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=0)

# TODO: 根据需要调整生成过程所需的超参数（包括检查点路径），以及采样方式等。


def generate(
    prompt: str, tokenizer: ChineseCharTokenizer, model: GPT, max_gen_len: int = 200
):
    """
    Text generation
    """
    # TODO: 在所有 Attention 层中插入 KV 缓存，以避免 KV 的重复计算，加速推理
    pass


def continuation(tokenizer: ChineseCharTokenizer, model: GPT):
    """Using GPT for fiction continuation.

    Args:
        model (nn.Cell): GPT model
    """
    print(
        'Continuing the text in the style of Jin Yong\'s novels. Press "Ctrl+D" to'
        " exit."
    )
    while True:
        try:
            print("输入一个开头：", end="")
            prompt = input()

            generate(prompt, tokenizer, model)

        except EOFError:
            print("\nBye!")
            break


def main():
    parser = argparse.ArgumentParser(description="GPT inferencing")
    parser.add_argument(
        "--task_type", type=str, default="continuation", help="Evaluation task."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./GPT.ckpt",
        help="path of checkpoint file.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./dataset/chinese_char_tokenizer.json",
        help="ChineseCharTokenizer 模型路径",
    )

    args = parser.parse_args()
    task = args.task_type
    ckpt_path = args.ckpt_path

    gpt_config: GPTConfig = GPTConfig(vocab_size=VOCAB_SIZE)
    ckpt_dict = load_checkpoint(ckpt_path)

    model = GPT(gpt_config)

    model.set_train(False)
    load_param_into_net(model, ckpt_dict)

    tokenizer = ChineseCharTokenizer.load(args.tokenizer_path)

    if task == "continuation":
        continuation(tokenizer=tokenizer, model=model)


if __name__ == "__main__":
    main()
