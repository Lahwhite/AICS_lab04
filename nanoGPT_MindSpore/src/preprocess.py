"""
Preprocess datasets and transform dataset to mindrecord.
使用 ChineseCharTokenizer 处理中文数据集
"""

import argparse
import os

import numpy as np
from mindspore.mindrecord import FileWriter
from tqdm.auto import tqdm

from chinese_char_tokenizer import ChineseCharTokenizer
from utils import SEQ_LEN, read_jinyong


def chunks(lst, n):
    """yield n sized chunks from list"""
    for i in tqdm(range(len(lst) - n + 1)):
        yield lst[i : i + n]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="./dataset/mindrecord")
    parser.add_argument("--file_partition", type=int, default=1)
    parser.add_argument("--num_process", type=int, default=16)
    parser.add_argument(
        "--jinyong_dir",
        type=str,
        default="./data/jinyong",
        help="金庸小说目录，可以是单个文件或目录",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./dataset/chinese_char_tokenizer.json",
        help="ChineseCharTokenizer 模型路径",
    )

    args = parser.parse_args()

    out_dir, out_file = os.path.split(os.path.abspath(args.output_file))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    mindrecord_schema = {
        "input_ids": {"type": "int32", "shape": [-1]},
    }

    # 加载或训练 ChineseCharTokenizer
    if os.path.exists(args.tokenizer_path):
        print(f"加载 tokenizer: {args.tokenizer_path}")
        tokenizer = ChineseCharTokenizer.load(args.tokenizer_path)
    else:
        print("Tokenizer 不存在，请先运行: python data/prepare_jinyong.py")
        print("或者训练新的 tokenizer...")
        from chinese_char_tokenizer import train_from_jinyong

        tokenizer = train_from_jinyong(
            args.jinyong_dir, vocab_size=8000, save_path=args.tokenizer_path
        )

    print("\nTokenizer 信息:")
    for key, value in tokenizer.get_vocab_info().items():
        print(f"  {key}: {value}")

    transforms_count = 0
    wiki_writer = FileWriter(file_name=args.output_file, shard_num=args.file_partition)
    wiki_writer.add_schema(mindrecord_schema, "JinYong fictions")

    # 读取金庸数据
    print(f"\n读取数据: {args.jinyong_dir}")
    data = read_jinyong(args.jinyong_dir)
    print(f"总字符数: {len(data):,}")

    # 使用 ChineseCharTokenizer 编码
    print("编码文本...")
    tokens = tokenizer.encode(data)
    print(f"总 token 数: {len(tokens):,}")
    print(f"压缩比: {len(tokens) / len(data):.3f}")

    # 切分为固定长度的序列并写入 MindRecord
    print(f"\n生成序列 (SEQ_LEN={SEQ_LEN})...")
    for x in chunks(tokens, SEQ_LEN):
        transforms_count += 1
        sample = {
            "input_ids": np.array(x, dtype=np.int32),
        }
        wiki_writer.write_raw_data([sample])
    wiki_writer.commit()
    print("Transformed {} records.".format(transforms_count))

    out_file = args.output_file
    if args.file_partition > 1:
        out_file += "0"
    print("Transform finished, output files refer: {}".format(out_file))
