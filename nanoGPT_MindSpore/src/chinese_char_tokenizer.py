import json
import os
import pickle
from typing import Dict, List, Sequence


class ChineseCharTokenizer:
    """
    中文字符级 Tokenizer
    """

    def __init__(self, vocab_size: int = 8000):
        """
        Args:
            vocab_size: 词表大小，建议 5000-10000 覆盖常用汉字
        """
        self.vocab_size = vocab_size

        # 特殊 token
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        # 初始化特殊 token
        self.special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        ]

        # 字符到 ID 的映射
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

        # 特殊 token ID
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

        # 初始化特殊 token 映射
        for idx, token in enumerate(self.special_tokens):
            self.char_to_id[token] = idx
            self.id_to_char[idx] = token

        self.n_words = len(self.special_tokens)  # 会在 train 后更新
        self.stop_tokens = [self.bos_id, self.eos_id]

    @classmethod
    def train(cls, texts: str, vocab_size: int = 8000) -> "ChineseCharTokenizer":
        """
        从文本中构建词表

        Args:
            texts: 训练文本（可以是多个文本的拼接）
            vocab_size: 词表大小

        Returns:
            训练好的 tokenizer
        """
        tokenizer = cls(vocab_size)

        # 统计字符频率
        char_freq: Dict[str, int] = {}
        for char in texts:
            if char not in tokenizer.special_tokens:
                char_freq[char] = char_freq.get(char, 0) + 1

        # 按频率排序，取前 vocab_size - len(special_tokens) 个
        sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)

        # 保留的字符数量
        num_chars = min(vocab_size - len(tokenizer.special_tokens), len(sorted_chars))

        # 构建词表
        for i, (char, freq) in enumerate(sorted_chars[:num_chars]):
            token_id = len(tokenizer.special_tokens) + i
            tokenizer.char_to_id[char] = token_id
            tokenizer.id_to_char[token_id] = char

        tokenizer.n_words = len(tokenizer.char_to_id)

        print("词表构建完成:")
        print(f"  - 总字符数: {len(char_freq)}")
        print(f"  - 词表大小: {tokenizer.n_words}")
        print(f"  - 特殊 token: {len(tokenizer.special_tokens)}")
        print(f"  - 普通字符: {tokenizer.n_words - len(tokenizer.special_tokens)}")

        return tokenizer

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        """
        将文本编码为 token ID 列表

        Args:
            text: 输入文本
            bos: 是否添加开始 token
            eos: 是否添加结束 token

        Returns:
            token ID 列表
        """
        token_ids = []

        if bos:
            token_ids.append(self.bos_id)

        for char in text:
            token_id = self.char_to_id.get(char, self.unk_id)
            token_ids.append(token_id)

        if eos:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        """
        将 token ID 列表解码为文本

        Args:
            token_ids: token ID 列表

        Returns:
            解码后的文本
        """
        chars = []
        for token_id in token_ids:
            # 跳过特殊 token
            if token_id in [self.pad_id, self.bos_id, self.eos_id]:
                continue
            char = self.id_to_char.get(token_id, self.unk_token)
            chars.append(char)

        return "".join(chars)

    def save(self, save_path: str):
        """
        保存 tokenizer 到文件

        Args:
            save_path: 保存路径（.pkl 或 .json）
        """
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )

        tokenizer_data = {
            "vocab_size": self.vocab_size,
            "char_to_id": self.char_to_id,
            "id_to_char": {
                str(k): v for k, v in self.id_to_char.items()
            },  # JSON 需要字符串 key
            "n_words": self.n_words,
        }

        if save_path.endswith(".pkl"):
            with open(save_path, "wb") as f:
                pickle.dump(tokenizer_data, f)
        elif save_path.endswith(".json"):
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError("save_path must end with .pkl or .json")

        print(f"Tokenizer 已保存到: {save_path}")

    @classmethod
    def load(cls, load_path: str) -> "ChineseCharTokenizer":
        """
        从文件加载 tokenizer

        Args:
            load_path: 文件路径

        Returns:
            加载的 tokenizer
        """
        if load_path.endswith(".pkl"):
            with open(load_path, "rb") as f:
                tokenizer_data = pickle.load(f)
        elif load_path.endswith(".json"):
            with open(load_path, "r", encoding="utf-8") as f:
                tokenizer_data = json.load(f)
            # 转换 id_to_char 的 key 为 int
            tokenizer_data["id_to_char"] = {
                int(k): v for k, v in tokenizer_data["id_to_char"].items()
            }
        else:
            raise ValueError("load_path must end with .pkl or .json")

        tokenizer = cls(tokenizer_data["vocab_size"])
        tokenizer.char_to_id = tokenizer_data["char_to_id"]
        tokenizer.id_to_char = tokenizer_data["id_to_char"]
        tokenizer.n_words = tokenizer_data["n_words"]

        print(f"Tokenizer 已从 {load_path} 加载")
        print(f"  - 词表大小: {tokenizer.n_words}")

        return tokenizer

    def get_vocab_info(self) -> Dict:
        """获取词表信息"""
        return {
            "vocab_size": self.vocab_size,
            "actual_vocab_size": self.n_words,
            "special_tokens": len(self.special_tokens),
            "regular_chars": self.n_words - len(self.special_tokens),
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
        }


def train_from_jinyong(
    jinyong_dir: str, vocab_size: int = 8000, save_path: str = None
) -> ChineseCharTokenizer:
    """
    从金庸数据集训练 tokenizer

    Args:
        jinyong_dir: 金庸小说目录
        vocab_size: 词表大小
        save_path: 保存路径，None 则不保存

    Returns:
        训练好的 tokenizer
    """
    import glob

    print("从金庸数据集训练字符级 tokenizer...")
    print(f"数据目录: {jinyong_dir}")

    # 读取所有文本
    all_text = []
    txt_files = glob.glob(os.path.join(jinyong_dir, "*.txt"))

    print(f"找到 {len(txt_files)} 个文本文件")

    for txt_file in txt_files:
        print(f"  读取: {os.path.basename(txt_file)}")
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
            all_text.append(text)

    combined_text = "".join(all_text)
    print(f"\n总字符数: {len(combined_text):,}")

    # 训练 tokenizer
    tokenizer = ChineseCharTokenizer.train(combined_text, vocab_size)

    # 保存
    if save_path:
        tokenizer.save(save_path)

    return tokenizer


if __name__ == "__main__":
    # 示例：训练并测试 tokenizer
    jinyong_dir = "./data/jinyong"
    save_path = "./dataset/chinese_char_tokenizer.json"

    # 训练
    tokenizer = train_from_jinyong(jinyong_dir, vocab_size=8000, save_path=save_path)

    # 测试
    test_text = "天龙八部是金庸先生的武侠小说代表作之一"
    print("\n测试编码解码:")
    print(f"原文: {test_text}")

    encoded = tokenizer.encode(test_text, bos=True, eos=True)
    print(f"编码: {encoded}")
    print(f"Token 数量: {len(encoded)}")

    decoded = tokenizer.decode(encoded)
    print(f"解码: {decoded}")

    # 显示词表信息
    print("\n词表信息:")
    for key, value in tokenizer.get_vocab_info().items():
        print(f"  {key}: {value}")
